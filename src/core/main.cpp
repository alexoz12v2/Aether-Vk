#include <atomic>
#include <boost/fiber/all.hpp>
#include <boost/fiber/operations.hpp>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "fiber/mpmc.h"

// ---- types ----
class Scheduler;
using JobId = uint64_t;
using JobFunc = /*extern "C"*/ void (*)(void* userData, uint32_t jobIndex);

// ---- small utility hash for job names ----
inline JobId makeJobId(const char* name) {
  // Simple 64-bit FNV-1a
  // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
  uint64_t h = 14695981039346656037ULL;
  while (*name) {
    h ^= (uint8_t)(*name++);
    h *= 1099511628211ULL;
  }
  return h ? h : 1;  // avoid zero id
}

class Scheduler {
 public:
  struct JobEntry {
    JobFunc fn = nullptr;
    void* userData = nullptr;
    JobId id = 0;

    std::atomic<bool> valid{false};
    std::atomic<bool> done{false};
    std::condition_variable_any done_cv;
    std::mutex done_mtx;

#ifdef AVK_DEBUG
    const char* debugName = nullptr;
#endif

    JobEntry() = default;
    JobEntry(JobEntry&& other) noexcept
        : fn(std::exchange(other.fn, nullptr)),
          userData(std::exchange(other.userData, nullptr)),
          id(std::exchange(other.id, 0)) {
      valid.store(other.valid.load());
      done.store(other.done.load());
#ifdef AVK_DEBUG
      debugName = other.debugName;
#endif
    }
    JobEntry& operator=(JobEntry&& other) noexcept {
      if (this != &other) {
        fn = std::exchange(other.fn, nullptr);
        userData = std::exchange(other.userData, nullptr);
        id = std::exchange(other.id, 0);
        valid.store(other.valid.load());
        done.store(other.done.load());
#ifdef AVK_DEBUG
        debugName = other.debugName;
#endif
      }
      return *this;
    }
    JobEntry(const JobEntry&) = delete;
    JobEntry& operator=(const JobEntry&) = delete;
  };
  Scheduler(size_t jobPoolSize, size_t fiberCount,
            avk::MPMCQueue<uint32_t>& jobQueue,
            unsigned workerCount = std::thread::hardware_concurrency())
      : m_jobPoolSize(jobPoolSize),
        m_totalFibers(fiberCount),
        m_jobQueue(jobQueue),
        m_workerCount(workerCount ? workerCount : 1) {
    m_jobs.resize(jobPoolSize);
    // DO NOT create fibers here (they must be created on the thread that will
    // run them).
    m_nextJobIndex.store(0);
  }

  ~Scheduler() { shutdown(); }

  JobEntry* jobs(size_t* count = nullptr) {
    if (count) *count = m_jobs.size();
    return m_jobs.data();
  };

  // submit with optional debugName
  uint32_t submit(JobFunc fn, void* userData, JobId id = 0
#ifdef AVK_DEBUG
                  ,
                  const char* debugName = nullptr
#endif
  ) {
    uint32_t idx = m_nextJobIndex.fetch_add(1, std::memory_order_acq_rel);
    if (idx >= m_jobPoolSize) return UINT32_MAX;

    JobEntry& entry = m_jobs[idx];
    entry.fn = fn;
    entry.userData = userData;
    entry.id = id;
#ifdef AVK_DEBUG
    entry.debugName = debugName ? debugName : "Unknown";
#endif
    entry.valid.store(true, std::memory_order_release);
    entry.done.store(false, std::memory_order_release);

    while (!m_jobQueue.push(idx)) std::this_thread::yield();
    m_inflightJobs.fetch_add(1, std::memory_order_acq_rel);
    return idx;
  }

  // Access a string for the current job for debugging/logging
  std::string getJobName([[maybe_unused]] uint32_t jobIndex) const {
#ifdef AVK_DEBUG
    const char* name = m_jobs[jobIndex].debugName;
    return name ? std::string(name) : std::to_string(jobIndex);
#else
    return std::to_string(boost::this_fiber::get_id());
#endif
  }

  void start() {
    for (unsigned i = 0; i < m_workerCount; ++i) {
      m_threads.emplace_back([this, i]() { this->workerMain(i); });
    }
  }

  void shutdown() {
    // Request shutdown
    m_shutdownRequest.store(true, std::memory_order_release);

    // Wake all workers/fibers by pushing sentinel job indices (UINT32_MAX).
    // We need to push enough sentinels so every fiber that might be blocked in
    // pop() can see one and exit. We conservatively push m_totalFibers
    // sentinels.
    for (size_t s = 0; s < m_totalFibers; ++s) {
      while (!m_jobQueue.push(UINT32_MAX)) std::this_thread::yield();
    }

    // join worker threads
    for (auto& t : m_threads)
      if (t.joinable()) t.join();
    m_threads.clear();
  }

  void waitUntilAllJobsDone() {
    std::unique_lock<std::mutex> lk(m_allDoneMutex);
    m_allDoneCV.wait(lk, [this]() { return m_inflightJobs.load() == 0; });
  }

 private:
  // workerMain: runs on a worker thread. It creates 'fibersPerWorker'
  // boost::fibers::fiber objects inside this thread, so they run on this
  // thread's fiber scheduler.
  void workerMain(unsigned workerIndex) {
    std::cout << "[Thread " << std::this_thread::get_id() << "] Started"
              << std::endl;
    // Install a fiber scheduler on this thread
    boost::fibers::use_scheduling_algorithm<boost::fibers::algo::round_robin>();

    size_t fibersPerWorker = m_totalFibers / m_workerCount;
    if (workerIndex < (m_totalFibers % m_workerCount)) ++fibersPerWorker;

    std::vector<boost::fibers::fiber> localFibers;
    localFibers.reserve(fibersPerWorker);

    // Create fibers that run fiberLoop()
    for (size_t f = 0; f < fibersPerWorker; ++f) {
      localFibers.emplace_back([this] { fiberLoop(); });
    }

    // Keep the worker thread alive until shutdown
    while (!m_shutdownRequest.load(std::memory_order_acquire)) {
      boost::this_fiber::yield();  // let fibers run
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Join all fibers cleanly
    for (auto& fb : localFibers) {
      if (fb.joinable()) fb.join();
    }
  }

  // fiberLoop: executed inside each boost::fibers::fiber created in
  // workerMain()
  void fiberLoop() {
    std::cout << "[Fiber " << boost::this_fiber::get_id() << "] started"
              << std::endl;
    while (true) {
      uint32_t jobIndex = UINT32_MAX;
      if (!m_jobQueue.pop(jobIndex)) {
        // nothing available, yield the current fiber cooperatively
        boost::this_fiber::yield();
        // check shutdown and continue
        if (m_shutdownRequest.load(std::memory_order_acquire)) {
          // If shutdown was requested and queue empty, exit loop
          return;
        }
        continue;
      }
      std::cout << "[Fiber " << boost::this_fiber::get_id() << "] got job "
                << jobIndex << std::endl;

      // if sentinel -> exit
      if (jobIndex == UINT32_MAX) {
        // push sentinel back so other fibers can see it too (optional)
        // but not necessary because we pushed enough sentinels in shutdown.
        return;
      }

      // Execute job
      JobEntry& job = m_jobs[jobIndex];
      if (job.fn) job.fn(job.userData, jobIndex);

      // mark done and notify waiters
      job.done.store(true, std::memory_order_release);
      {
        std::unique_lock<std::mutex> lk(job.done_mtx);
        job.done_cv.notify_all();
      }

      // decrement inflight counter and possibly notify all-done
      if (m_inflightJobs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::unique_lock<std::mutex> lk(m_allDoneMutex);
        m_allDoneCV.notify_all();
      }

      // yield to give other fibers a chance
      boost::this_fiber::yield();
    }
  }

  std::vector<JobEntry> m_jobs;
  std::atomic<uint32_t> m_nextJobIndex{0};
  size_t m_jobPoolSize;
  size_t m_totalFibers;

  std::vector<boost::fibers::fiber> m_fibers;

  avk::MPMCQueue<uint32_t>& m_jobQueue;

  std::atomic<uint32_t> m_inflightJobs{0};
  std::mutex m_allDoneMutex;
  std::condition_variable m_allDoneCV;

  unsigned m_workerCount;
  std::vector<std::thread> m_threads;

  std::atomic<bool> m_shutdownRequest{false};
};

#ifdef AVK_DEBUG
#define JOB_NAME(sched, idx) (sched).getJobName(idx)
#define SUBMIT_JOB(sched, fn, userData, id, name) \
  (sched).submit(fn, userData, id, name)
#else
#define JOB_NAME(sched, idx) (std::to_string(boost::this_fiber::get_id()))
#define SUBMIT_JOB(sched, fn, userData, id, name) \
  (sched).submit(fn, userData, id)
#endif

struct Payload {
  Scheduler* sched;
  int value;
  uint32_t dependencyJobIndex;  // Job this depends on (UINT32_MAX if none)
};

// Mock math function: adds 1 to value, waits for dependency
void mathJob(void* userData, uint32_t jobIndex) {
  Payload* p = static_cast<Payload*>(userData);

  std::cout << "[Fiber " << JOB_NAME(*p->sched, jobIndex) << "] Job Started!"
            << std::endl;

  if (p->dependencyJobIndex != UINT32_MAX) {
    Scheduler::JobEntry* depJob = &p->sched->jobs()[p->dependencyJobIndex];
    while (!depJob->done.load(std::memory_order_acquire)) {
      boost::this_fiber::yield();
    }
  }

  p->value = p->value * 2 + 1;
  std::cout << "[Fiber " << JOB_NAME(*p->sched, jobIndex)
            << "] Job finished with value = " << p->value << std::endl;
}

int main() {
  constexpr size_t jobPoolSize = 1024;
  constexpr size_t fiberCount = 128;

  avk::MPMCQueue<uint32_t> jobQueue(jobPoolSize);
  std::cout << "Start: Created a job queue" << std::endl;

  {
    Scheduler sched(jobPoolSize, fiberCount, jobQueue, 1);
    std::cout << "Start: Created a Scheduler" << std::endl;
    sched.start();
    std::cout << "Start: Started Scheduler" << std::endl;

    std::vector<std::unique_ptr<Payload>> payloads(jobPoolSize);

    // Submit some jobs
    for (uint32_t i = 0; i < 10; ++i) {
      payloads[i] = std::make_unique<Payload>();
      payloads[i]->sched = &sched;
      payloads[i]->value = i;
      payloads[i]->dependencyJobIndex =
          (i == 0) ? UINT32_MAX : i - 1;  // chain dependencies

      uint32_t jobIndex =
          SUBMIT_JOB(sched, mathJob, payloads[i].get(), i, "MathJob");
      if (jobIndex == UINT32_MAX) {
        std::cerr << "Failed to submit job " << i << std::endl;
      } else {
        std::cout << "Submitted Job " << i << std::endl;
      }
    }

    // Wait for all jobs to complete
    std::cout << "Waiting..." << std::endl;
    sched.waitUntilAllJobsDone();
    std::cout << "All Done!" << std::endl;
  }

  std::cout
      << "Scheduler Destroyed! All jobs finished! (press anything to exit)"
      << std::endl;
  std::cin.get();
}