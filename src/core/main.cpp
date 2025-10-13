#include <atomic>
#include <boost/fiber/all.hpp>
#include <boost/fiber/operations.hpp>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include "fiber/mpmc.h"

enum class JobPriority { High = 0, Medium, Low };

struct Job {
  friend class Scheduler;

 public:
  void (*fn)(void* data, std::string const& name) = nullptr;
  void* data = nullptr;
  JobPriority priority = JobPriority::Medium;
#ifdef AVK_DEBUG
  std::string debugName = {};
#endif

  Job(size_t initialCap = 16) : m_continuations() {
    m_continuations.reserve(initialCap);
  }

  void addDepencency(Job* job) {
    job->m_continuations.push_back(this);
    m_remainingDependencies.fetch_add(1, std::memory_order_acq_rel);
  }

 private:
  std::vector<Job*> m_continuations;
  std::atomic<int> m_remainingDependencies{0};
};

class Scheduler {
  static constexpr uintptr_t Sentinel = 1ULL;

 public:
  Scheduler(size_t fiberCount, avk::MPMCQueue<Job*>* highP,
            avk::MPMCQueue<Job*>* medP, avk::MPMCQueue<Job*>* lowP,
            unsigned workerCount = std::thread::hardware_concurrency())
      : m_totalFibers(fiberCount),
        m_workerCount(workerCount ? workerCount : 1),
        m_shutdownRequest(false) {
    m_queues[0] = highP;
    m_queues[1] = medP;
    m_queues[2] = lowP;
  }

  ~Scheduler() { shutdown(); }

  void start() {
    for (unsigned i = 0; i < m_workerCount; ++i) {
      m_threads.emplace_back([this]() { workerMain(); });
    }
  }

  void shutdown() {
    m_shutdownRequest.store(true, std::memory_order_release);
    // Push enough sentinels to wake all fibers
    for (size_t s = 0; s < m_totalFibers; ++s) {
      pushTask(reinterpret_cast<Job*>(Sentinel), JobPriority::High);
    }
    for (auto& t : m_threads)
      if (t.joinable()) t.join();
    m_threads.clear();
  }

  void submitTask(Job* task) {
    // push only if no deps
    if (task->m_remainingDependencies.load(std::memory_order_acquire) == 0) {
      pushTask(task, task->priority);
    }
  }

#ifdef AVK_DEBUG
  std::string getTaskName(Job* task) const {
    return task ? task->debugName : "Sentinel";
  }
#else
  std::string getTaskName(Task*) const {
    return std::to_string(boost::this_fiber::get_id());
  }
#endif

 private:
  void pushTask(Job* task, JobPriority prio) {
    if (task && task != reinterpret_cast<Job*>(Sentinel)) {
      m_inflightTasks.fetch_add(1, std::memory_order_acq_rel);
    }
    size_t const idx = static_cast<size_t>(prio);
    while (!m_queues[idx]->push(task)) {
      std::this_thread::yield();
    }
  }

  Job* popTask() {
    for (int i = 0; i < 3; ++i) {
      Job* t = nullptr;
      if (m_queues[i]->pop(t)) return t;
    }
    return nullptr;
  }

  void workerMain() {
    boost::fibers::use_scheduling_algorithm<boost::fibers::algo::round_robin>();
    size_t fibersPerWorker = m_totalFibers / m_workerCount;
    if (fibersPerWorker == 0) fibersPerWorker = 1;

    std::vector<boost::fibers::fiber> localFibers;
    localFibers.reserve(fibersPerWorker);

    for (size_t f = 0; f < fibersPerWorker; ++f) {
      localFibers.emplace_back([this] { fiberLoop(); });
    }

    while (!m_shutdownRequest.load(std::memory_order_acquire)) {
      boost::this_fiber::yield();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    for (auto& fb : localFibers)
      if (fb.joinable()) fb.join();
  }

  void fiberLoop() {
    while (true) {
      Job* task = popTask();
      if (!task) {
        if (m_shutdownRequest.load(std::memory_order_acquire)) return;
        boost::this_fiber::yield();
        continue;
      }

      if (task == reinterpret_cast<Job*>(Sentinel)) return;  // sentinel

      if (task->fn) task->fn(task->data, getTaskName(task));

      for (auto* cont : task->m_continuations) {
        // fetch_sub returns the previous value; if previous == 1 then we became
        // zero now.
        int prev = cont->m_remainingDependencies.fetch_sub(
            1, std::memory_order_acq_rel);
        if (prev == 1) {
          // now zero -> ready to run
          pushTask(cont, cont->priority);
        }
      }

      if (m_inflightTasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::unique_lock<std::mutex> lk(m_doneMutex);
        m_doneCV.notify_all();
      }

      boost::this_fiber::yield();
    }
  }

 public:
  void waitUntilAllTasksDone() {
    std::unique_lock<std::mutex> lk(m_doneMutex);
    m_doneCV.wait(lk, [this]() { return m_inflightTasks.load() == 0; });
  }

 private:
  size_t m_totalFibers;
  unsigned m_workerCount;
  std::vector<std::thread> m_threads;
  std::atomic<bool> m_shutdownRequest;

  std::atomic<int> m_inflightTasks{0};
  std::mutex m_doneMutex;
  std::condition_variable m_doneCV;

  avk::MPMCQueue<Job*>* m_queues[3];
};

#ifdef AVK_DEBUG
#define AVK_JOB(jobPtr, fnPtr, dataPtr, prio, nameStr) \
  do {                                                 \
    (jobPtr)->fn = (fnPtr);                            \
    (jobPtr)->data = (dataPtr);                        \
    (jobPtr)->priority = (prio);                       \
    (jobPtr)->debugName = (nameStr);                   \
  } while (0)
#else
#define AVK_JOB(jobPtr, fnPtr, dataPtr, prio, nameStr) \
  do {                                                 \
    (jobPtr)->fn = (fnPtr);                            \
    (jobPtr)->data = (dataPtr);                        \
    (jobPtr)->priority = (prio);                       \
  } while (0)
#endif

// ================================================================

struct Payload {
  int value;
};

void mathStep1(void* userData, std::string const& name) {
  auto* p = static_cast<Payload*>(userData);
  p->value *= 2;
  std::cout << "[Fiber " << name << "] Step1: doubled to " << p->value
            << std::endl;
}

void mathStep2(void* userData, std::string const& name) {
  auto* p = static_cast<Payload*>(userData);
  p->value += 1;
  std::cout << "[Fiber " << name << "] Step2: added one \xE2\x86\x92 "
            << p->value << std::endl;
}

// Build continuation chain: Step1 → Step2
Job* mathJobWithContinuations(Scheduler& sched, Payload* payload) {
  auto* jobs = new Job[2];
  auto* job1 = &jobs[0];
  job1->fn = mathStep1;
  job1->data = payload;
  job1->priority = JobPriority::Medium;
#ifdef AVK_DEBUG
  job1->debugName = "MathJobStep1";
#endif

  auto* job2 = &jobs[1];
  job2->fn = mathStep2;
  job2->data = payload;
  job2->priority = JobPriority::Low;
#ifdef AVK_DEBUG
  job2->debugName = "MathJobStep2";
#endif

  // job2 depends on job1
  job2->addDepencency(job1);
  sched.submitTask(job1);
  return jobs;
}

int main() {
  constexpr size_t jobPoolSize = 1024;
  constexpr size_t fiberCount = 128;

#if _WIN32
  // data
  SetConsoleOutputCP(CP_UTF8);
  // Enable buffering to prevent VS from chopping up UTF-8 byte sequences
  setvbuf(stdout, nullptr, _IOFBF, 1000);
#endif

  avk::MPMCQueue<Job*> highQ(jobPoolSize);
  avk::MPMCQueue<Job*> medQ(jobPoolSize);
  avk::MPMCQueue<Job*> lowQ(jobPoolSize);

  std::cout << "Start: Created job queues" << std::endl;

  {
    Scheduler sched(fiberCount, &highQ, &medQ, &lowQ, 1);
    std::cout << "Start: Created Scheduler" << std::endl;
    sched.start();
    std::cout << "Start: Started Scheduler" << std::endl;

    std::vector<std::unique_ptr<Payload>> payloads(10);

    for (uint32_t i = 0; i < 10; ++i) {
      payloads[i] = std::make_unique<Payload>();
      payloads[i]->value = i;
      mathJobWithContinuations(sched, payloads[i].get());
      std::cout << "Submitted math job chain " << i << std::endl;
    }

    std::cout << "Waiting..." << std::endl;
    sched.waitUntilAllTasksDone();
    std::cout << "All Done!" << std::endl;
  }

  std::cout
      << "Scheduler Destroyed! All jobs finished! (press any key to exit)\n";
  std::cin.get();
  return 0;
}
