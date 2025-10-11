#include <atomic>
#include <boost/context/continuation.hpp>
#include <boost/context/fiber.hpp>
#include <boost/context/fixedsize_stack.hpp>
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
using JobFunc = /*extern "C"*/ void (*)(void* userData);

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

// --- Scheduler class ---
class Scheduler {
 public:
  using JobId = uint64_t;
  using JobFunc = void (*)(void*);

  struct JobEntry {
    //
    // 1. Constructors and Destructor
    //

    // Default Constructor (Provided)
    JobEntry() = default;

    // Custom Move Constructor (Provided, slightly refined)
    JobEntry(JobEntry&& other) noexcept
        : fn(std::exchange(other.fn, nullptr)),
          userData(std::exchange(other.userData, nullptr)),
          id(std::exchange(other.id, 0)) {
      // Note: done_cv and done_mtx are implicitly moved/default-constructed.
      // For std::mutex and std::condition_variable_any, this is fine;
      // they are move-enabled in C++11/C++14 onwards by being trivial or
      // non-copyable.
      valid.store(other.valid);
      done.store(other.done);
    }

    // Destructor (Default is fine as members handle their own cleanup)
    ~JobEntry() = default;

    // 2. Move-Only Control (Key for std::vector)

    // Move Assignment Operator (Essential for move-only types in vector)
    JobEntry& operator=(JobEntry&& other) noexcept {
      if (this != &other) {
        // Move the trivial/atomic members
        fn = std::exchange(other.fn, nullptr);
        userData = std::exchange(other.userData, nullptr);
        id = std::exchange(other.id, 0);
        valid.store(other.valid);
        done.store(other.done);

        // Note: done_cv and done_mtx are *not* moved or assigned here.
        // C++ standard containers (like std::vector) will use the move
        // constructor followed by a destructor during operations like resize,
        // which is why the move constructor is the primary concern.
        // However, a complete move assignment is good practice.
        // std::mutex and std::condition_variable_any are *not* assignable
        // (even via move), so we leave them untouched.
        // This is generally acceptable for this use-case since you
        // typically only rely on the move constructor for container operations.
      }
      return *this;
    }

    // Explicitly delete Copy Constructor (Makes it move-only)
    JobEntry(const JobEntry& other) = delete;

    // Explicitly delete Copy Assignment Operator (Makes it move-only)
    JobEntry& operator=(const JobEntry& other) = delete;

    // 3. Member Variables
    JobFunc fn = nullptr;
    void* userData = nullptr;
    JobId id = 0;
    std::atomic<bool> valid{false};
    std::atomic<bool> done{false};
    std::condition_variable_any done_cv;
    std::mutex done_mtx;
  };

  // Fiber is placed (via placement-new) into reserved top-of-stack space.
  // It contains the continuation 'ctx' and runtime state.
  struct Fiber {
    boost::context::continuation ctx;  // created in placement-new via callcc
    Scheduler* sched = nullptr;
    JobEntry* currentJob = nullptr;
    bool waiting = false;
    JobId waitingFor = 0;

    // Keep the stack_context and allocator pointer so we can deallocate later
    boost::context::stack_context sctx;
    boost::context::fixedsize_stack salloc;

    // Constructor called by placement-new; create the continuation here.
    Fiber(void* sp, std::size_t size, boost::context::stack_context sctx_in,
          boost::context::fixedsize_stack&& salloc, Scheduler* S)
        : ctx(),
          sched(S),
          currentJob(nullptr),
          waiting(false),
          waitingFor(0),
          sctx(sctx_in),
          salloc(salloc) {
      // create captured continuation using the preallocated area which contains
      // *this*
      ctx = boost::context::callcc(  //
          std::allocator_arg, boost::context::preallocated(sp, size, sctx),
          salloc,
          [this](boost::context::continuation&& caller)
              -> boost::context::continuation {
            // set TLS so Scheduler::fiberYield() works
            Scheduler::tls_in_fiber = true;
            Scheduler::tls_fiber_yield = [&caller]() {
              caller = caller.resume();
            };

            // run per-fiber loop
            this->fiberLoop(std::move(caller));

            // clear TLS
            Scheduler::tls_fiber_yield = nullptr;
            Scheduler::tls_in_fiber = false;
            return std::move(caller);
          });
    }

    ~Fiber() = default;

    // fiber loop: job execution loop - runs **inside** the fiber stack/context
    void fiberLoop(boost::context::continuation&& caller) {
      // 'caller' is the scheduler continuation. To yield to scheduler we set
      // caller = caller.resume();
      for (;;) {
        // If shutdown requested, exit fiberLoop and allow continuation to
        // return
        if (sched->m_shutdown.load(std::memory_order_acquire)) break;

        if (!currentJob) {
          // no job assigned: yield back to scheduler and wait for assignment
          caller = caller.resume();
          continue;
        }

        // Execute the job (user function)
        JobEntry* e = currentJob;
        if (e->fn) e->fn(e->userData);

        // mark done & notify
        e->done.store(true, std::memory_order_release);
        {
          std::unique_lock<std::mutex> lk(e->done_mtx);
          e->done_cv.notify_all();
        }

        // decrement inflight count and notify if zero
        if (sched->m_inflightJobs.fetch_sub(1, std::memory_order_seq_cst) ==
            1) {
          std::unique_lock<std::mutex> lk(sched->m_all_done_mtx);
          sched->m_all_done_cv.notify_all();
        }

        e->valid.store(false, std::memory_order_release);

        // clear currentJob and yield back to scheduler (caller)
        currentJob = nullptr;
        caller = caller.resume();
      }
    }
  };

  // ctor: jobs[] (preallocated array), jobPoolCount, stacksBuffer (contiguous),
  // stackSize, fiberCount, jobQueue reference, workerCount
  Scheduler(JobEntry* jobs, size_t jobPoolCount, size_t stackSize,
            size_t fiberCount, avk::MPMCQueue<uint32_t>& jobQueue,
            unsigned workerCount = std::thread::hardware_concurrency())
      : m_jobs(jobs),
        m_jobPoolCount(jobPoolCount),
        m_stackSize(stackSize),
        m_fiberCount(fiberCount),
        m_workerCount(workerCount ? workerCount : 1),
        m_jobQueue(jobQueue) {
    assert(jobs && jobPoolCount > 0 && fiberCount > 0);
    // initialize job pool
    for (size_t i = 0; i < m_jobPoolCount; ++i) {
      m_jobs[i].valid.store(false, std::memory_order_relaxed);
      m_jobs[i].done.store(false, std::memory_order_relaxed);
    }

    // Prepare fibers: for each fiber, allocate a stack_context and
    // placement-new a Fiber at top
    m_fiberPtrs.resize(m_fiberCount, nullptr);

    // We'll use a local fixedsize_stack allocator object for each allocation,
    // because it carries any size info the implementation needs.
    for (size_t i = 0; i < m_fiberCount; ++i) {
      // create an allocator with the desired stack size:
      boost::context::fixedsize_stack salloc(m_stackSize);

      // allocate a stack_context using the allocator
      boost::context::stack_context sctx = salloc.allocate();

      // Reserve space for our Fiber object at top of stack (aligned)
      void* sp = static_cast<char*>(sctx.sp) -
                 static_cast<std::ptrdiff_t>(sizeof(Fiber));
      std::size_t usable_size = sctx.size - sizeof(Fiber);

      // placement-new the Fiber object into the reserved space (the object will
      // call callcc to create ctx)
      Fiber* f = new (sp) Fiber(sp, usable_size, sctx, std::move(salloc), this);

      // store pointer so scheduler can resume it
      m_fiberPtrs[i] = f;
      // We do NOT call salloc.deallocate(); the stack remains valid for the
      // lifetime of the fiber. Note: salloc is a temporary, but callcc captured
      // the needed allocator state internally.
    }

    m_freeJobIndex.store(0, std::memory_order_relaxed);
    m_nextFiberIndex.store(0, std::memory_order_relaxed);
  }

  ~Scheduler() {
    shutdown();

    for (size_t i = 0; i < m_fiberPtrs.size(); ++i) {
      Fiber* f = m_fiberPtrs[i];
      if (f) {
        // Save allocator + stack info before destroying Fiber (since destructor
        // is trivial)
        auto sctx = f->sctx;
        auto salloc = f->salloc;

        // Explicitly destroy the placement-new Fiber
        f->~Fiber();

        // Now it’s safe to free the stack memory
        try {
          salloc.deallocate(sctx);
        } catch (...) {
          // ignore
        }

        m_fiberPtrs[i] = nullptr;
      }
    }
  }

  // submit job; returns job-slot index or UINT32_MAX on failure
  uint32_t submit(JobFunc fn, void* userData, JobId id = 0) {
    uint32_t idx = m_freeJobIndex.fetch_add(1, std::memory_order_acq_rel);
    if (idx >= m_jobPoolCount) return UINT32_MAX;
    JobEntry& entry = m_jobs[idx];
    entry.fn = fn;
    entry.userData = userData;
    entry.id = id;
    entry.valid.store(true, std::memory_order_release);
    entry.done.store(false, std::memory_order_release);

    // push job index to queue (spin until succeeds)
    while (!m_jobQueue.push(idx)) std::this_thread::yield();

    m_inflightJobs.fetch_add(1, std::memory_order_acq_rel);
    return idx;
  }

  void start() {
    for (unsigned i = 0; i < m_workerCount; ++i) {
      m_threads.emplace_back([this]() { workerMain(); });
    }
  }

  void shutdown() {
    if (m_shutdown.exchange(true, std::memory_order_acq_rel)) return;

    // join threads
    for (auto& t : m_threads) {
      if (t.joinable()) t.join();
    }
    m_threads.clear();
  }

  void pause() { m_paused.store(true, std::memory_order_release); }
  void resume() {
    m_paused.store(false, std::memory_order_release);
    // no CV used for worker blocking in this simple example
  }

  void waitUntilAllJobsDone() {
    std::unique_lock<std::mutex> lk(m_all_done_mtx);
    m_all_done_cv.wait(lk, [this] {
      return m_inflightJobs.load(std::memory_order_acquire) == 0;
    });
  }

  // wait for all job slots with JobId id to complete
  void waitForJob(JobId id) {
    if (id == 0) return;
    if (!tls_in_fiber) {
      // owner thread: block on per-job condition variables
      for (;;) {
        bool found = false;
        for (size_t i = 0; i < m_jobPoolCount; ++i) {
          JobEntry& e = m_jobs[i];
          if (e.valid.load(std::memory_order_acquire) && e.id == id) {
            found = true;
            std::unique_lock<std::mutex> lk(e.done_mtx);
            e.done_cv.wait(
                lk, [&e] { return e.done.load(std::memory_order_acquire); });
            break;  // re-scan in case multiple slots with same id exist
          }
        }
        if (!found) break;
      }
    } else {
      // fiber: cooperative yield loop
      for (;;) {
        bool any = false;
        for (size_t i = 0; i < m_jobPoolCount; ++i) {
          JobEntry& e = m_jobs[i];
          if (e.valid.load(std::memory_order_acquire) && e.id == id) {
            any = true;
            break;
          }
        }
        if (!any) break;
        fiberYield();
      }
    }
  }

  // fiber-only cooperative yield
  static void fiberYield() {
    if (tls_fiber_yield) tls_fiber_yield();
  }

 private:
  // worker thread main loop
  void workerMain() {
    while (!m_shutdown.load(std::memory_order_acquire)) {
      if (m_paused.load(std::memory_order_acquire)) {
        std::this_thread::yield();
        continue;
      }

      uint32_t jobIndex = 0;
      if (!m_jobQueue.pop(jobIndex)) {
        // nothing to do
        std::this_thread::yield();
        continue;
      }

      if (jobIndex == UINT32_MAX) {
        // sentinel (not used here) - ignore
        continue;
      }

      // pick a fiber (round-robin)
      size_t fidx = m_nextFiberIndex.fetch_add(1, std::memory_order_relaxed) %
                    m_fiberPtrs.size();
      Fiber* f = m_fiberPtrs[fidx];
      assert(f);

      // assign job to fiber and resume it
      f->currentJob = &m_jobs[jobIndex];

      // resume fiber by calling its continuation
      // the resume() returns a continuation we can ignore; conventionally we
      // can write:
      f->ctx = f->ctx.resume();
      // after the fiber yields/finishes, control returns here and loop
      // continues
    }
  }

  JobEntry* m_jobs;
  size_t m_jobPoolCount;

  size_t m_stackSize;
  size_t m_fiberCount;

  unsigned m_workerCount;
  std::vector<std::thread> m_threads;

  std::atomic<bool> m_shutdown{false};
  std::atomic<bool> m_paused{false};

  std::atomic<uint32_t> m_inflightJobs{0};
  std::mutex m_all_done_mtx;
  std::condition_variable m_all_done_cv;

  std::atomic<uint32_t> m_freeJobIndex{0};

  // pointers to Fiber objects that live inside each fiber's stack top
  std::vector<Fiber*> m_fiberPtrs;

  std::atomic<uint32_t> m_nextFiberIndex{0};

  // job queue (user-provided)
  avk::MPMCQueue<uint32_t>& m_jobQueue;

  // TLS helpers for fiber yield
  static thread_local std::function<void()> tls_fiber_yield;
  static thread_local bool tls_in_fiber;
};

// thread_local init
thread_local std::function<void()> Scheduler::tls_fiber_yield = nullptr;
thread_local bool Scheduler::tls_in_fiber = false;

struct JobData {
  Scheduler* sched;
  double value;
  double result;
  JobData* jobValues;
};

void jobFunc(void* ptr) {
  JobData* data = reinterpret_cast<JobData*>(ptr);
  // simulate heavy computation
  std::this_thread::sleep_for(std::chrono::milliseconds(50 + rand() % 50));
  data->result = std::sqrt(data->value) * 3.1415;
  std::cout << "Job finished: input=" << data->value
            << " result=" << data->result << "\n";
};

int main() {
  constexpr size_t jobPoolSize = 1024;
  std::vector<Scheduler::JobEntry> jobs;
  jobs.reserve(jobPoolSize);
  for (size_t i = 0; i < jobs.capacity(); ++i) {
    jobs.emplace_back();
  }

  constexpr size_t queueSize = 1024;
  avk::MPMCQueue<uint32_t> jobQueue(queueSize);

  constexpr size_t fiberCount = 128;
  constexpr size_t stackSize = 256 * 1024;  // 256 KB per fiber
  // uint8_t* stackBuffer = reinterpret_cast<uint8_t*>(operator new( fiberCount
  // * stackSize, std::align_val_t(16)));

  std::unique_ptr<Scheduler> sched = std::make_unique<Scheduler>(
      jobs.data(), jobPoolSize, stackSize, fiberCount, jobQueue, 1);
  sched->start();

  // Simulate a graph of dependent jobs:
  //   A -> B -> C
  //   D -> E
  //   F -> G -> H -> I

  JobData jobValues[10];
  for (auto& j : jobValues) {
    j.sched = sched.get();
    j.jobValues = jobValues;
  }

  // Job functions
  // Submit independent jobs first
  jobValues[0].value = 10.0;  // A
  [[maybe_unused]] uint32_t jobA =
      sched->submit(jobFunc, &jobValues[0], makeJobId("A"));

  jobValues[3].value = 7.0;  // D
  [[maybe_unused]] uint32_t jobD =
      sched->submit(jobFunc, &jobValues[3], makeJobId("D"));

  jobValues[5].value = 20.0;  // F
  [[maybe_unused]] uint32_t jobF =
      sched->submit(jobFunc, &jobValues[5], makeJobId("F"));

  // Submit dependent jobs
  // B depends on A
  jobValues[1].value = 0.0;  // will compute after A
  [[maybe_unused]] uint32_t jobB = sched->submit(
      [](void* ptr) {
        JobData* data = reinterpret_cast<JobData*>(ptr);
        data->sched->waitForJob(makeJobId("A"));
        data->value = data->jobValues[0].result + 5.0;
        jobFunc(data);
      },
      &jobValues[1], makeJobId("B"));

  // C depends on B
  jobValues[2].value = 0.0;
  [[maybe_unused]] uint32_t jobC = sched->submit(
      [](void* ptr) {
        JobData* data = reinterpret_cast<JobData*>(ptr);
        data->sched->waitForJob(makeJobId("B"));
        data->value = data->jobValues[1].result + 2.0;
        jobFunc(data);
      },
      &jobValues[2], makeJobId("C"));

  // E depends on D
  jobValues[4].value = 0.0;
  [[maybe_unused]] uint32_t jobE = sched->submit(
      [](void* ptr) {
        JobData* data = reinterpret_cast<JobData*>(ptr);
        data->sched->waitForJob(makeJobId("D"));
        data->value = data->jobValues[3].result + 1.0;
        jobFunc(data);
      },
      &jobValues[4], makeJobId("E"));

  // G depends on F
  jobValues[6].value = 0.0;
  [[maybe_unused]] uint32_t jobG = sched->submit(
      [](void* ptr) {
        JobData* data = reinterpret_cast<JobData*>(ptr);
        data->sched->waitForJob(makeJobId("F"));
        data->value = data->jobValues[5].result * 2.0;
        jobFunc(data);
      },
      &jobValues[6], makeJobId("G"));

  // H depends on G
  jobValues[7].value = 0.0;
  [[maybe_unused]] uint32_t jobH = sched->submit(
      [](void* ptr) {
        JobData* data = reinterpret_cast<JobData*>(ptr);
        data->sched->waitForJob(makeJobId("G"));
        data->value = data->jobValues[6].result + 3.0;
        jobFunc(data);
      },
      &jobValues[7], makeJobId("H"));

  // I depends on H
  jobValues[8].value = 0.0;
  [[maybe_unused]] uint32_t jobI = sched->submit(
      [](void* ptr) {
        JobData* data = reinterpret_cast<JobData*>(ptr);
        data->sched->waitForJob(makeJobId("H"));
        data->value = data->jobValues[7].result - 1.0;
        jobFunc(data);
      },
      &jobValues[8], makeJobId("I"));

  // Wait for all jobs to finish
  sched->waitUntilAllJobsDone();
  std::cout << "All jobs finished!\n";

  sched->shutdown();
  return 0;
}