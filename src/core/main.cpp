
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <vector>

// ---- externs: platform-specific, implemented in assembly per-ABI ----
extern "C" void* make_fcontext(void* stack_top, size_t stack_size,
                               void (*fn)(void*));
extern "C" intptr_t jump_fcontext(void* to_context, void* from_context_storage,
                                  intptr_t arg, int preserve_fpu);

// ---- types ----
class Scheduler;
using fcontext_t = void*;
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

// --- Vyukov bounded MPMC queue (template)
// https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
// ---
template <typename T>
class MPMCQueue {
 public:
  explicit MPMCQueue(size_t capacity_pow2)
      : m_buffer(nullptr),
        m_capacity(capacity_pow2),
        m_mask(capacity_pow2 - 1) {
    assert((capacity_pow2 & (capacity_pow2 - 1)) == 0);
    m_buffer = reinterpret_cast<Cell*>(
        ::operator new[](sizeof(Cell) * m_capacity, std::nothrow));
    assert(m_buffer);
    for (size_t i = 0; i < m_capacity; ++i) {
      new (&m_buffer[i]) Cell();
      m_buffer[i].seq.store(i, std::memory_order_relaxed);
    }
    m_enqueuePos.store(0);
    m_dequeuePos.store(0);
  }
  MPMCQueue(MPMCQueue const&) = delete;
  MPMCQueue(MPMCQueue&&) noexcept = delete;
  MPMCQueue& operator=(MPMCQueue const&) = delete;
  MPMCQueue& operator=(MPMCQueue&&) noexcept = delete;
  ~MPMCQueue() noexcept {
    for (size_t i = 0; i < m_capacity; ++i) {
      m_buffer[i].~Cell();
    }
    ::operator delete[](m_buffer);
  }

  bool push(const T& v) {
    Cell* cell = nullptr;
    size_t pos = m_enqueuePos.load(std::memory_order_relaxed);
    for (;;) {
      cell = &m_buffer[pos & m_mask];
      size_t const seq = cell->seq.load(std::memory_order_acquire);
      intptr_t const dif =
          static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
      if (dif == 0) {
        if (m_enqueuePos.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed))
          break;
      } else if (dif < 0) {
        return false;  // full
      } else {
        pos = m_enqueuePos.load(std::memory_order_relaxed);
      }
    }
    cell->value = v;
    cell->seq.store(pos + 1, std::memory_order_release);
    return true;
  }

  bool pop(T& out) {
    Cell* cell = nullptr;
    size_t pos = m_dequeuePos.load(std::memory_order_relaxed);
    for (;;) {
      cell = &m_buffer[pos & m_mask];
      size_t const seq = cell->seq.load(std::memory_order_acquire);
      intptr_t const dif =
          static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
      if (dif == 0) {
        if (m_dequeuePos.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed))
          break;
      } else if (dif < 0) {
        return false;  // empty
      } else {
        pos = m_dequeuePos.load(std::memory_order_relaxed);
      }
    }
    out = cell->value;
    cell->seq.store(pos + m_capacity, std::memory_order_release);
    return true;
  }

 private:
  struct Cell {
    std::atomic<size_t> seq;
    T value;
    Cell() : seq(0), value() {}
  };

  Cell* m_buffer;
  size_t m_capacity;
  size_t m_mask;
  std::atomic<size_t> m_enqueuePos;
  std::atomic<size_t> m_dequeuePos;
};

// ---- Job Representation ----
struct Job {
  JobId id = 0;
  JobFunc func = nullptr;
  void* userData = nullptr;
  // 0 = free/invalid, 1 = queued, 2 = running, 3= waiting, 4 = completed
  std::atomic<uint32_t> state{0};
  // support multiple waiters or repeated runs
  std::atomic<uint32_t> completionCount{0};

  Job() = default;
  Job(JobId id, JobFunc func, void* user)
      : id(id), func(func), userData(user) {}

  // Define Move Semantics (Required for a "shallow" transfer/pop-and-push)
  // Move Constructor
  Job(Job&& other) noexcept
      : id{other.id},
        func{other.func},
        userData{other.userData},
        // std::atomic is non-copyable but *can* be constructed from a temporary
        // or a moved value via its load() and store() methods or direct access
        // to its underlying value during move construction. The standard
        // library guarantees that a simple load/store transfer is a valid
        // "move" for atomic types.
        state{other.state.load()},
        completionCount{other.completionCount.load()} {
    // Standard practice for move source cleanup: Reset the moved-from object's
    // state so it's in a valid, but default-constructed/empty, state.
    // For a Job, setting state to 0 (free/invalid) is appropriate.
    other.state.store(0);
    other.completionCount.store(0);
  }

  // Move Assignment Operator
  Job& operator=(Job&& other) noexcept {
    if (this != &other) {
      // Transfer non-atomic members
      id = other.id;
      func = other.func;
      userData = other.userData;

      // Transfer atomic members via load/store
      state.store(other.state.load());
      completionCount.store(other.completionCount.load());

      // Cleanup the source object
      other.state.store(0);
      other.completionCount.store(0);
      other.id = 0;              // Optional: Also clear the ID
      other.func = nullptr;      // Optional: Also clear the function pointer
      other.userData = nullptr;  // Optional: Also clear the user data pointer
    }
    return *this;
  }
};

// Forward declaration
class Scheduler;

// ---- Fiber Representation ----
struct Fiber {
  Scheduler* scheduler = nullptr;
  fcontext_t ctx = nullptr;
  void* ctx_storage = nullptr;
  uint8_t* stack = nullptr;
  size_t stack_size = 0;
  std::atomic<int> state{0};  // 0=idle, 1=ready, 2=running, 3=waiting
  JobId currentJob = 0;
  void* scratch = nullptr;
};

// ---- External trampoline ----
extern "C" void fiberEntryTrampoline(void* arg);

// ---- Scheduler ----
class Scheduler {
 public:
  Scheduler(size_t numThreads = std::thread::hardware_concurrency(),
            size_t fibersPerThread = 16, size_t queueCapacity = 1024)
      : m_numThreads(numThreads),
        m_totalFibers(numThreads * fibersPerThread),
        m_jobQueue{queueCapacity},
        m_stopFlag(false) {
    m_fibers.reserve(m_totalFibers);
    for (size_t i = 0; i < m_totalFibers; ++i) {
      m_fibers.emplace_back(std::make_unique<Fiber>());
      m_fibers.back()->scheduler = this;
    }
    for (size_t i = 0; i < m_numThreads; ++i) {
      m_workers.emplace_back([this, i]() { workerMain(i); });
    }
  }

  ~Scheduler() {
    m_stopFlag.store(true);
    for (auto& w : m_workers)
      if (w.joinable()) w.join();
    for (auto& f : m_fibers) delete[] f->stack;
  }

  bool submitJob(JobFunc func, void* userData, const char* name) {
    JobId jid = makeJobId(name);
    Job j{jid, func, userData};
    {
      std::lock_guard<std::mutex> lk(m_jobStoreMutex);
      j.state = 1;  // QUEUED
      auto [it, ok] = m_jobStore.try_emplace(jid, std::move(j));
      if (!ok) return false;
    }
    m_jobQueue.push(jid);
    return true;
  }

  void fiberYield() {
    Fiber* self = s_tls.currentFiber;
    assert(self);

    self->state.store(1, std::memory_order_release);

    Fiber* next = nullptr;
    while (!(next = nextFiberForThread())) {
      std::this_thread::yield();
    }

    s_tls.currentFiber = next;

    if (!next->ctx) {
      next->stack = new uint8_t[DefaultStackSize];
      next->stack_size = DefaultStackSize;
      void* top = next->stack + next->stack_size;
      next->ctx = make_fcontext(top, next->stack_size, &fiberEntryTrampoline);
    }

    jump_fcontext(next->ctx, self->ctx_storage,
                  reinterpret_cast<intptr_t>(next), 1);
  }

  void fiberWaitFor(JobId jid) {
    Fiber* self = s_tls.currentFiber;
    assert(self);

    self->state.store(3, std::memory_order_release);
    self->currentJob = jid;

    {
      std::lock_guard<std::mutex> lk(m_waitMapMutex);
      m_waitMap.emplace(jid, self);
    }

    Fiber* next = nullptr;
    while (!(next = nextFiberForThread())) {
      std::this_thread::yield();
    }

    s_tls.currentFiber = next;

    if (!next->ctx) {
      next->stack = new uint8_t[DefaultStackSize];
      next->stack_size = DefaultStackSize;
      void* top = next->stack + next->stack_size;
      next->ctx = make_fcontext(top, next->stack_size, &fiberEntryTrampoline);
    }

    jump_fcontext(next->ctx, self->ctx_storage,
                  reinterpret_cast<intptr_t>(next), 1);
  }

 private:
  static constexpr size_t DefaultStackSize = 64 * 1024;

  size_t m_numThreads;
  size_t m_totalFibers;

  std::vector<std::unique_ptr<Fiber>> m_fibers;
  std::vector<std::thread> m_workers;

  std::mutex m_jobStoreMutex;
  std::unordered_map<JobId, Job> m_jobStore;

  std::mutex m_waitMapMutex;
  std::unordered_multimap<JobId, Fiber*> m_waitMap;

  MPMCQueue<JobId> m_jobQueue;
  std::atomic<bool> m_stopFlag;

  struct PerThread {
    size_t workerIndex = 0;
    Fiber* currentFiber = nullptr;
    size_t lastFiberIndex = 0;
  };
  static thread_local PerThread s_tls;

  Fiber* nextFiberForThread() {
    size_t total = m_fibers.size();
    size_t start = s_tls.lastFiberIndex;

    for (size_t offset = 1; offset <= total; ++offset) {
      size_t idx = (start + offset) % total;
      auto& f = *m_fibers[idx];
      int expected = f.state.load(std::memory_order_acquire);
      if (expected == 0 || expected == 1) {
        if (f.state.compare_exchange_strong(expected, 2,
                                            std::memory_order_acq_rel)) {
          s_tls.lastFiberIndex = idx;
          if (!f.currentJob) {
            JobId jid;
            if (m_jobQueue.pop(jid)) f.currentJob = jid;
          }
          return &f;
        }
      }
    }
    return nullptr;
  }

  void workerMain(size_t workerIndex) {
    s_tls.workerIndex = workerIndex;
    s_tls.currentFiber = nullptr;
    s_tls.lastFiberIndex = 0;

    while (!m_stopFlag.load()) {
      Fiber* next = nextFiberForThread();
      if (!next) {
        std::this_thread::yield();
        continue;
      }

      s_tls.currentFiber = next;

      if (!next->ctx) {
        next->stack = new uint8_t[DefaultStackSize];
        next->stack_size = DefaultStackSize;
        void* top = next->stack + next->stack_size;
        next->ctx = make_fcontext(top, next->stack_size, &fiberEntryTrampoline);
        next->scheduler = this;
      }

      jump_fcontext(next->ctx, nullptr, reinterpret_cast<intptr_t>(next), 1);
    }
  }

  friend void fiberEntryTrampoline(void* arg);
};

// TLS
thread_local Scheduler::PerThread Scheduler::s_tls;

// ---- Fiber trampoline ----
extern "C" void fiberEntryTrampoline(void* arg) {
  Fiber* f = reinterpret_cast<Fiber*>(arg);
  Scheduler* sched = f->scheduler;
  while (true) {
    if (!f->currentJob) {
      sched->fiberYield();
      continue;
    }

    Job* j = nullptr;
    {
      std::lock_guard<std::mutex> lk(sched->m_jobStoreMutex);
      auto it = sched->m_jobStore.find(f->currentJob);
      if (it != sched->m_jobStore.end()) {
        j = &it->second;
        it->second.state.store(2, std::memory_order_release);
      } else {
        f->currentJob = 0;
        sched->fiberYield();
        continue;
      }
    }

    j->func(j->userData);

    {
      std::lock_guard<std::mutex> lk(sched->m_jobStoreMutex);
      auto it = sched->m_jobStore.find(j->id);
      if (it != sched->m_jobStore.end()) {
        it->second.state.store(4, std::memory_order_release);
        it->second.completionCount.fetch_add(1, std::memory_order_relaxed);
      }
    }

    {
      std::lock_guard<std::mutex> lk(sched->m_waitMapMutex);
      auto range = sched->m_waitMap.equal_range(j->id);
      for (auto itw = range.first; itw != range.second; ++itw) {
        itw->second->state.store(1, std::memory_order_release);
      }
      sched->m_waitMap.erase(range.first, range.second);
    }

    f->currentJob = 0;
    f->state.store(0, std::memory_order_release);
    sched->fiberYield();
  }
}

struct SharedData {
  int value{0};
  std::mutex coutMtx;
};

template <typename T>
struct JobData {
  Scheduler* sheduler;
  T val;
};

extern "C" void jobWrite(void* user) {
  SharedData* data = &reinterpret_cast<JobData<SharedData>*>(user)->val;
  Scheduler* sched = reinterpret_cast<JobData<SharedData>*>(user)->sheduler;
  {
    std::lock_guard<std::mutex> lk{data->coutMtx};
    std::cout << "[JobWrite] Writing 42\n";
  }

  data->value = 42;

  // simulate some work
  sched->fiberYield();

  {
    std::lock_guard<std::mutex> lk{data->coutMtx};
    std::cout << "[JobWrite] Done\n";
  }
}

extern "C" void jobRead(void* user) {
  SharedData* data = &reinterpret_cast<JobData<SharedData>*>(user)->val;
  Scheduler* sched = reinterpret_cast<JobData<SharedData>*>(user)->sheduler;

  // Wait until jobWrite completes
  JobId writeJobId = makeJobId("writeJob");
  {
    std::lock_guard<std::mutex> lk{data->coutMtx};
    std::cout << "[JobRead] Waiting for writeJob to complete\n";
  }
  sched->fiberWaitFor(writeJobId);  // this yields until writeJob finishes

  int val = data->value;
  {
    std::lock_guard<std::mutex> lk{data->coutMtx};
    std::cout << "[JobRead] Read value: " << val << "\n";
  }

  sched->fiberYield();

  {
    std::lock_guard<std::mutex> lk{data->coutMtx};
    std::cout << "[JobRead] Done\n";
  }
}

extern "C" void jobIncrement(void* user) {
  SharedData* data = &reinterpret_cast<JobData<SharedData>*>(user)->val;
  Scheduler* sched = reinterpret_cast<JobData<SharedData>*>(user)->sheduler;

  // Wait for writeJob as well
  JobId writeJobId = makeJobId("writeJob");
  sched->fiberWaitFor(writeJobId);

  data->value += 10;
  {
    std::lock_guard<std::mutex> lk{data->coutMtx};
    std::cout << "[JobIncrement] Added 10, new value: " << data->value << "\n";
  }
}

int main() {
  Scheduler sched(2, 8, 128);  // 2 threads, 8 fibers each

  SharedData shared;
  std::cout << "Starting Fiber Test: " << shared.value << "\n";

  // submit jobs
  sched.submitJob(jobWrite, &shared, "writeJob");
  sched.submitJob(jobRead, &shared, "readJob");
  sched.submitJob(jobIncrement, &shared, "incrementJob");

  // give the scheduler time to run fibers
  std::this_thread::sleep_for(std::chrono::seconds(2));

  std::cout << "Final value: " << shared.value << "\n";
  return 0;
}