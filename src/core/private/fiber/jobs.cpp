#include "fiber/jobs.h"

#include <atomic>
#include <boost/fiber/operations.hpp>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace avk {
Job::Job(size_t initialCap) : m_continuations() {
  m_continuations.reserve(initialCap);
}

void Job::addDepencency(Job* job) {
  {
    std::unique_lock<std::shared_mutex> ul(m_continuationsMutex);
    job->m_continuations.push_back(this);
  }

  m_remainingDependencies.fetch_add(1, std::memory_order_acq_rel);
}

void Job::reset() {
  // 1) Ensure exclusive access to continuations, waiting for any readers to
  // finish.
  {
    std::unique_lock<std::shared_mutex> ul(m_continuationsMutex);
    // safe to clear now — no readers running that depend on previous storage
    m_continuations.clear();
    // after clear, leave this scope to release the shared_mutex
  }

  // 2) Reset dependency & done state under done-mutex (not strictly required to
  // hold both
  //    locks simultaneously, but we ensure done state changes are
  //    synchronized).
  {
    std::lock_guard<std::mutex> lk(m_doneMutex);
    // Ensure no dangling waiters: notify any possible waiters that job is
    // 'done' (This mirrors behavior of waitFor; caller should normally
    // guarantee they waited).
    m_done.store(true, std::memory_order_release);
    m_doneCV.notify_all();

    // Now set to fresh state for reuse
    m_done.store(false, std::memory_order_relaxed);
    m_remainingDependencies.store(0, std::memory_order_relaxed);
  }

  // 3) Reset public fields (safe now)
  fn = nullptr;
  data = nullptr;
  priority = JobPriority::Medium;
#ifdef AVK_DEBUG
  debugName.clear();
#endif
}

// ------------------------ Scheduler -----------------------------------------

Scheduler::Scheduler(size_t fiberCount, avk::MPMCQueue<Job*>* highP,
                     avk::MPMCQueue<Job*>* medP, avk::MPMCQueue<Job*>* lowP,
                     uint32_t workerCount)
    : m_totalFibers(fiberCount),
      m_workerCount(workerCount ? workerCount : 1),
      m_shutdownRequest(false) {
  m_queues[0] = highP;
  m_queues[1] = medP;
  m_queues[2] = lowP;
}

Scheduler::~Scheduler() { shutdown(); }

void Scheduler::start() {
  for (uint32_t i = 0; i < m_workerCount; ++i) {
    m_threads.emplace_back([this, i]() { workerMain(i); });
  }
}

void Scheduler::shutdown() {
  m_shutdownRequest.store(true, std::memory_order_release);
  // Push enough sentinels to wake all fibers
  for (size_t s = 0; s < m_totalFibers; ++s) {
    pushTask(reinterpret_cast<Job*>(Sentinel), JobPriority::High);
  }
  for (auto& t : m_threads)
    if (t.joinable()) t.join();
  m_threads.clear();
}

bool Scheduler::trySubmitTask(Job* task) {
  // push only if no deps
  if (task->m_remainingDependencies.load(std::memory_order_acquire) == 0) {
    return pushTask(task, task->priority);
  }
  return false;
}

void Scheduler::safeSubmitTask(Job* task) {
  bool const isFiber =
      boost::this_fiber::get_id() != boost::fibers::fiber::id();
  if (task &&
      task->m_remainingDependencies.load(std::memory_order_acquire) == 0) {
    while (!pushTask(task, task->priority)) {
      if (isFiber) {
        boost::this_fiber::yield();
      } else {
        std::this_thread::yield();
      }
    }
  }
}

void Scheduler::waitUntilAllTasksDone() {
  std::unique_lock<std::mutex> lk(m_doneMutex);
  m_doneCV.wait(lk, [this]() { return m_inflightTasks.load() == 0; });
}

bool Scheduler::pushTask(Job* task, JobPriority prio) {
  if (task && task != reinterpret_cast<Job*>(Sentinel)) {
    m_inflightTasks.fetch_add(1, std::memory_order_acq_rel);
    const size_t idx = static_cast<size_t>(prio);
    return m_queues[idx]->push(task);
  }
  return false;
}

Job* Scheduler::popTask() {
  for (int i = 0; i < 3; ++i) {
    Job* t = nullptr;
    if (m_queues[i]->pop(t)) {
      std::cout << "[popTask] popped task=" << t << " prio=" << i << std::endl;
      return t;
    }
  }
  return nullptr;
}

void Scheduler::workerMain(uint32_t threadIndex) {
  boost::fibers::use_scheduling_algorithm<boost::fibers::algo::round_robin>();
  size_t fibersPerWorker = m_totalFibers / m_workerCount;
  if (fibersPerWorker == 0) fibersPerWorker = 1;

  std::vector<boost::fibers::fiber> localFibers;
  localFibers.reserve(fibersPerWorker);

  for (size_t f = 0; f < fibersPerWorker; ++f) {
    localFibers.emplace_back([this, threadIndex, f] { fiberLoop(threadIndex, f); });
  }

  while (!m_shutdownRequest.load(std::memory_order_acquire)) {
    boost::this_fiber::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  for (auto& fb : localFibers)
    if (fb.joinable()) fb.join();
}

void Scheduler::waitFor(Job* job) {
  if (!job) return;
  // if complete return
  if (job->m_done.load(std::memory_order_acquire)) return;
  // if we are in a fiber yield
  if (boost::this_fiber::get_id() != boost::fibers::fiber::id()) {
    while (!job->m_done.load(std::memory_order_relaxed)) {
      boost::this_fiber::yield();
      if (job->m_done.load(std::memory_order_acquire)) {
        break;
      }
    }
  } else {
    // external thread -> block with condition variable
    std::unique_lock<std::mutex> lk(job->m_doneMutex);
    job->m_doneCV.wait(
        lk, [job]() { return job->m_done.load(std::memory_order_acquire); });
  }
}

void Scheduler::fiberLoop(uint32_t threadIndex, uint32_t fiberIndex) {
  std::vector<Job*> copy;
  copy.reserve(64);

  while (true) {
    Job* task = popTask();
    if (!task) {
      if (m_shutdownRequest.load(std::memory_order_acquire)) return;
      boost::this_fiber::yield();
      continue;
    }

    if (task == reinterpret_cast<Job*>(Sentinel)) return;  // sentinel

    std::cout << "[fiberLoop] running on thread " << std::this_thread::get_id()
              << " with task" << getTaskName(task) << std::endl;
    if (task->fn) task->fn(task->data, getTaskName(task), threadIndex, fiberIndex);

    // signal that the job is finished
    {
      std::lock_guard<std::mutex> lk(task->m_doneMutex);
      task->m_done.store(true, std::memory_order_release);
      task->m_doneCV.notify_all();
    }

    // trigger continuation jobs
    {
      std::shared_lock<std::shared_mutex> sl(task->m_continuationsMutex);
      if (!task->m_continuations.empty()) {
        copy = task->m_continuations;
      }
    }

    if (!copy.empty()) {
      for (auto* cont : copy) {
        // fetch_sub returns the previous value; if previous == 1 then we became
        // zero now.
        int prev = cont->m_remainingDependencies.fetch_sub(
            1, std::memory_order_acq_rel);
        if (prev == 1) {
          // now zero -> ready to run
          pushTask(cont, cont->priority);
        }
      }
      copy.clear();
    }

    if (m_inflightTasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::unique_lock<std::mutex> lk(m_doneMutex);
      m_doneCV.notify_all();
    }

    boost::this_fiber::yield();
  }
}

}  // namespace avk