#include "fiber/jobs.h"

#include <atomic>
#include <boost/fiber/operations.hpp>
#include <mutex>

namespace avk {
Job::Job(size_t initialCap) : m_continuations() {
  m_continuations.reserve(initialCap);
}

void Job::addDepencency(Job* job) {
  job->m_continuations.push_back(this);
  m_remainingDependencies.fetch_add(1, std::memory_order_acq_rel);
}

// ------------------------ Scheduler -----------------------------------------

Scheduler::Scheduler(size_t fiberCount, avk::MPMCQueue<Job*>* highP,
                     avk::MPMCQueue<Job*>* medP, avk::MPMCQueue<Job*>* lowP,
                     unsigned workerCount)
    : m_totalFibers(fiberCount),
      m_workerCount(workerCount ? workerCount : 1),
      m_shutdownRequest(false) {
  m_queues[0] = highP;
  m_queues[1] = medP;
  m_queues[2] = lowP;
}

Scheduler::~Scheduler() { shutdown(); }

void Scheduler::start() {
  for (unsigned i = 0; i < m_workerCount; ++i) {
    m_threads.emplace_back([this]() { workerMain(); });
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

void Scheduler::submitTask(Job* task) {
  // push only if no deps
  if (task->m_remainingDependencies.load(std::memory_order_acquire) == 0) {
    pushTask(task, task->priority);
  }
}

void Scheduler::waitUntilAllTasksDone() {
  std::unique_lock<std::mutex> lk(m_doneMutex);
  m_doneCV.wait(lk, [this]() { return m_inflightTasks.load() == 0; });
}

void Scheduler::pushTask(Job* task, JobPriority prio) {
  std::cout << "------------------------------------------------" << std::endl;
  if (task && task != reinterpret_cast<Job*>(Sentinel)) {
    m_inflightTasks.fetch_add(1, std::memory_order_acq_rel);
  }
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

  const size_t idx = static_cast<size_t>(prio);
  // Keep trying until push succeeds; avoid busy spin by yielding
  while (true) {
  std::cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << std::endl;
    if (m_queues[idx]->push(task)) {
      std::cout << "[pushTask] pushed task=" << task << " prio=" << idx
                << " inflight=" << m_inflightTasks.load() << std::endl;
      return;
    }
    // queue full — yield CPU to other threads/fibers
    std::this_thread::yield();
  }
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

void Scheduler::workerMain() {
  boost::fibers::use_scheduling_algorithm<boost::fibers::algo::round_robin>();
  size_t fibersPerWorker = m_totalFibers / m_workerCount;
  if (fibersPerWorker == 0) fibersPerWorker = 1;

  std::vector<boost::fibers::fiber> localFibers;
  localFibers.reserve(fibersPerWorker);

  std::cout << "worker thread=" << std::this_thread::get_id()
            << " totalFibers=" << m_totalFibers
            << " workerCount=" << m_workerCount << std::endl;

  for (size_t f = 0; f < fibersPerWorker; ++f) {
    std::cout << "[fiber] created on thread " << std::this_thread::get_id()
              << std::endl;
    localFibers.emplace_back([this] { fiberLoop(); });
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

void Scheduler::fiberLoop() {
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
    if (task->fn) task->fn(task->data, getTaskName(task));

    // signal that the job is finished
    {
      std::lock_guard<std::mutex> lk(task->m_doneMutex);
      task->m_done.store(true, std::memory_order_release);
      task->m_doneCV.notify_all();
    }

    // trigger continuation jobs
    for (auto* cont : task->m_continuations) {
      // fetch_sub returns the previous value; if previous == 1 then we became
      // zero now.
      int prev =
          cont->m_remainingDependencies.fetch_sub(1, std::memory_order_acq_rel);
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

}  // namespace avk