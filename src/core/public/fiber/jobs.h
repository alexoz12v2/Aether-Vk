#pragma once
#include <atomic>
#include <boost/fiber/all.hpp>
#include <boost/fiber/operations.hpp>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <thread>

#include "fiber/mpmc.h"


namespace avk {

enum class JobPriority { High = 0, Medium, Low };

struct Job {
  friend class Scheduler;

 public:
  void (*fn)(void* data, std::string const& name, uint32_t threadIndex, uint32_t fiberIndex) = nullptr;
  void* data = nullptr;
  JobPriority priority = JobPriority::Medium;
#ifdef AVK_DEBUG
  std::string debugName = {};
#endif

  Job(size_t initialCap = 16);

  void addDepencency(Job* job);
  void reset();

 private:
  std::vector<Job*> m_continuations;
  std::atomic<int> m_remainingDependencies{0};

  // waitable state for scheduler's waitFor
  std::atomic_bool m_done = false;
  std::mutex m_doneMutex;
  std::condition_variable m_doneCV;

  // synchronize access to continuations
  mutable std::shared_mutex m_continuationsMutex;
};

class Scheduler {
  static constexpr uintptr_t Sentinel = 1ULL;

 public:
  Scheduler(size_t fiberCount, avk::MPMCQueue<Job*>* highP,
            avk::MPMCQueue<Job*>* medP, avk::MPMCQueue<Job*>* lowP,
            uint32_t workerCount = std::thread::hardware_concurrency());
  ~Scheduler();

  void start();
  void shutdown();
  bool trySubmitTask(Job* task);
  void safeSubmitTask(Job* task);
  inline uint32_t threadCount() const {
    return static_cast<uint32_t>(m_threads.size());
  }

#ifdef AVK_DEBUG
  inline std::string getTaskName(Job* task) const {
    return task ? task->debugName : "Sentinel";
  }
#else
  inline std::string getTaskName(Job*) const {
    return std::to_string(boost::this_fiber::get_id());
  }
#endif

  void waitFor(Job* job);
  void waitUntilAllTasksDone();

 private:
  bool pushTask(Job* task, JobPriority prio);
  Job* popTask();
  void workerMain(uint32_t threadIndex);
  void fiberLoop(uint32_t threadIndex, uint32_t fiberIndex);

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
    (jobPtr)->reset();                                 \
    (jobPtr)->fn = (fnPtr);                            \
    (jobPtr)->data = (dataPtr);                        \
    (jobPtr)->priority = (prio);                       \
    (jobPtr)->debugName = (nameStr);                   \
  } while (0)
#else
#define AVK_JOB(jobPtr, fnPtr, dataPtr, prio, nameStr) \
  do {                                                 \
    (jobPtr)->reset();                                 \
    (jobPtr)->fn = (fnPtr);                            \
    (jobPtr)->data = (dataPtr);                        \
    (jobPtr)->priority = (prio);                       \
  } while (0)
#endif

}  // namespace avk