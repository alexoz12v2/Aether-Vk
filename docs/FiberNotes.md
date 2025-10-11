# Fiber Notes

## Fiber creation

- Allocate necessary memory for the data structure

  - (Fiber Count) Fibers struct,
  - (Fiber Count) Array of Bytes working as a stack
  - (Max Job Count) allocated in a Multi-Producer, Multi-Consumer Lock-Free FIFO Queue (mpmc)
  - (Thread Count) Thread local storage (which contains the **ABI Dependent Context**)

## Result

I tried writing my own using x64 ABI on windows. I couldn't make it to work properly, because I incorrectly restored the stack
somehow, handling the stack pointer resulted in me trying to execute stack memory as code, resulting in a Data Execution Protection
Exception cause I didn't mark the page as Executable with VirtualProtect

Hence let me switch to Boost.Fiber

I'll Detail How that works in some ABIs inside the PDF document

| Task Type                             | Fiber/Thread suggestion                                                |
| ------------------------------------- | ---------------------------------------------------------------------- |
| Win32 message handling                | Dedicated thread (low-latency)                                         |
| Window management (`ShowWindowAsync`) | Fiber on UI thread OR async thread                                     |
| Particle physics / AI                 | Fibers on worker threads (many cores)                                  |
| Rendering command prep                | Fibers for small CPU jobs; final `vkQueueSubmit` on a dedicated thread |
| GPU synchronization                   | Dedicated thread or callbacks; do **not block fibers**                 |
