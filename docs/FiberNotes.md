# Fiber Notes

## Fiber creation

- Allocate necessary memory for the data structure

  - (Fiber Count) Fibers struct,
  - (Fiber Count) Array of Bytes working as a stack
  - (Max Job Count) allocated in a Multi-Producer, Multi-Consumer Lock-Free FIFO Queue (mpmc)
  - (Thread Count) Thread local storage (which contains the **ABI Dependent Context**)

