#include <stdio.h>
#include <time.h>

int main(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  printf("MONOTOMIC: %ld.%09ld\n", ts.tv_sec, ts.tv_nsec);

  clock_gettime(CLOCK_REALTIME, &ts);
  printf("REALTIME: %ld.%09ld\n", ts.tv_sec, ts.tv_nsec);
}