
message(STATUS "Adding test targets...")

if (AVK_OS STREQUAL WINDOWS)
  add_subdirectory(win32)
endif ()
