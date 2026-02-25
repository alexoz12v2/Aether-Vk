
// android stuff
#include <game-activity/GameActivity.h>
#include <game-activity/GameActivityEvents.h>
#include <game-activity/GameActivityLog.h>
#include <game-activity/native_app_glue/android_native_app_glue.h>
#include <jni.h>
#include <unistd.h>

// std

// our stuff
#include "avk-android-app.h"
#include "avk-android-dbg-utils.h"
#include "os/android/avk-basic-logcat.h"

static avk::AndroidOut s_mlogiBufavk{"main.cpp", ANDROID_LOG_INFO};
static std::ostream s_mlogi{&s_mlogiBufavk};

static int isThreadAlive(pthread_t tid) {
  int const rc = pthread_kill(tid, 0);
  if (rc == ESRCH)
    return 0;  // thread doesn't exist
  else if (rc == 0)
    return 1;  // thread exists (might be running, might be paused)
  else
    return -1;  // other error
}

/// to be called after `app->signalStopRendering`
static void timedJoinThreads(avk::AndroidApp *app) {
  void *retval = nullptr;
  s_mlogi << "[main] still alive 1 " << std::endl;

  constexpr int timeout_ms = 20;
  constexpr int interval_ms = 5;      // check every 5ms
  app->signalStopRendering();
  {// timed kill render thread (cause join hanged on a old version)
    int waited = 0;

    bool ready = false;
    s_mlogi << "[main] still alive 2 " << std::endl;

    // Measure real time using CLOCK_MONOTONIC
    timespec start{}, now{};
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (waited < timeout_ms) {
      s_mlogi << "[main] still alive loop " << std::endl;
      // poll RT acknowledgement
      if (app->checkRTAck()) {
        ready = true;
        break;
      }
      s_mlogi << "[main] still alive BEFORE sleep" << std::endl;

      // sleep a little so we don’t burn CPU
      usleep(interval_ms * 1000);
      s_mlogi << "[main] still alive sleep" << std::endl;

      // recompute elapsed real time
      clock_gettime(CLOCK_MONOTONIC, &now);
      waited = (int) ((now.tv_sec - start.tv_sec) * 1000
          + (now.tv_nsec - start.tv_nsec) / 1000000);
    }
    s_mlogi << "[main] still alive 3" << std::endl;

    // If the worker signaled it's ready, try to join.
    s_mlogi << "[main] is render thread ready to die? " << ready << std::endl;
    if (ready) {
      int r = pthread_join(app->RenderThread, &retval);
      if (r != 0) {
        s_mlogi << "[main] failed to join render thread: " << strerror(r)
                << std::endl;
        pthread_kill(app->RenderThread, SIGTERM);
      } else {
        s_mlogi << "[main] render thread joined" << std::endl;
      }
    } else {
      s_mlogi << "[main] timeout waiting for ack, killing render thread"
              << std::endl;
      pthread_kill(app->RenderThread, SIGTERM);
    }
    app->RenderThread = 0;
  }
  // TODO check if this hangs
  app->signalStopUpdating();
  pthread_join(app->UpdateThread, &retval);
  app->UpdateThread = 0;

  s_mlogi << "[main] killed all threads" << std::endl;
}

// TODO move to posix path
static pthread_t createThreadOrExit(void *(*proc)(void *),
                                    void *__restrict arg) {
  pthread_t thread = 0;
  if (pthread_create(&thread, nullptr, proc, arg)) {
    s_mlogi << "Failed to create render thread!" << std::endl;
    avk::showErrorScreenAndExit("Failed to create Thread");
  } else {
    s_mlogi << "Render thread created." << std::endl;
  }
  return thread;
}

// TODO refector for all pthread platforms
static void *renderThreadFunc(void *arg) {
  auto *app = reinterpret_cast<avk::AndroidApp *>(arg);
  avk::ApplicationBase::RTmain(app);
  pthread_exit(nullptr);
  return nullptr;
}

static void *updateThreadFunc(void *arg) {
  auto *app = reinterpret_cast<avk::AndroidApp *>(arg);
  avk::ApplicationBase::UTmain(app);
  pthread_exit(nullptr);
  return nullptr;
}

// TODO on low memory
static void onAppCmd(android_app *app, int32_t cmd) {
  static bool destroyRequested = false;
  assert(app && !destroyRequested);
  if (!app->userData) {
    LOGW << "[AVK Activity Lifecycle] no user data, TID: " << std::hex
         << std::this_thread::get_id() << std::dec << std::endl;
    return;
  }
  auto *application =
      static_cast<avk::DelayedConstruct<avk::AndroidApp> *>(app->userData)->get();
  switch (cmd) {
    case APP_CMD_INIT_WINDOW: {
      s_mlogi << "APP_CMD_INIT_WINDOW" << std::endl;
      bool const firstInit = !application->windowInitializedOnce();
      application->onWindowInit();
      if (firstInit) {
        s_mlogi << "FIRST APP_CMD_INIT_WINDOW" << std::endl;
        application->RenderThread =
            createThreadOrExit(renderThreadFunc, application);
        application->UpdateThread =
            createThreadOrExit(updateThreadFunc, application);
      }
      break;
    }
    case APP_CMD_TERM_WINDOW: {
      s_mlogi << "APP_CMD_TERM_WINDOW" << std::endl;
      application->onSurfaceLost();
      break;
    }
    case APP_CMD_GAINED_FOCUS: {
      s_mlogi << "APP_CMD_GAINED_FOCUS" << std::endl;
      application->resumeRendering();
      break;
    }
    case APP_CMD_LOST_FOCUS: {
      s_mlogi << "APP_CMD_LOST_FOCUS" << std::endl;
      application->pauseRendering();
      break;
    }
    case APP_CMD_CONTENT_RECT_CHANGED:
    case APP_CMD_CONFIG_CHANGED:
    case APP_CMD_WINDOW_RESIZED: {
      s_mlogi << "[Activity Lifecycle] There's a resize here" << std::endl;
      break;
    }
    case APP_CMD_RESUME: {
      s_mlogi << "APP_CMD_RESUME" << std::endl;
      application->onRestoreState();
      if (application->windowInitializedOnce()) {
        if (isThreadAlive(application->RenderThread)) {
          s_mlogi << "Render Thread is alive" << std::endl;
        } else {
          s_mlogi << "Render Thread is Dead!" << std::endl;
        }
      } else {
        s_mlogi << "Resume, but window on Render thread never init"
                << std::endl;
      }
      break;
    }
    case APP_CMD_PAUSE: {
      s_mlogi << "APP_CMD_PAUSE" << std::endl;
      application->onSaveState();
      break;
    }
    case APP_CMD_DESTROY: {
      s_mlogi << "APP_CMD_DESTROY" << std::endl;
      // join render thread
      timedJoinThreads(application);
      // final clean
      application = nullptr;
      static_cast<avk::DelayedConstruct<avk::AndroidApp> *>(app->userData)->destroy();
      destroyRequested = true;
      break;
    }
    default:break;
  }
}

// Note: ACTION_MOVE batch together multiple pointer coordinates in a single
// event object. earlier coordinates are accessed with `.historicalX`, while
// current with `.x`
// Note: ACTION_DOWN should handle `buttonState` to see if you interacted with a
// View element
static void printMotionEvent(GameActivityMotionEvent const &motionEvent) {
  // extract masked action (used in switch case) and pointer index (which finger
  // is it)
  // assumes type is motion (this method is from base class)
  // int const type = AInputEvent_getType(&motionEvent)
  // source: joy or or screen (should not be a joystick)
  bool const isJoy = (AInputEvent_getSource(
      reinterpret_cast<const AInputEvent *>(&motionEvent)) &
      AINPUT_SOURCE_CLASS_MASK) == AINPUT_SOURCE_CLASS_JOYSTICK;
  if (isJoy) {
    s_mlogi << "Motion Event JOYSTICK, time: " << motionEvent.eventTime
            << std::endl;
    return;
  }

  int const actionMasked = motionEvent.action & AMOTION_EVENT_ACTION_MASK;
  // pointer index changes, pointer id stays the same
  int const ptrIndex =
      (motionEvent.action & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >>
                                                                     AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
  int32_t const pointerId = motionEvent.pointers[ptrIndex].id;
  // history size: number of previous positions + current
  int32_t const historySize = motionEvent.historySize;
  uint32_t const pointerCount = motionEvent.pointerCount;
  bool const onScreen = motionEvent.source == AINPUT_SOURCE_TOUCHSCREEN;

  s_mlogi << "Motion Event " << motionEvent.eventTime
          << " action: " << avk::dbg::stringFromMotionEventAction(actionMasked)
          << " pointer id " << pointerId << (onScreen ? "onScreen " : "")
          << " {";
  for (int32_t h = 0; h < historySize; ++h) {
    s_mlogi << "\n\t[" << h
            << "] t:" << motionEvent.historicalEventTimesMillis[h]
            << " ptr indices: {";
    for (uint32_t p = 0; p < pointerCount; ++p) {
      s_mlogi << "\n\t\tid: " << motionEvent.pointers[p].id << ": " << "( x: "
              << GameActivityMotionEvent_getHistoricalX(&motionEvent, p, h)
              << " y: "
              << GameActivityMotionEvent_getHistoricalY(&motionEvent, p, h)
              << " }";
    }
  }
  s_mlogi << std::endl;
}

static void printKeyEvent(GameActivityKeyEvent const &keyEvent) {
  // AInputEvent_getType should be AINPUT_EVENT_TYPE_KEY
  // meta state contains all possible modifiers and more
  bool const hasCtrl = keyEvent.metaState & AMETA_CTRL_ON;
  bool const hasAlt = keyEvent.metaState & AMETA_ALT_ON;
  bool const hasCaps = keyEvent.metaState & AMETA_CAPS_LOCK_ON;
  bool const hasShift = keyEvent.metaState & AMETA_SHIFT_ON;
  s_mlogi << "Key Event, time: " << keyEvent.eventTime
          << " source int: " << keyEvent.source;
  s_mlogi << "\n\t" << (hasCtrl ? "CTRL " : "") << (hasAlt ? "ALT " : "")
          << (hasCaps ? "CAPS_LOCK " : "") << (hasShift ? "SHIFT " : "")
          << avk::dbg::stringFromKeyCode(keyEvent.keyCode) << std::endl;
}

// <https://github.com/android/ndk-samples/blob/65e4ae2daea3d2b89e0e9bc095e35a29757bfaff/endless-tunnel/app/src/main/cpp/input_util.hpp#L31>
static void handleInputEvents([[maybe_unused]] struct android_app *app,
                              android_input_buffer *input) {
  // 1. Process Touch/Motion Events
  // <https://developer.android.com/games/agdk/add-touch-support>
  for (uint32_t i = 0; i < input->motionEventsCount; ++i) {
    printMotionEvent(input->motionEvents[i]);
  }
  // 2. Process Key Events
  // <https://developer.android.com/reference/android/view/KeyEvent>
  for (uint32_t i = 0; i < input->keyEventsCount; ++i) {
    printKeyEvent(input->keyEvents[i]);
  }
}

extern "C" void android_main(android_app *app) {
  // log something
  s_mlogi << "Hello Android World TID: " << std::hex
          << std::this_thread::get_id()
          << std::dec << std::endl;

  // TODO should retore state?
  if (app->savedState != nullptr) {
    s_mlogi << "Preexisting State!" << std::endl;
  }

  // attach current thread to JNI to get java objects (`jobject`)
  JNIEnv *jniEnv = nullptr;
  app->activity->vm->AttachCurrentThread(&jniEnv, nullptr);

  // initialization code: Object placed in DelayedConstruct
  // such that in can be on the stack while being constructed here and
  // destroyed on Activity Destroy
  avk::DelayedConstruct<avk::AndroidApp> application;
  application.create(app, jniEnv);
  jniEnv = nullptr; // managed by class

  // setup app commands callback
  app->userData = &application;
  app->onAppCmd = onAppCmd;
  // Wait for debugger in debug builds

  while (!app->destroyRequested) {
    // 1. Poll Lifecycle Events
    // use poll once with -1 timeout if need to wait for update
    // and not render as fast as possible
    struct android_poll_source *source = nullptr;
    while (ALooper_pollOnce(0, nullptr, nullptr, (void **) &source) > 0) {
      if (source) source->process(app, source); // cmd thread is main
      if (app->destroyRequested) break;
    }

    if (app->destroyRequested) { break; }

    // 2. Handle Buffered Input
    android_input_buffer *input = android_app_swap_input_buffers(app);
    if (input) {
      handleInputEvents(app, input); // input thread is main
      android_app_clear_key_events(input);
      android_app_clear_motion_events(input);
    }
  }

  // Here `application` was already destroyed

  app->userData = nullptr;
}
