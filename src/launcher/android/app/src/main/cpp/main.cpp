
// android stuff
#include <game-activity/GameActivity.h>
#include <game-activity/GameActivityEvents.h>
#include <game-activity/GameActivityLog.h>
#include <game-activity/native_app_glue/android_native_app_glue.h>
#include <jni.h>

// std

// our stuff
#include "avk-android-app.h"
#include "avk-android-dbg-utils.h"
#include "os/android/avk-basic-logcat.h"

static avk::AndroidOut s_mlogiBufavk{"main.cpp", ANDROID_LOG_INFO};
static std::ostream s_mlogi{&s_mlogiBufavk};

static void onAppCmd([[maybe_unused]] struct android_app* app, int32_t cmd) {
  auto* appManager = reinterpret_cast<avk::AndroidApp*>(app->userData);
  switch (cmd) {
    case APP_CMD_INIT_WINDOW: {
      s_mlogi << "APP_CMD_INIT_WINDOW" << std::endl;
      appManager->onWindowInit();
      break;
    }
    case APP_CMD_TERM_WINDOW: {
      s_mlogi << "APP_CMD_TERM_WINDOW" << std::endl;
      break;
    }
    case APP_CMD_GAINED_FOCUS: {
      s_mlogi << "APP_CMD_GAINED_FOCUS" << std::endl;
      appManager->resumeRendering();
      break;
    }
    case APP_CMD_LOST_FOCUS: {
      s_mlogi << "APP_CMD_LOST_FOCUS" << std::endl;
      appManager->pauseRendering();
      break;
    }
    case APP_CMD_CONTENT_RECT_CHANGED: {
      s_mlogi << "APP_CMD_CONTENT_RECT_CHANGED: Request Swapchain Creation"
              << std::endl;
      [[maybe_unused]] int32_t const width =
          app->contentRect.right - app->contentRect.bottom;
      [[maybe_unused]] int32_t const height =
          app->contentRect.bottom - app->contentRect.top;
      // TODO request resize
      break;
    }
    case APP_CMD_RESUME:
      // TODO -> vkDeviceWaitIdle, recreate Swapchain
      appManager->resumeRendering();
    case APP_CMD_PAUSE:
      // TODO -> destroy swapchain/discard swapchain (or just stop rendering)
      appManager->pauseRendering();
    default:
      break;
  }
}

// Note: ACTION_MOVE batch together multiple pointer coordinates in a single
// event object. earlier coordinates are accessed with `.historicalX`, while
// current with `.x`
// Note: ACTION_DOWN should handle `buttonState` to see if you interacted with a
// View element
static void printMotionEvent(GameActivityMotionEvent const& motionEvent) {
  // extract masked action (used in switch case) and pointer index (which finger
  // is it)
  // assumes type is motion (this method is from base class)
  // int const type = AInputEvent_getType(&motionEvent)
  // source: joy or or screen (should not be a joystick)
  bool const isJoy = (AInputEvent_getSource(
                          reinterpret_cast<const AInputEvent*>(&motionEvent)) &
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

static void printKeyEvent(GameActivityKeyEvent const& keyEvent) {
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
static void handleInputEvents([[maybe_unused]] struct android_app* app,
                              android_input_buffer* input) {
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

// TODO see if stuff allocated in android_main is correctly
// destructed even if you exit from a android lifecycle
extern "C" void android_main(struct android_app* app) {
  // log something
  s_mlogi << "Hello Android World" << std::endl;

  // TODO should retore state?
  if (app->savedState != nullptr) {
    s_mlogi << "Preexisting State!" << std::endl;
  }

  // initialization code
  avk::AndroidApp application{app};

  // setup app commands callback
  app->userData = &application;
  app->onAppCmd = onAppCmd;

  while (!app->destroyRequested) {
    // 1. Poll Lifecycle Events
    // use poll once with -1 timeout if need to wait for update
    // and not render as fast as possible
    struct android_poll_source* source = nullptr;
    while (ALooper_pollOnce(0, nullptr, nullptr, (void**)&source) > 0) {
      if (source) source->process(app, source);
      if (app->destroyRequested) return;
    }

    // 2. Handle Buffered Input
    android_input_buffer* input = android_app_swap_input_buffers(app);
    if (input) {
      handleInputEvents(app, input);
      android_app_clear_key_events(input);
      android_app_clear_motion_events(input);
    }

    // 3. Update State
    // TODO

    // 4. Render
    // TODO
  }

  app->userData = nullptr;

  s_mlogi << "Detaching current thread from JNI" << std::endl;
}
