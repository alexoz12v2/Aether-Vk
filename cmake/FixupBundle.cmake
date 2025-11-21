# cannot be included at configure time
include(BundleUtilities)

if (NOT DEFINED APP_BUNDLE)
  message(FATAL_ERROR "APP_NUNDLE not defined")
endif ()

if (NOT DEFINED MVK_FRAMEWORK_SRC)
  message(FATAL_ERROR "MVK_FRAMEWORK_SRC not defined")
endif ()

set(MVK_FRAMEWORK_DST "${APP_BUNDLE}/Contents/Frameworks/MoltenVK.framework")

### NOTE: Not needed, fixup_bundle does that
# 1. copy MoltenVK.framework into the bundle
# file(MAKE_DIRECTORY "${APP_BUNDLE}/Contents/Frameworks")
# message(STATUS "Copying MoltenVK.framework into bundle...")
# file(COPY "${MVK_FRAMEWORK_SRC}" DESTINATION "${APP_BUNDLE}/Contents/Frameworks")

# 2. fixup the app executable: create plist if it isn't there
# Make sure the app has Info.plist
set(APP_INFO_PLIST "${APP_BUNDLE}/Contents/Info.plist")
file(MAKE_DIRECTORY "${APP_BUNDLE}/Contents")
if (NOT EXISTS "${APP_INFO_PLIST}")
  file(WRITE "${APP_INFO_PLIST}"
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
    "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
    "<plist version=\"1.0\">\n"
    "<dict>\n"
    "  <key>CFBundleIdentifier</key>\n"
    "  <string>org.aethervkproj.avk-gui</string>\n"
    "  <key>CFBundleVersion</key>\n"
    "  <string>1.0</string>\n"
    "  <key>CFBundleExecutable</key>\n"
    "  <string>avk-gui</string>\n"
    "  <key>CFBundlePackageType</key>\n"
    "  <string>APPL</string>\n"
    "</dict>\n"
    "</plist>\n"
  )
endif ()


# fixup_bundle arguments:
# 1. app bundle directory
# 2. list of non-system libraries to embed (empty)
# 3. search dirs for dependent libraries/frameworks (optional)
fixup_bundle("${APP_BUNDLE}" "" "${MVK_FRAMEWORK_SRC}")
# redundant, fixup_bundle calls that
# verify_app("${APP_BUNDLE}")

# 3 code sign (- for developer, you should put your apple developer certificate)
message(STATUS "Code-signing framework and app (ad-hoc)...")
execute_process(
  # sign inner binary first
  COMMAND codesign --force --deep --sign - "${MVK_FRAMEWORK_DST}/Versions/A/MoltenVK"
  # then sign framework
  COMMAND codesign --force --deep --sign - "${MVK_FRAMEWORK_DST}"
  # then sign app bundle
  COMMAND codesign --force --deep --sign - "${APP_BUNDLE}"
)
