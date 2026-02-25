# cannot be included at configure time
include(BundleUtilities)

set(REQUIRED_ARGS APP_BUNDLE MVK_DYLIB MVK_ICD_JSON VULKAN_DYLIB)

foreach (arg ${REQUIRED_ARGS})
  if (NOT DEFINED ${arg})
    message(FATAL_ERROR "${arg} not defined")
  endif ()
endforeach ()

# fixup the app executable: create plist if it isn't there
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

# TODO remove
function(avk_verify_rpath_status)
  message(STATUS ----------------------------------------------------------)
  execute_process(
    COMMAND sh -c "otool -l ${APP_BUNDLE}/Contents/MacOS/avk-gui | grep -A4 LC_RPATH"
    COMMAND_ECHO STDOUT
    RESULT_VARIABLE RES
  )
  if (RES GREATER 1)
    message(FATAL_ERROR Wrong)
  endif ()
  message(STATUS ----------------------------------------------------------)
endfunction()

avk_verify_rpath_status()

# Create directories first
file(MAKE_DIRECTORY "${APP_BUNDLE}/Contents/Frameworks")
file(MAKE_DIRECTORY "${APP_BUNDLE}/Contents/Resources/vulkan/icd.d")
file(MAKE_DIRECTORY "${APP_BUNDLE}/Contents/Resources/vulkan/explicit_layer.d")

# if vulkan SDK, then copy validation layers
if (IS_DIRECTORY "$ENV{VULKAN_SDK}" AND USE_VULKAN_SDK)
  message(STATUS "Environment Variable For Vulkan SDK Found, inserting validation layers from there inside bundle")
  file(COPY "$ENV{VULKAN_SDK}/lib/libVkLayer_khronos_validation.dylib" DESTINATION "${APP_BUNDLE}/Contents/Frameworks" FOLLOW_SYMLINK_CHAIN)
  set(JQ_FILTER ".layer.library_path = \"../../../Frameworks/libVkLayer_khronos_validation.dylib\"")
  execute_process(
    COMMAND jq "${JQ_FILTER}" "$ENV{VULKAN_SDK}/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json"
    OUTPUT_FILE "${APP_BUNDLE}/Contents/Resources/vulkan/explicit_layer.d/VkLayer_khronos_validation.json"
    COMMAND_ECHO STDOUT
    COMMAND_ERROR_IS_FATAL ANY
  )
endif ()

# Copy the dylib
if (NOT DEFINED SCRIPT)
  file(COPY "${MVK_DYLIB}" DESTINATION "${APP_BUNDLE}/Contents/Frameworks" FOLLOW_SYMLINK_CHAIN)
  file(COPY "${VULKAN_DYLIB}" DESTINATION "${APP_BUNDLE}/Contents/Frameworks" FOLLOW_SYMLINK_CHAIN)
else ()
  # TODO: Note that volk expects "vulkan", not "Vulkan". We are fine though since it then loads "MoltenVK". Note that on a system with APFS case-insensitive the behaviour can change. Fix this later.
  execute_process(
    COMMAND ${SCRIPT} "${APP_BUNDLE}" "${MVK_DYLIB}" "${VULKAN_DYLIB}"
    COMMAND_ECHO STDOUT
    COMMAND_ERROR_IS_FATAL ANY
  )
endif ()

# now copy ICD JSON (Update library_path so that MoltenVK points to bundled dylib)
if (NOT DEFINED SCRIPT)
  set(JQ_FILTER ".ICD.library_path = \"../../../Frameworks/libMoltenVK.dylib\"")
else ()
  set(JQ_FILTER ".ICD.library_path = \"../../../Frameworks/MoltenVK.framework/MoltenVK\"")
endif ()

execute_process(
  COMMAND jq "${JQ_FILTER}" "${MVK_ICD_JSON}"
  OUTPUT_FILE "${APP_BUNDLE}/Contents/Resources/vulkan/icd.d/MoltenVK_icd.json"
  COMMAND_ECHO STDOUT
  COMMAND_ERROR_IS_FATAL ANY
)
# fixup_bundle arguments:
# 1. app bundle directory
# 2. list of non-system libraries to embed (empty)
# 3. search dirs for dependent libraries/frameworks (optional)
fixup_bundle("${APP_BUNDLE}" "" "${APP_BUNDLE}/Contents/Frameworks")
# redundant, fixup_bundle calls that
# verify_app("${APP_BUNDLE}")

# force the linker to embed an LC_RPATH unconditionally
# They are already present, kept for reference
# this is necessary because `fixup_bundle` removes any LC_RPATH entry from the
# Mach-O executable which thinks is outside of the bundle
execute_process(
  COMMAND install_name_tool -add_rpath @loader_path/../Frameworks "${APP_BUNDLE}/Contents/MacOS/avk-gui"
  COMMAND_ERROR_IS_FATAL ANY
  COMMAND_ECHO STDOUT
)

# disappeared!
avk_verify_rpath_status()

# 3 code sign (- for developer, you should put your apple developer certificate)
message(STATUS "Code-signing framework and app (ad-hoc)...")
execute_process(
  # then sign app bundle
  COMMAND codesign --force --deep --sign - "${APP_BUNDLE}"
)
