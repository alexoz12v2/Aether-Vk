import com.android.build.api.dsl.Packaging
import org.jetbrains.kotlin.compose.compiler.gradle.ComposeFeatureFlag
import org.jetbrains.kotlin.gradle.fus.internal.isJenkins
import java.net.URI
import java.net.URL
import java.util.zip.ZipInputStream

///////////////////////////////////////////////////////////////////////////////////////////////////////

// Directory for JNI Libraries debug, exposed through a different
// source set configuration https://developer.android.com/build/build-variants
// the main source set is created by default by AGP and inserted into all build types
val jniLibsDir = layout.buildDirectory.dir("debug-jniLibs")

// --- Property configurable with -DvulkanValidationVersion on command line of from properties file ---
val vulkanValidationVersion =
  project.findProperty("vulkanValidationVersion")?.toString() ?: "1.4.309.0"
val abiList = listOf("arm64-v8a", "armeabi-v7a", "x86", "x86_64")

// === Derived URLs & paths ===
val validationBaseUrl = "https://github.com/KhronosGroup/Vulkan-ValidationLayers/releases/download"
val validationZipName = "android-binaries-$vulkanValidationVersion.zip"
val validationUrl = "$validationBaseUrl/vulkan-sdk-$vulkanValidationVersion/$validationZipName"

val validationDownloadDirectory =
  layout.buildDirectory.dir("downloaded-vulkan-validation-layers/$vulkanValidationVersion")
val validationZipFile = validationDownloadDirectory.map { it.file(validationZipName) }

// assuming version of clang in the NDK is 19 (WARNING)
val clangVersion =
  project.property("ndkClangVersion") ?: throw GradleException("Define DndkClangVersion={whatever}")
// Define mapping from ASAN Dynamic Library filename to ABI folders (WARNING: Assumes names are fixed)
val asanFiles = mapOf(
  "libclang_rt.asan-aarch64-android.so" to "arm64-v8a",
  "libclang_rt.asan-arm-android.so" to "armeabi-v7a",
  "libclang_rt.asan-i686-android.so" to "x86",
  "libclang_rt.asan-x86_64-android.so" to "x86_64"
)
val debugAsanJniLibsDir = layout.buildDirectory.dir("debug-asan/jniLibs")

///////////////////////////////////////////////////////////////////////////////////////////////////////

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.android)
  // necessary to compile @Composable
  alias(libs.plugins.compose.compiler)
  kotlin("kapt")
}

composeCompiler {
  // https://developer.android.com/develop/ui/compose/performance/stability
  reportsDestination = layout.buildDirectory.dir("compose_compiler")
  stabilityConfigurationFile = rootProject.layout.projectDirectory.file("stability_config.conf")
  // these reports have meaning only on a release build. Make sure to disable them on a prod build
  reportsDestination = layout.buildDirectory.dir("compose_compiler")
  metricsDestination = layout.buildDirectory.dir("compose_compiler")
  // enable strong skipping mode, for which all restartable composable
  // become skippable
  // https://kotlinlang.org/docs/compose-compiler-options.html#purpose-and-use-of-feature-flags
  featureFlags = setOf(ComposeFeatureFlag.StrongSkipping)
}

android {
  namespace = "org.aethervkproj.aethervk"
  compileSdk = 36

  // Debug Additional resources: Vulkan Validation Layers, HWAsan starter script for arm64-v8 abi
  // requires necessary configuration below on buildTypes
  sourceSets.getByName("debug") {
    jniLibs.srcDir(jniLibsDir)
    jniLibs.srcDir(debugAsanJniLibsDir)
  }

  defaultConfig {
    ndkVersion = "28.2.13676358"
    applicationId = "org.aethervkproj.aethervk"
    minSdk = 34 // Android 14
    targetSdk = 36
    versionCode = 1
    versionName = "1.0"

    // do not enable injection of external .so libraries (external vulkan validation layers eg. renderdoc)
    // by default
    manifestPlaceholders["injectLayersEnabled"] = "false"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    @Suppress("UnstableApiUsage")
    externalNativeBuild {
      cmake {
        // https://developer.android.com/ndk/guides/hwasan#cmake-gradle-kotlin
        arguments += "-DANDROID_STL=c++_shared"
        arguments += "-DANDROID_ARM_MODE=arm"
        // arguments += "-DANDROID_USE_LEGACY_TOOLCHAIN_FILE=OFF"
      }
    }
  }

  // https://developer.android.com/build/build-variants
  buildTypes {
    // https://developer.android.com/studio/debug
    debug {
      // necessary for wrap.sh and remote debugging
      isDebuggable = true
      isJniDebuggable = true
      manifestPlaceholders["injectLayersEnabled"] = "true"
      // applicationIdSuffix = ".debug"

      // the wrap script needs legacy packaging
      packaging {
        jniLibs {
          useLegacyPackaging = true
        }
      }

      ndk {
        isDebuggable = true
        isJniDebuggable = true
        debugSymbolLevel = "FULL"
      }
      @Suppress("UnstableApiUsage")
      externalNativeBuild {
        cmake {
          // https://developer.android.com/ndk/guides/hwasan#cmake-gradle-kotlin
          // arguments += "-DANDROID_SANITIZE=hwaddress"
          // ^^ Added from cmake cause hwaddress is supported only on arm64-v8a
          // hence rollback to old address sanitizer on other
          arguments += "-DAVK_USE_SANITIZERS=ON"
        }
      }
    }
    release {
      isDebuggable = false
      isMinifyEnabled = true
      proguardFiles(
        getDefaultProguardFile("proguard-android-optimize.txt"),
        "proguard-rules.pro"
      )
    }
    create("RelWithDbg") {
      initWith(getByName("release"))
      isDebuggable = true
      isMinifyEnabled = false
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
  }
  kotlinOptions {
    version = "2.1.0"
    jvmTarget = "21"
  }
  buildFeatures {
    buildConfig = true
    prefab = true
    compose = true
  }
  externalNativeBuild {
    cmake {
      path = file("../../../../CMakeLists.txt")
      version = "4.1.0"
    }
  }
}

// Copy Task for shaders

val copyShaders by tasks.registering(Copy::class) {
  from("../../../../shaders") // from Proj/shaders ...
  include("**/*.spv") // ... copy SPIR-V files ...
  into("$projectDir/src/main/assets/shaders") // ... into assets/shaders

  println("Copying Shaders");

  // force copy of shaders always
  outputs.upToDateWhen { false }

  // Optional: preserve folder structure relative to shaders/
  eachFile {
    path = relativePath.pathString
  }
  includeEmptyDirs = false
}

// Custom Task to download vulkan validation layers only on debug builds

// helper to check if all vulkan layer validation shared libs are present
fun hasValidationLayers(): Boolean {
  return abiList.all { abi ->
    jniLibsDir.get().dir(abi).file("libVkLayer_khronos_validation.so").asFile.exists()
  }
}

// task to download from GitHub validation layers android release (if link is broken or filename is wrong, this will break)
val downloadVulkanValidation by tasks.registering {
  outputs.file(validationZipFile)
  onlyIf { !validationZipFile.get().asFile.exists() }
  doLast {
    println("Downloading Vulkan Validation Layers $vulkanValidationVersion from $validationUrl ...")
    validationDownloadDirectory.get().asFile.mkdirs()
    // Note to self: .use is the equivalent of try-with-resources in Java
    URI(validationUrl).toURL().openStream().use { input ->
      validationZipFile.get().asFile.outputStream().use { output ->
        input.copyTo(output)
      }
    }
    println("Download of Vulkan Validation Layers Completed")
  }
}

// task to unzip validation layers
val unzipVulkanValidation by tasks.registering {
  dependsOn(downloadVulkanValidation)
  inputs.file(validationZipFile)
  outputs.dir(jniLibsDir) // TODO track individial lib file if more libs needed
  onlyIf { !hasValidationLayers() }
  doLast {
    println("Extracting Vulkan-ValidationLayers into ${jniLibsDir.get()} ...")
    ZipInputStream(validationZipFile.get().asFile.inputStream()).use { zip ->
      var entry = zip.nextEntry
      while (entry != null) {
        // next entry returns paths like
        // - android-binaries-1.4.309.0/
        // - android-binaries-1.4.309.0/arm64-v8a/
        // - android-binaries-1.4.309.0/arm64-v8a/file.so
        // hence we want to skip everything which is not a file and doesn't have a Android ABI
        // in one of its path components. Copy them inside jniLibs
        if (!entry.isDirectory && entry.name.endsWith(".so")) {
          val parts = entry.name.split('/')
          val abi = parts.find { it in abiList }
          if (abi != null) {
            val destDir = jniLibsDir.get().dir(abi).asFile
            destDir.mkdirs()
            val destFile = File(destDir, parts.last())
            // copy current opened entry
            destFile.outputStream().use { zip.copyTo(it) }
          }
        }

        // advance
        zip.closeEntry()
        entry = zip.nextEntry
      }
    }
  }
}

// task to copy asan files into the build directory
val copyASanSoFilesDebug by tasks.registering(Copy::class) {
  // determine host abi  (done here because specifying ndkVersion under android plugin changes its result!)
  val hostTag =
    File(android.ndkDirectory, "toolchains/llvm/prebuilt").listFiles()
      ?.firstOrNull { it.isDirectory }?.name
      ?: throw GradleException("No prebuilt host folder found under NDK")
  asanFiles.forEach { (fileName, abiFolder) ->
    val sourceFile = File(
      android.ndkDirectory,
      "toolchains/llvm/prebuilt/$hostTag/lib/clang/$clangVersion/lib/linux/$fileName"
    )
    val targetDir = debugAsanJniLibsDir.get().dir(abiFolder).asFile
    targetDir.mkdirs()
    if (!sourceFile.exists()) {
      throw GradleException("$sourceFile doesn't exist")
    }
    from(sourceFile) {
      // relative subfolder from the "global" into
      into(abiFolder)
    }
  }
  // "global" into: Specify root of destination files
  into(debugAsanJniLibsDir)
}

// hook the extract dependency into every debug task
androidComponents {
  onVariants(selector().withBuildType("debug")) {
    tasks.named("preBuild").configure {
      dependsOn(unzipVulkanValidation, copyASanSoFilesDebug)
    }
  }
}

// run this TaskProvider before building this module ...
tasks.named("preBuild") {
  dependsOn(copyShaders)
}
// ... and also before external build
tasks.matching { it.name.startsWith("externalNativeBuild") }.configureEach {
  dependsOn(copyShaders)
}

dependencies {
  implementation(platform(libs.androidx.compose.bom))

  // Core + Lifecycle
  implementation(libs.androidx.core.ktx)
  val lifecycle_version = "2.9.4"

  // ViewModel
  implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:$lifecycle_version")
  // ViewModel utilities for Compose
  implementation("androidx.lifecycle:lifecycle-viewmodel-compose:$lifecycle_version")
  // LiveData
  implementation("androidx.lifecycle:lifecycle-livedata-ktx:$lifecycle_version")
  // Lifecycles only (without ViewModel or LiveData)
  implementation("androidx.lifecycle:lifecycle-runtime-ktx:$lifecycle_version")
  // Lifecycle utilities for Compose
  implementation("androidx.lifecycle:lifecycle-runtime-compose:$lifecycle_version")

  // Saved state module for ViewModel
  implementation("androidx.lifecycle:lifecycle-viewmodel-savedstate:$lifecycle_version")

  // ViewModel integration with Navigation3
  implementation("androidx.lifecycle:lifecycle-viewmodel-navigation3:2.10.0-rc01")

  // Annotation processor
  kapt("androidx.lifecycle:lifecycle-compiler:$lifecycle_version")
  // alternately - if using Java8, use the following instead of lifecycle-compiler
  implementation("androidx.lifecycle:lifecycle-common-java8:$lifecycle_version")

  // Compose
  implementation(libs.androidx.activity.compose)
  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.material3)
  implementation(libs.androidx.compose.runtime)
  implementation(libs.androidx.compose.ui.tooling.preview)

  // Android + Material
  implementation(libs.androidx.appcompat)
  implementation(libs.androidx.material)

  // Game SDK
  implementation(libs.androidx.games.activity)

  // Testing
  testImplementation(libs.junit)
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.espresso.core)
}