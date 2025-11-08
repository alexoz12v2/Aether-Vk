import org.jetbrains.kotlin.compose.compiler.gradle.ComposeFeatureFlag
import org.jetbrains.kotlin.gradle.fus.internal.isJenkins

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

  defaultConfig {
    ndkVersion = "28.2.13676358"
    applicationId = "org.aethervkproj.aethervk"
    minSdk = 30 // Minimum for GameActivity
    targetSdk = 36
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    externalNativeBuild {
      cmake {
        // cppFlags += "-std=c++17"
        arguments.add("-DANDROID_USE_LEGACY_TOOLCHAIN_FILE=OFF")
      }
    }
  }

  // https://developer.android.com/build/build-variants
  buildTypes {
    // https://developer.android.com/studio/debug
    debug {
      isDebuggable = true
      // applicationIdSuffix = ".debug"
      ndk {
        isDebuggable = true
        isJniDebuggable = true
        debugSymbolLevel = "FULL"
      }
    }
    release {
      isDebuggable = true
      isMinifyEnabled = false
      proguardFiles(
        getDefaultProguardFile("proguard-android-optimize.txt"),
        "proguard-rules.pro"
      )
    }
    create("RelWithDbg") {
      initWith(getByName("release"))
      isDebuggable = true
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

val copyShaders by tasks.registering(Copy::class) {
  from("../../../../shaders") // from Proj/shaders ...
  include("**/*.spv") // ... copy SPIR-V files ...
  into("$projectDir/src/main/assets/shaders") // ... into assets/shaders

  // Optional: preserve folder structure relative to shaders/
  eachFile {
    path = relativePath.pathString
  }
  includeEmptyDirs = false
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