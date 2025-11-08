package org.aethervkproj.aethervk

import android.os.Debug
import com.google.androidgamesdk.GameActivity

class MainActivity : GameActivity() {
  companion object {
    init {
      System.loadLibrary("aethervk")
    }
  }
}
