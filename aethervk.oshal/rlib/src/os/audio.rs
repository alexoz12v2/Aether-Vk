//! OS Audio Abstraction Layer
//!
//! Provides the underlying OS thread and buffer polling mechanism to pump audio to the
//! system speakers. This layer is `#![no_std]` compliant and uses raw OS bindings.
//!
//! # Platform Equivalents:
//! - **Windows**: WASAPI (Windows Audio Session API) via COM (`windows` crate)
//! - **macOS/iOS**: CoreAudio (`mach2`, `libc`, manual C-bindings)
//! - **Linux**: ALSA (`libc::dlopen("libasound.so")`)

pub trait AudioDevice {
  /// Starts the background OS audio thread.
  /// The provided callback is executed on the high-priority OS audio thread
  /// to fill the hardware buffer with interleaved float samples.
  fn start(&mut self, render_callback: fn(&mut [f32]));

  /// Stops the audio thread and releases hardware resources.
  fn stop(&mut self);
}

#[cfg(windows)]
pub mod windows_wasapi {
  use super::AudioDevice;
  use core::ffi::c_void;
  use windows::Win32::{
    Foundation::*,
    Media::Audio::*,
    System::{Com::*, Threading::*},
  };

  pub struct WasapiDevice {
    is_running: core::sync::atomic::AtomicBool,
    thread_handle: HANDLE,
    callback: Option<fn(&mut [f32])>,
  }

  unsafe impl Send for WasapiDevice {}
  unsafe impl Sync for WasapiDevice {}

  impl WasapiDevice {
    pub fn new() -> Self {
      Self {
        is_running: core::sync::atomic::AtomicBool::new(false),
        thread_handle: HANDLE::default(),
        callback: None,
      }
    }
  }

  struct ThreadContext {
    is_running: *const core::sync::atomic::AtomicBool,
    callback: fn(&mut [f32]),
  }

  unsafe extern "system" fn wasapi_thread_func(param: *mut c_void) -> u32 {
    let context = &*(param as *const ThreadContext);

    let _ = CoInitializeEx(None, COINIT_MULTITHREADED);

    if let Ok(enumerator) =
      CoCreateInstance::<_, IMMDeviceEnumerator>(&MMDeviceEnumerator, None, CLSCTX_ALL)
    {
      if let Ok(device) = enumerator.GetDefaultAudioEndpoint(eRender, eConsole) {
        if let Ok(client) = device.Activate::<IAudioClient>(CLSCTX_ALL, None) {
          if let Ok(mix_format_ptr) = client.GetMixFormat() {
            let mix_format = *mix_format_ptr;
            let hns_requested_duration = 10_000_000 / 10; // 100ms

            if client
              .Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                0,
                hns_requested_duration,
                0,
                mix_format_ptr,
                None,
              )
              .is_ok()
            {
              if let Ok(render_client) = client.GetService::<IAudioRenderClient>() {
                if let Ok(buffer_frame_count) = client.GetBufferSize() {
                  if client.Start().is_ok() {
                    let sleep_duration = (hns_requested_duration / 10_000 / 2) as u32;

                    while (*context.is_running).load(core::sync::atomic::Ordering::Relaxed) {
                      Sleep(sleep_duration);

                      if let Ok(num_frames_padding) = client.GetCurrentPadding() {
                        let num_frames_available = buffer_frame_count - num_frames_padding;

                        if num_frames_available > 0 {
                          if let Ok(buffer_ptr) = render_client.GetBuffer(num_frames_available) {
                            let num_samples =
                              (num_frames_available as usize) * (mix_format.nChannels as usize);
                            let slice =
                              core::slice::from_raw_parts_mut(buffer_ptr as *mut f32, num_samples);

                            (context.callback)(slice);

                            let _ = render_client.ReleaseBuffer(num_frames_available, 0);
                          }
                        }
                      }
                    }
                    let _ = client.Stop();
                  }
                }
              }
            }
            CoTaskMemFree(Some(mix_format_ptr as *mut c_void));
          }
        }
      }
    }

    CoUninitialize();
    let _ = alloc::boxed::Box::from_raw(param as *mut ThreadContext);
    0
  }

  impl AudioDevice for WasapiDevice {
    fn start(&mut self, render_callback: fn(&mut [f32])) {
      self.callback = Some(render_callback);
      self.is_running.store(true, core::sync::atomic::Ordering::Relaxed);

      let context = alloc::boxed::Box::new(ThreadContext {
        is_running: &self.is_running as *const _,
        callback: render_callback,
      });

      let context_ptr = alloc::boxed::Box::into_raw(context) as *mut c_void;

      unsafe {
        self.thread_handle = CreateThread(
          None,
          0,
          Some(wasapi_thread_func),
          Some(context_ptr),
          THREAD_CREATION_FLAGS(0),
          None,
        )
        .unwrap_or_default();

        if !self.thread_handle.is_invalid() {
          let _ = SetThreadPriority(self.thread_handle, THREAD_PRIORITY_HIGHEST);
        }
      }
    }

    fn stop(&mut self) {
      self.is_running.store(false, core::sync::atomic::Ordering::Relaxed);
      if !self.thread_handle.is_invalid() {
        unsafe {
          WaitForSingleObject(self.thread_handle, INFINITE);
          let _ = CloseHandle(self.thread_handle);
        }
        self.thread_handle = HANDLE::default();
      }
    }
  }
}

#[cfg(target_os = "macos")]
pub mod macos_coreaudio {
  use super::AudioDevice;
  use core::ffi::c_void;

  #[repr(C)]
  struct AudioComponentDescription {
    componentType: u32,
    componentSubType: u32,
    componentManufacturer: u32,
    componentFlags: u32,
    componentFlagsMask: u32,
  }

  #[repr(C)]
  struct AURenderCallbackStruct {
    inputProc: extern "C" fn(*mut c_void, *mut u32, *const c_void, u32, u32, *mut c_void) -> i32,
    inputProcRefCon: *mut c_void,
  }

  #[repr(C)]
  struct AudioStreamBasicDescription {
    mSampleRate: f64,
    mFormatID: u32,
    mFormatFlags: u32,
    mBytesPerPacket: u32,
    mFramesPerPacket: u32,
    mBytesPerFrame: u32,
    mChannelsPerFrame: u32,
    mBitsPerChannel: u32,
    mReserved: u32,
  }

  #[link(name = "CoreAudio", kind = "framework")]
  #[link(name = "AudioUnit", kind = "framework")]
  unsafe extern "C" {
    fn AudioComponentFindNext(
      inComponent: *mut c_void,
      inDesc: *const AudioComponentDescription,
    ) -> *mut c_void;
    fn AudioComponentInstanceNew(inComponent: *mut c_void, outInstance: *mut *mut c_void) -> i32;
    fn AudioUnitSetProperty(
      inUnit: *mut c_void,
      inID: u32,
      inScope: u32,
      inElement: u32,
      inData: *const c_void,
      inDataSize: u32,
    ) -> i32;
    fn AudioUnitInitialize(inUnit: *mut c_void) -> i32;
    fn AudioOutputUnitStart(inUnit: *mut c_void) -> i32;
    fn AudioOutputUnitStop(inUnit: *mut c_void) -> i32;
    fn AudioComponentInstanceDispose(inInstance: *mut c_void) -> i32;
  }

  const K_AUDIO_UNIT_TYPE_OUTPUT: u32 = 0x61756f75; // 'auou'
  const K_AUDIO_UNIT_SUBTYPE_DEFAULT_OUTPUT: u32 = 0x6465666f; // 'defo'
  const K_AUDIO_UNIT_MANUFACTURER_APPLE: u32 = 0x6170706c; // 'appl'
  const K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK: u32 = 23;
  const K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT: u32 = 8;
  const K_AUDIO_UNIT_SCOPE_INPUT: u32 = 1;
  const K_AUDIO_FORMAT_LINEAR_PCM: u32 = 0x6c70636d; // 'lpcm'
  const K_AUDIO_FORMAT_FLAG_IS_FLOAT: u32 = 1 << 0;
  const K_AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 1 << 3;

  pub struct CoreAudioDevice {
    unit: *mut c_void,
    callback: Option<fn(&mut [f32])>,
  }

  unsafe impl Send for CoreAudioDevice {}
  unsafe impl Sync for CoreAudioDevice {}

  // The high-priority interrupt fired directly by Apple's CoreAudio daemon!
  extern "C" fn render_callback_impl(
    inRefCon: *mut c_void,
    _ioActionFlags: *mut u32,
    _inTimeStamp: *const c_void,
    _inBusNumber: u32,
    inNumberFrames: u32,
    ioData: *mut c_void,
  ) -> i32 {
    unsafe {
      let device = &mut *(inRefCon as *mut CoreAudioDevice);

      // ioData points to an AudioBufferList.
      // mNumberBuffers is the first u32 field
      let m_number_buffers = *(ioData as *const u32);
      if m_number_buffers > 0 {
        // Parse the first AudioBuffer in the array
        // struct AudioBuffer { mNumberChannels: u32, mDataByteSize: u32, mData: *mut c_void }
        let audio_buffer_ptr = (ioData as *mut u8).add(8) as *mut u32;
        let data_ptr = *(audio_buffer_ptr.add(2) as *const *mut f32);

        let num_samples = (inNumberFrames * 2) as usize; // Stereo interleaved
        let slice = core::slice::from_raw_parts_mut(data_ptr, num_samples);

        if let Some(cb) = device.callback {
          // Pull from the Core Engine's lock-free math mixer!
          cb(slice);
        } else {
          for sample in slice.iter_mut() {
            *sample = 0.0;
          }
        }
      }
    }
    0 // return noErr
  }

  impl CoreAudioDevice {
    pub fn new() -> Self {
      Self {
        unit: core::ptr::null_mut(),
        callback: None,
      }
    }
  }

  impl AudioDevice for CoreAudioDevice {
    fn start(&mut self, render_callback: fn(&mut [f32])) {
      self.callback = Some(render_callback);

      let desc = AudioComponentDescription {
        componentType: K_AUDIO_UNIT_TYPE_OUTPUT,
        componentSubType: K_AUDIO_UNIT_SUBTYPE_DEFAULT_OUTPUT,
        componentManufacturer: K_AUDIO_UNIT_MANUFACTURER_APPLE,
        componentFlags: 0,
        componentFlagsMask: 0,
      };

      unsafe {
        let comp = AudioComponentFindNext(core::ptr::null_mut(), &desc);
        if comp.is_null() {
          return;
        }

        if AudioComponentInstanceNew(comp, &mut self.unit) != 0 {
          return;
        }

        let cb_struct = AURenderCallbackStruct {
          inputProc: render_callback_impl,
          inputProcRefCon: self as *mut _ as *mut c_void,
        };

        AudioUnitSetProperty(
          self.unit,
          K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK,
          K_AUDIO_UNIT_SCOPE_INPUT,
          0,
          &cb_struct as *const _ as *const c_void,
          core::mem::size_of::<AURenderCallbackStruct>() as u32,
        );

        let format = AudioStreamBasicDescription {
          mSampleRate: 44100.0,
          mFormatID: K_AUDIO_FORMAT_LINEAR_PCM,
          mFormatFlags: K_AUDIO_FORMAT_FLAG_IS_FLOAT | K_AUDIO_FORMAT_FLAG_IS_PACKED,
          mBytesPerPacket: 8, // 2 channels * 4 bytes per float
          mFramesPerPacket: 1,
          mBytesPerFrame: 8,
          mChannelsPerFrame: 2,
          mBitsPerChannel: 32,
          mReserved: 0,
        };

        AudioUnitSetProperty(
          self.unit,
          K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT,
          K_AUDIO_UNIT_SCOPE_INPUT,
          0,
          &format as *const _ as *const c_void,
          core::mem::size_of::<AudioStreamBasicDescription>() as u32,
        );

        AudioUnitInitialize(self.unit);
        AudioOutputUnitStart(self.unit);
      }
    }

    fn stop(&mut self) {
      if !self.unit.is_null() {
        unsafe {
          AudioOutputUnitStop(self.unit);
          AudioComponentInstanceDispose(self.unit);
        }
        self.unit = core::ptr::null_mut();
      }
    }
  }
}

#[cfg(target_os = "linux")]
pub mod linux_alsa {
  use super::AudioDevice;
  use core::ffi::{c_char, c_int, c_uint, c_void};

  type FnSndPcmOpen = unsafe extern "C" fn(*mut *mut c_void, *const c_char, c_int, c_int) -> c_int;
  type FnSndPcmSetParams =
    unsafe extern "C" fn(*mut c_void, c_int, c_int, c_uint, c_uint, c_int, c_uint) -> c_int;
  type FnSndPcmWritei = unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> isize;
  type FnSndPcmRecover = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> c_int;
  type FnSndPcmClose = unsafe extern "C" fn(*mut c_void) -> c_int;

  #[derive(Clone, Copy)]
  struct AlsaApi {
    lib_handle: *mut c_void,
    snd_pcm_open: FnSndPcmOpen,
    snd_pcm_set_params: FnSndPcmSetParams,
    snd_pcm_writei: FnSndPcmWritei,
    snd_pcm_recover: FnSndPcmRecover,
    snd_pcm_close: FnSndPcmClose,
  }

  const SND_PCM_STREAM_PLAYBACK: c_int = 0;
  const SND_PCM_FORMAT_FLOAT_LE: c_int = 14;
  const SND_PCM_ACCESS_RW_INTERLEAVED: c_int = 3;

  pub struct AlsaDevice {
    is_running: core::sync::atomic::AtomicBool,
    thread_handle: libc::pthread_t,
    callback: Option<fn(&mut [f32])>,
  }

  unsafe impl Send for AlsaDevice {}
  unsafe impl Sync for AlsaDevice {}

  impl AlsaDevice {
    pub fn new() -> Self {
      Self {
        is_running: core::sync::atomic::AtomicBool::new(false),
        thread_handle: 0,
        callback: None,
      }
    }
  }

  struct ThreadContext {
    is_running: *const core::sync::atomic::AtomicBool,
    callback: fn(&mut [f32]),
    api: AlsaApi,
  }

  extern "C" fn alsa_thread_func(param: *mut c_void) -> *mut c_void {
    let context = unsafe { &*(param as *const ThreadContext) };

    let mut pcm: *mut c_void = core::ptr::null_mut();
    let default_name = b"default\0".as_ptr() as *const c_char;

    unsafe {
      if (context.api.snd_pcm_open)(&mut pcm, default_name, SND_PCM_STREAM_PLAYBACK, 0) < 0 {
        let _ = alloc::boxed::Box::from_raw(param as *mut ThreadContext);
        return core::ptr::null_mut();
      }

      if (context.api.snd_pcm_set_params)(
        pcm,
        SND_PCM_FORMAT_FLOAT_LE,
        SND_PCM_ACCESS_RW_INTERLEAVED,
        2,
        44100,
        1,
        50000,
      ) < 0
      {
        (context.api.snd_pcm_close)(pcm);
        let _ = alloc::boxed::Box::from_raw(param as *mut ThreadContext);
        return core::ptr::null_mut();
      }
    }

    let frames_per_buffer = 1024;
    let num_samples = frames_per_buffer * 2;
    let mut buffer = alloc::vec![0.0f32; num_samples];

    while unsafe { (*context.is_running).load(core::sync::atomic::Ordering::Relaxed) } {
      (context.callback)(&mut buffer);

      unsafe {
        let mut frames = (context.api.snd_pcm_writei)(
          pcm,
          buffer.as_ptr() as *const c_void,
          frames_per_buffer as usize,
        );
        if frames < 0 {
          frames = (context.api.snd_pcm_recover)(pcm, frames as c_int, 0) as isize;
        }
      }
    }

    unsafe {
      (context.api.snd_pcm_close)(pcm);
      libc::dlclose(context.api.lib_handle);
      let _ = alloc::boxed::Box::from_raw(param as *mut ThreadContext);
    }
    core::ptr::null_mut()
  }

  impl AudioDevice for AlsaDevice {
    fn start(&mut self, render_callback: fn(&mut [f32])) {
      self.callback = Some(render_callback);
      self.is_running.store(true, core::sync::atomic::Ordering::Relaxed);

      let lib_handle = unsafe {
        let lib_name = b"libasound.so.2\0".as_ptr() as *const c_char;
        let handle = libc::dlopen(lib_name, libc::RTLD_NOW);
        if handle.is_null() {
          let fallback_name = b"libasound.so\0".as_ptr() as *const c_char;
          libc::dlopen(fallback_name, libc::RTLD_NOW)
        } else {
          handle
        }
      };

      if lib_handle.is_null() {
        // If we can't load ALSA, we just don't play sound.
        return;
      }

      let api = unsafe {
        AlsaApi {
          lib_handle,
          snd_pcm_open: core::mem::transmute(libc::dlsym(
            lib_handle,
            b"snd_pcm_open\0".as_ptr() as *const c_char,
          )),
          snd_pcm_set_params: core::mem::transmute(libc::dlsym(
            lib_handle,
            b"snd_pcm_set_params\0".as_ptr() as *const c_char,
          )),
          snd_pcm_writei: core::mem::transmute(libc::dlsym(
            lib_handle,
            b"snd_pcm_writei\0".as_ptr() as *const c_char,
          )),
          snd_pcm_recover: core::mem::transmute(libc::dlsym(
            lib_handle,
            b"snd_pcm_recover\0".as_ptr() as *const c_char,
          )),
          snd_pcm_close: core::mem::transmute(libc::dlsym(
            lib_handle,
            b"snd_pcm_close\0".as_ptr() as *const c_char,
          )),
        }
      };

      let context = alloc::boxed::Box::new(ThreadContext {
        is_running: &self.is_running as *const _,
        callback: render_callback,
        api,
      });

      let context_ptr = alloc::boxed::Box::into_raw(context) as *mut c_void;

      unsafe {
        libc::pthread_create(
          &mut self.thread_handle,
          core::ptr::null(),
          alsa_thread_func,
          context_ptr,
        );
      }
    }

    fn stop(&mut self) {
      self.is_running.store(false, core::sync::atomic::Ordering::Relaxed);
      if self.thread_handle != 0 {
        unsafe {
          libc::pthread_join(self.thread_handle, core::ptr::null_mut());
        }
        self.thread_handle = 0;
      }
    }
  }
}
