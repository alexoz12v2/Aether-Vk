use aethervk_oshal_rlib::os::files::MappedFile;
use alloc::{boxed::Box, vec::Vec};
use pure_wav::{Parser, ProcessDataOutput};

pub enum AvkSoundEvent {
  UiClick = 0,
  UiGrab = 1,
  UiDrop = 2,
  PhysicsCollisionSoft = 3,
  PhysicsCollisionHard = 4,
}

pub enum AvkAudioPlaybackMode {
  MonoSpatial = 0,
  StereoDirect = 1,
}

pub struct AvkAudioParams {
  pub volume: f32,
  pub pitch: f32,
  pub pan: f32,
  pub mode: AvkAudioPlaybackMode,
}

/// A decoded sound buffer ready to be played.
pub struct SoundBuffer {
  pub samples_left: Vec<f32>,
  pub samples_right: Vec<f32>,
  pub sample_rate: u32,
}

impl SoundBuffer {
  /// Decode a WAV file from memory using Symphonia in `no_std`
  pub fn from_wav_mapped(mmap: &MappedFile) -> Self {
    let bytes = mmap.as_slice();
    let mut parser = Parser::default();

    // 1. Run the state machine to parse the RIFF and find the data chunk
    let meta = loop {
      let instruction = parser.read_instruction();
      let pos = instruction.position as usize;
      let len = instruction.len as usize;

      // Validate bounds to prevent slicing panics on malformed files
      if pos + len > bytes.len() {
        panic!("WAV file is truncated or invalid");
      }

      let chunk = &bytes[pos..pos + len];

      match parser.process_data(chunk).expect("Failed to parse WAV chunk") {
        ProcessDataOutput::Done(meta_data) => break meta_data,
        ProcessDataOutput::InProgress(next_parser) => parser = next_parser,
      }
    };

    // 2. Extract values (using .get() to convert zerocopy wrappers to native primitives)
    let format_tag = meta.fmt.format_tag.get();
    let channels = meta.fmt.n_channels.get() as usize;
    let sample_rate = meta.fmt.n_samples_per_sec.get();
    let bits_per_sample = meta.fmt.w_bits_per_sample.get();

    let data_pos = meta.data_position as usize;
    let data_len = meta.data_len as usize;

    if data_pos + data_len > bytes.len() {
      panic!("WAV data chunk exceeds file length");
    }

    // 3. Extract the exact byte slice containing just the PCM audio data
    let pcm_bytes = &bytes[data_pos..data_pos + data_len];

    let mut samples_left = Vec::new();
    let mut samples_right = Vec::new();

    let bytes_per_sample = meta.fmt.w_bits_per_sample.get() as usize / 8;
    let frame_size = bytes_per_sample * meta.fmt.n_channels.get() as usize;

    // Note: Check the exact field name in the crate you end up using.
    // It is typically called `audio_format`, `format_tag`, or `audio_format_code`.
    match (meta.fmt.format_tag.get(), meta.fmt.w_bits_per_sample.get()) {
      (1, 16) => {
        // Format 1: Integer PCM (16-bit)
        for frame in pcm_bytes.chunks_exact(frame_size) {
          let left_val = i16::from_le_bytes([frame[0], frame[1]]);
          samples_left.push(left_val as f32 / 32768.0);

          if meta.fmt.n_channels > 1 {
            let right_val = i16::from_le_bytes([frame[2], frame[3]]);
            samples_right.push(right_val as f32 / 32768.0);
          } else {
            samples_right.push(left_val as f32 / 32768.0);
          }
        }
      }
      (3, 32) => {
        // Format 3: IEEE Float (32-bit)
        for frame in pcm_bytes.chunks_exact(frame_size) {
          let left_val = f32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]);
          samples_left.push(left_val); // Already scaled -1.0 to 1.0

          if meta.fmt.n_channels > 1 {
            let right_val = f32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]]);
            samples_right.push(right_val);
          } else {
            samples_right.push(left_val);
          }
        }
      }
      (1, 32) => {
        panic!(
          "Found 32-bit Integer PCM (Format 1). This is currently unsupported. Only 32-bit Float (Format 3) is supported."
        );
      }
      (format_code, bits) => {
        panic!(
          "Unsupported WAV format: format code {} with {} bits per sample. \
                     Only 16-bit PCM (code 1) and 32-bit Float (code 3) are supported.",
          format_code, bits
        );
      }
    }

    Self {
      samples_left,
      samples_right,
      sample_rate: meta.fmt.n_samples_per_sec.get() as _,
    }
  }
}

pub struct PlayingSound {
  buffer_id: usize,
  cursor: f32, // f32 for pitch interpolation
  params: AvkAudioParams,
}

pub struct AudioMixer {
  buffers: Vec<SoundBuffer>,
  playing: Vec<PlayingSound>,
  pub sample_rate: u32,
}

impl AudioMixer {
  pub fn new(sample_rate: u32) -> Self {
    Self {
      buffers: Vec::new(),
      playing: Vec::new(),
      sample_rate,
    }
  }

  pub fn load_buffer(&mut self, buffer: SoundBuffer) -> usize {
    let id = self.buffers.len();
    self.buffers.push(buffer);
    id
  }

  pub fn play(&mut self, buffer_id: usize, params: AvkAudioParams) {
    if buffer_id < self.buffers.len() {
      self.playing.push(PlayingSound {
        buffer_id,
        cursor: 0.0,
        params,
      });
    }
  }

  /// Mix active sounds into a provided stereo interleaved float buffer.
  pub fn mix(&mut self, output: &mut [f32]) {
    // Clear buffer
    for sample in output.iter_mut() {
      *sample = 0.0;
    }

    let mut finished = Vec::new();

    for (i, playing) in self.playing.iter_mut().enumerate() {
      let buffer = &self.buffers[playing.buffer_id];
      let pitch_factor =
        playing.params.pitch * (buffer.sample_rate as f32 / self.sample_rate as f32);

      let mut out_idx = 0;
      while out_idx < output.len() {
        let current_idx = playing.cursor as usize;
        if current_idx >= buffer.samples_left.len() {
          finished.push(i);
          break;
        }

        let s_left = buffer.samples_left[current_idx];
        let s_right = buffer.samples_right[current_idx];

        let (mixed_l, mixed_r) = match playing.params.mode {
          AvkAudioPlaybackMode::MonoSpatial => {
            let mono = (s_left + s_right) * 0.5 * playing.params.volume;
            let pan_l = if playing.params.pan < 0.0 {
              1.0
            } else {
              1.0 - playing.params.pan
            };
            let pan_r = if playing.params.pan > 0.0 {
              1.0
            } else {
              1.0 + playing.params.pan
            };
            (mono * pan_l, mono * pan_r)
          }
          AvkAudioPlaybackMode::StereoDirect => (
            s_left * playing.params.volume,
            s_right * playing.params.volume,
          ),
        };

        output[out_idx] += mixed_l;
        output[out_idx + 1] += mixed_r;

        playing.cursor += pitch_factor;
        out_idx += 2;
      }
    }

    // Remove finished sounds in reverse order
    for &idx in finished.iter().rev() {
      self.playing.remove(idx);
    }

    // Hard limiter to avoid clipping
    for sample in output.iter_mut() {
      *sample = sample.clamp(-1.0, 1.0);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use alloc::vec;

  #[test]
  fn test_audio_mixer_no_std() {
    let mut mixer = AudioMixer::new(44100);

    // Dummy 1 second stereo buffer at 44100
    let mut buf_left = vec![0.0; 44100];
    let mut buf_right = vec![0.0; 44100];
    buf_left[0] = 0.5;
    buf_right[0] = 0.5;

    let buf = SoundBuffer {
      samples_left: buf_left,
      samples_right: buf_right,
      sample_rate: 44100,
    };

    let id = mixer.load_buffer(buf);
    mixer.play(
      id,
      AvkAudioParams {
        volume: 1.0,
        pitch: 1.0,
        pan: -1.0, // Hard left panning
        mode: AvkAudioPlaybackMode::MonoSpatial,
      },
    );

    let mut output = vec![0.0; 256]; // 128 stereo frames
    mixer.mix(&mut output);

    // The first frame should be left-panned
    // Left channel (index 0) = 0.5 (mono avg) * 1.0 (vol) * 1.0 (pan L) = 0.5
    // Right channel (index 1) = 0.5 (mono avg) * 1.0 (vol) * 0.0 (pan R) = 0.0
    assert_eq!(output[0], 0.5);
    assert_eq!(output[1], 0.0);

    // Ensure subsequent are 0
    assert_eq!(output[2], 0.0);
  }

  #[test]
  fn test_audio_mixer_stereo_direct_and_pitch() {
    let mut mixer = AudioMixer::new(44100);

    // Dummy 1 second stereo buffer at 44100
    let mut buf_left = vec![0.0; 44100];
    let mut buf_right = vec![0.0; 44100];
    buf_left[0] = 0.5;
    buf_left[1] = 0.25;
    buf_right[0] = 1.0;
    buf_right[1] = 0.5;

    let buf = SoundBuffer {
      samples_left: buf_left,
      samples_right: buf_right,
      sample_rate: 44100,
    };

    let id = mixer.load_buffer(buf);

    // Play at 2.0x pitch (should skip every other sample), and half volume
    mixer.play(
      id,
      AvkAudioParams {
        volume: 0.5,
        pitch: 2.0,
        pan: 0.0,
        mode: AvkAudioPlaybackMode::StereoDirect,
      },
    );

    let mut output = vec![0.0; 256];
    mixer.mix(&mut output);

    // First frame (idx 0 of buffer)
    // Left: 0.5 * 0.5 = 0.25
    // Right: 1.0 * 0.5 = 0.5
    assert_eq!(output[0], 0.25);
    assert_eq!(output[1], 0.5);

    // Second frame: because pitch is 2.0, cursor moved by 2.0. So it reads idx 2 of buffer!
    // But idx 2 is 0.0 for both channels.
    assert_eq!(output[2], 0.0);
    assert_eq!(output[3], 0.0);
  }
}