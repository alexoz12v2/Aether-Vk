use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cmp::min;

extern crate symphonia_core;
extern crate symphonia_format_wav;
extern crate symphonia_codec_pcm;

use symphonia_core::audio::{AudioBufferRef, Signal};
use symphonia_core::codecs::{Decoder, DecoderOptions, CODEC_TYPE_PCM_S16LE};
use symphonia_core::formats::{FormatOptions, FormatReader};
use symphonia_core::io::{MediaSourceStream, ReadOnlySource};
use symphonia_core::probe::Hint;
use symphonia_format_wav::WavReader;
use symphonia_codec_pcm::PcmDecoder;

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
    pub fn from_wav_bytes(bytes: &'static [u8]) -> Self {
        let source = Box::new(ReadOnlySource::new(bytes));
        let mss = MediaSourceStream::new(source, Default::default());
        let mut reader = WavReader::try_new(mss, &FormatOptions::default()).unwrap();
        let track = reader.default_track().unwrap().clone();
        
        let mut decoder = PcmDecoder::try_new(&track.codec_params, &DecoderOptions::default()).unwrap();
        
        let mut samples_left = Vec::new();
        let mut samples_right = Vec::new();
        
        while let Ok(packet) = reader.next_packet() {
            if let Ok(decoded) = decoder.decode(&packet) {
                match decoded {
                    AudioBufferRef::S16(buf) => {
                        let chan_l = buf.chan(0);
                        let chan_r = if buf.spec().channels.count() > 1 { buf.chan(1) } else { chan_l };
                        
                        for (&l, &r) in chan_l.iter().zip(chan_r.iter()) {
                            samples_left.push(l as f32 / 32768.0);
                            samples_right.push(r as f32 / 32768.0);
                        }
                    }
                    AudioBufferRef::F32(buf) => {
                        let chan_l = buf.chan(0);
                        let chan_r = if buf.spec().channels.count() > 1 { buf.chan(1) } else { chan_l };
                        
                        samples_left.extend_from_slice(chan_l);
                        samples_right.extend_from_slice(chan_r);
                    }
                    _ => {} // Other formats not supported for this simple UI audio
                }
            }
        }
        
        Self {
            samples_left,
            samples_right,
            sample_rate: track.codec_params.sample_rate.unwrap_or(44100),
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
            let pitch_factor = playing.params.pitch * (buffer.sample_rate as f32 / self.sample_rate as f32);
            
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
                        let pan_l = if playing.params.pan < 0.0 { 1.0 } else { 1.0 - playing.params.pan };
                        let pan_r = if playing.params.pan > 0.0 { 1.0 } else { 1.0 + playing.params.pan };
                        (mono * pan_l, mono * pan_r)
                    }
                    AvkAudioPlaybackMode::StereoDirect => {
                        (s_left * playing.params.volume, s_right * playing.params.volume)
                    }
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
        mixer.play(id, AvkAudioParams {
            volume: 1.0,
            pitch: 1.0,
            pan: -1.0, // Hard left panning
            mode: AvkAudioPlaybackMode::MonoSpatial,
        });
        
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
}
