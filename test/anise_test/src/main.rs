#![no_std]
#![no_main]

extern crate alloc;
extern crate libc;

use alloc::format;
use alloc::string::String;

use aethervk_oshal_rlib::log;
use aethervk_oshal_rlib::os::fs;
use anise::almanac::Almanac;
use anise::time::{Duration, Epoch};
use bytes::{Bytes, BytesMut};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(target_env = "msvc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[unsafe(no_mangle)]
pub extern "C" fn main(_argc: isize, _argv: *const *const u8) -> isize {
  log!("--- Starting Anise no_std Logging Example ---");

  let manifest_dir = env!("CARGO_MANIFEST_DIR");
  let bsp_path_str = format!("{}/../../assets/planets/de440.bsp", manifest_dir);
  let bsp_path = fs::PathBuf::from(&bsp_path_str);

  log!("Attempting to read SPK file from: {}", bsp_path_str);

  let file_data = match fs::read(&bsp_path) {
    Ok(d) => d,
    Err(e) => {
      log!("Failed to read file. {:?}", e);
      return 1;
    }
  };

  let bytes = Bytes::from_owner(file_data);

  let almanac = match Almanac::default().load_from_bytes(bytes, "Default") {
    Ok(a) => a,
    Err(e) => {
      log!("Failed to load SPK: {:?}", e);
      return 1;
    }
  };

  log!("Successfully loaded almanac.");

  // Note that the origin of our coordinate system is the Solar system barycentre
  // 399 = Earth
  let target = 399;
  // 0 = Solar System Barycenter
  let observer = 0;
  // 1 = J2000
  let frame = 1;

  // Let's create an epoch corresponding to year 2000
  let epoch_start = Epoch::from_gregorian_utc_at_midnight(2000, 1, 1);
  let epoch_end = Epoch::from_gregorian_utc_at_midnight(2001, 1, 1);

  let mut current_epoch = epoch_start;

  // We'll accumulate the state logs in a simple CSV format.
  let mut output_data = String::new();
  output_data.push_str("Epoch,X,Y,Z,VX,VY,VZ\n");

  let mut log_count = 0;

  while current_epoch <= epoch_end {
    match almanac.spk_ezr(target, current_epoch, frame, observer, None) {
      Ok(state) => {
        // A simple data selection filter: let's only log when the radius x component is positive, for example.
        if state.radius_km[0] >= 0.0 {
          let s = format!(
            "{},{},{},{},{},{},{}\n",
            current_epoch,
            state.radius_km[0],
            state.radius_km[1],
            state.radius_km[2],
            state.velocity_km_s[0],
            state.velocity_km_s[1],
            state.velocity_km_s[2]
          );
          output_data.push_str(&s);
          log_count += 1;
        }
      }
      Err(e) => {
        log!("spk_ezr failed at {}: {:?}", current_epoch, e);
      }
    }

    current_epoch += Duration::from_days(1.0);
  }

  let out_path_str = format!("{}/earth_trajectory_log.csv", manifest_dir);
  let out_path = fs::PathBuf::from(&out_path_str);
  log!("Writing {} records to {}", log_count, out_path_str);

  match fs::write(&out_path, output_data.as_bytes()) {
    Ok(_) => log!("Successfully wrote trajectory data."),
    Err(_) => {
      log!("Failed to write trajectory data to file.");
      return 1;
    }
  }

  log!("--- Example Finished ---");
  0
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
  aethervk_oshal_rlib::panic_handler_impl();
}

#[cfg(not(test))]
#[unsafe(no_mangle)]
pub extern "C" fn rust_eh_personality() {}

#[cfg(not(test))]
#[unsafe(no_mangle)]
pub extern "C" fn __rust_drop_panic() -> ! {
  loop {}
}
