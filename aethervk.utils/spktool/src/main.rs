use anise::almanac::Almanac;
use anise::constants::celestial_objects;
use anise::constants::orientations;
use hifitime::Epoch;
use std::env;
use std::path::Path;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut args: Vec<String> = env::args().skip(1).collect();
  let mut asset_dir = env::var("SPKTOOL_ASSET_DIR").unwrap_or_else(|_| "".to_string());

  let mut i = 0;
  while i < args.len() {
    if args[i] == "--asset-dir" && i + 1 < args.len() {
      asset_dir = args[i + 1].clone();
      args.remove(i);
      args.remove(i);
    } else {
      i += 1;
    }
  }

  if args.is_empty() {
    eprintln!("Usage: spktool [--asset-dir <dir>] <command> <args...>");
    eprintln!("Commands:");
    eprintln!("  spksummary <file>");
    eprintln!("  cartesian_state <file> <epoch_list_comma_separated>");
    std::process::exit(1);
  }

  let command = &args[0];

  // Load basic kernels
  let mut almanac = Almanac::default();

  if !asset_dir.is_empty() {
    let base = Path::new(&asset_dir);
    let pck = base.join("planets/pck00011.pca");
    let gm = base.join("planets/gm_de431.pca");
    let de = base.join("planets/de442.bsp");

    if pck.exists() {
      almanac = almanac.load(pck.to_str().unwrap()).expect("Failed to load PCK");
    } else {
      eprintln!("Warning: PCK not found at {:?}", pck);
    }
    if gm.exists() {
      almanac = almanac.load(gm.to_str().unwrap()).expect("Failed to load GM");
    } else {
      eprintln!("Warning: GM not found at {:?}", gm);
    }
    if de.exists() {
      almanac = almanac.load(de.to_str().unwrap()).expect("Failed to load DE442");
    } else {
      eprintln!("Warning: DE442 not found at {:?}", de);
    }
  } else {
    eprintln!("Warning: No asset dir provided (SPKTOOL_ASSET_DIR or --asset-dir)");
  }

  match command.as_str() {
    "spksummary" => {
      if args.len() < 2 {
        eprintln!("Usage: spktool spksummary <file>");
        std::process::exit(1);
      }
      let file = &args[1];
      almanac = almanac.load(file).expect("Failed to load SPK file");

      let domains = almanac.spk_domains().expect("Failed to get SPK domains");
      for (id, (start, end)) in domains {
        println!("ID: {}", id);
        println!("  Start: {}", start);
        println!("  End:   {}", end);
        if let Ok((summary, _, _, _)) = almanac.spk_summary_at_epoch(id, start) {
          println!("  Type:  {}", summary.data_type_i);
        }
      }
    }
    "cartesian_state" => {
      if args.len() < 3 {
        eprintln!("Usage: spktool cartesian_state <file> <epoch_list_comma_separated>");
        std::process::exit(1);
      }
      let file = &args[1];
      let epochs_str = &args[2];
      almanac = almanac.load(file).expect("Failed to load SPK file");

      // We need to figure out which ID to query.
      let domains = almanac.spk_domains().expect("Failed to get SPK domains");
      let mut target_id = 0;
      for (id, _) in domains {
        if id != 0 {
          target_id = id;
          break;
        }
      }

      if target_id == 0 {
        eprintln!("No valid target ID found in SPK");
        std::process::exit(1);
      }

      println!("Target ID: {}", target_id);

      let epochs: Vec<&str> = epochs_str.split(',').collect();
      for e_str in epochs {
        let e_str = e_str.trim();
        if e_str.is_empty() {
          continue;
        }

        let epoch = match Epoch::from_str(e_str) {
          Ok(e) => e,
          Err(err) => {
            eprintln!("Failed to parse epoch {}: {:?}", e_str, err);
            continue;
          }
        };

        match almanac.spk_ezr(
          target_id,
          epoch,
          orientations::J2000,
          celestial_objects::SUN,
          None,
        ) {
          Ok(state) => {
            println!("Epoch: {}", epoch);
            println!("  Pos: {:?}", state.radius_km);
            println!("  Vel: {:?}", state.velocity_km_s);
          }
          Err(e) => {
            eprintln!("Error getting state at {}: {}", epoch, e);
          }
        }
      }
    }
    _ => {
      eprintln!("Unknown command: {}", command);
      std::process::exit(1);
    }
  }

  Ok(())
}
