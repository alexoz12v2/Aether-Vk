use anise::naif::kpl::parser::{convert_tpc_items, parse_file};
use anise::naif::kpl::tpc::TPCItem;
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

// Note: used for Platetary constants ANISE (.pca) generation -> `cargo run -- ../../assets/planets/pck00011.tpc ../../assets/planets/pck00011.pca ../../assets/planets/gm_de431.tpc`

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args: Vec<String> = env::args().collect();

  if args.len() < 3 || args.len() > 4 {
    eprintln!("Usage: tpc2pca <input.tpc> <output.pca> [gm_input.tpc]");
    eprintln!("");
    eprintln!("  <input.tpc>    Path to the primary Text PCK file (e.g. pck00011.tpc)");
    eprintln!("  <output.pca>   Path to the output PCA file");
    eprintln!("  [gm_input.tpc] Optional path to a Gravity Mass TPC file (e.g. gm_de431.tpc).");
    eprintln!(
      "                 Usually required because Anise needs GM values to construct Planetary datasets."
    );
    std::process::exit(1);
  }

  let input_path = &args[1];
  let output_path = PathBuf::from(&args[2]);

  println!("Parsing planetary data from: {}", input_path);
  let planetary_data = parse_file::<_, TPCItem>(input_path, false)?;

  let gravity_data = if args.len() == 4 {
    let gm_path = &args[3];
    println!("Parsing gravity data from: {}", gm_path);
    parse_file::<_, TPCItem>(gm_path, false)?
  } else {
    println!("No GM TPC provided, using empty gravity data.");
    HashMap::new()
  };

  println!("Converting to PCA dataset...");
  let dataset = convert_tpc_items(planetary_data, gravity_data)?;

  println!("Saving PCA to: {:?}", output_path);
  dataset.save_as(&output_path, true)?;

  println!("Conversion completed successfully.");
  Ok(())
}