mod cli;
mod commands;
mod repl;
mod state;
mod strings;

use aethervk_core_rlib::simulation_api::structs::SerializedEntity;
use miette::{IntoDiagnostic, Result};
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    // Initialize logging/miette
    miette::set_hook(Box::new(|_| {
        Box::new(miette::MietteHandlerOpts::new().terminal_links(true).build())
    })).unwrap();

    let args = cli::parse();

    let mut state = if let Some(path_str) = args.path {
        let path = Path::new(&path_str);
        
        if !path.exists() {
            if args.create_directory {
                fs::create_dir_all(path).into_diagnostic()?;
            } else {
                println!("{}", strings::ERR_DIR_NOT_FOUND);
                std::process::exit(1);
            }
        }
        
        let scene_file = path.join("scene.bin");
        if !scene_file.exists() {
            // For a newly created dir or empty dir, we might want to start with an empty scene.
            // But per specs, if scene dir is NOT empty, we validate.
            // Let's assume an empty scene for now if missing, or we can just report missing.
            let is_empty = fs::read_dir(path).into_diagnostic()?.next().is_none();
            if is_empty {
                state::AppState::new(Vec::new(), Some(path.to_path_buf()))
            } else {
                println!("{}", strings::ERR_SCENE_FILE_MISSING);
                std::process::exit(1);
            }
        } else {
            let bytes = fs::read(&scene_file).into_diagnostic()?;
            let (dump, _): (Vec<SerializedEntity>, usize) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .map_err(|_| miette::miette!(strings::ERR_INVALID_SCENE))?;
            state::AppState::new(dump, Some(path.to_path_buf()))
        }
    } else {
        state::AppState::new(Vec::new(), None) // No path given, start empty
    };

    if args.batch {
        if let Some(cmd_str) = args.command {
            let cmds = cli::parse_line(&cmd_str);
            for arg_vec in cmds {
                if let Ok(cmd) = commands::Command::parse(&arg_vec) {
                    cmd.execute(&mut state, &arg_vec);
                }
            }
        } else if let Some(file_str) = args.file {
            let content = fs::read_to_string(&file_str).into_diagnostic()?;
            if content.starts_with('\u{FEFF}') {
                return Err(miette::miette!("File contains BOM, which is not supported."));
            }
            
            for line in content.lines() {
                let cmds = cli::parse_line(line);
                for arg_vec in cmds {
                    if let Ok(cmd) = commands::Command::parse(&arg_vec) {
                        cmd.execute(&mut state, &arg_vec);
                    }
                }
            }
        }
    } else {
        repl::run_repl(&mut state)?;
    }

    Ok(())
}
