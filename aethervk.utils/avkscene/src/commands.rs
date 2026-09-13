use crate::state::AppState;
use aethervk_core_rlib::simulation_api::structs::SerializedComponent;
use miette::{Diagnostic, Result};
use thiserror::Error;
use std::fs;

#[derive(Error, Debug, Diagnostic)]
pub enum CommandError {
    #[error("Unknown command: {0}")]
    UnknownCommand(String),
    #[error("Missing argument: {0}")]
    MissingArgument(String),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

#[derive(Clone, Debug)]
pub enum Command {
    Quit,
    GetCursor,
    SetCursor(String),
    Summary,
    PrintTree(u32),
    ListComponents,
    PrintComponent(String),
    Save,
    Revert { yes: bool },
    Diff,
    Log { limit: usize },
    AddEntity(String),
    DeleteEntity { force: bool },
    AddComponent { name: String, file: Option<String>, eol: Option<String> },
    DeleteComponent(String),
}

impl Command {
    pub fn parse(args: &[String]) -> Result<Self, CommandError> {
        if args.is_empty() {
            return Err(CommandError::UnknownCommand(String::new()));
        }

        match args[0].as_str() {
            "quit" | "exit" | "q" => Ok(Command::Quit),
            "get-cursor" | "pc" => Ok(Command::GetCursor),
            "set-cursor" | "sc" => {
                if args.len() < 2 {
                    return Err(CommandError::MissingArgument("path".to_string()));
                }
                Ok(Command::SetCursor(args[1].clone()))
            }
            "summary" | "smy" => Ok(Command::Summary),
            "print-tree" | "pt" => {
                let depth = if args.len() > 1 {
                    args[1].parse().map_err(|_| CommandError::InvalidArgument("max depth must be integer".into()))?
                } else {
                    0
                };
                Ok(Command::PrintTree(depth))
            }
            "list-components" | "lc" => Ok(Command::ListComponents),
            "print-component" | "c" => {
                if args.len() < 2 {
                    return Err(CommandError::MissingArgument("component name".to_string()));
                }
                Ok(Command::PrintComponent(args[1].clone()))
            }
            "save" => Ok(Command::Save),
            "revert" => {
                let yes = args.iter().any(|a| a == "--yes" || a == "-y");
                Ok(Command::Revert { yes })
            }
            "diff" => Ok(Command::Diff),
            "log" => {
                let mut limit = 50;
                let mut iter = args.iter().skip(1);
                while let Some(arg) = iter.next() {
                    if arg == "--limit" || arg == "-n" {
                        if let Some(val) = iter.next() {
                            limit = val.parse().map_err(|_| CommandError::InvalidArgument("limit must be integer".into()))?;
                        }
                    }
                }
                Ok(Command::Log { limit })
            }
            "add-entity" => {
                let name = args.iter().skip(1).find(|a| !a.starts_with('-')).ok_or_else(|| CommandError::MissingArgument("name".to_string()))?;
                Ok(Command::AddEntity(name.clone()))
            }
            "delete-entity" => {
                let force = args.iter().any(|a| a == "--force" || a == "-f");
                Ok(Command::DeleteEntity { force })
            }
            "add-component" => {
                let mut file = None;
                let mut eol = None;
                let mut name = None;
                
                let mut iter = args.iter().skip(1);
                while let Some(arg) = iter.next() {
                    if arg == "--file" {
                        if let Some(val) = iter.next() {
                            file = Some(val.clone());
                        }
                    } else if arg.starts_with("--file=") {
                        file = Some(arg.trim_start_matches("--file=").to_string());
                    } else if arg == "--eol" {
                        if let Some(val) = iter.next() {
                            eol = Some(val.clone());
                        }
                    } else if arg.starts_with("--eol=") {
                        eol = Some(arg.trim_start_matches("--eol=").to_string());
                    } else if !arg.starts_with('-') && name.is_none() {
                        name = Some(arg.clone());
                    }
                }
                
                let name = name.ok_or_else(|| CommandError::MissingArgument("name".to_string()))?;
                Ok(Command::AddComponent { name, file, eol })
            }
            "delete-component" => {
                let name = args.iter().skip(1).find(|a| !a.starts_with('-')).ok_or_else(|| CommandError::MissingArgument("name".to_string()))?;
                Ok(Command::DeleteComponent(name.clone()))
            }
            other => Err(CommandError::UnknownCommand(other.to_string())),
        }
    }

    pub fn execute(&self, state: &mut AppState, args_raw: &[String]) -> bool {
        match self {
            Command::Quit => {
                return false;
            }
            Command::GetCursor => {
                println!("{}", state.cursor_path);
            }
            Command::SetCursor(path) => {
                if !state.set_cursor(path) {
                    println!("{}", crate::strings::ERR_PATH_NOT_FOUND.replace("{}", path));
                }
            }
            Command::Summary => {
                println!("Scene Entities: {}", state.entities.len());
                if let Some(root) = state.root_id {
                    println!("Root ID: {}", root);
                }
            }
            Command::PrintTree(max_depth) => {
                if let Some(start_id) = state.cursor_id {
                    Self::print_node(state, start_id, 0, *max_depth);
                } else {
                    println!("Cursor is not set on a valid entity.");
                }
            }
            Command::ListComponents => {
                if let Some(cursor_id) = state.cursor_id {
                    if let Some(node) = state.entities.get(&cursor_id) {
                        let mut components: Vec<String> = node.entity.components
                            .iter()
                            .map(|c| AppState::get_comp_name(c))
                            .collect();
                        components.sort();
                        for comp in components {
                            println!("- {}", comp);
                        }
                    }
                }
            }
            Command::PrintComponent(name) => {
                if let Some(cursor_id) = state.cursor_id {
                    if let Some(node) = state.entities.get(&cursor_id) {
                        let mut found = false;
                        for comp in &node.entity.components {
                            let comp_name = AppState::get_comp_name(comp);
                            if comp_name == *name {
                                println!("{:#?}", comp);
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            println!("Component '{}' not found on current entity.", name);
                        }
                    }
                }
            }
            Command::Save => {
                match state.save() {
                    Ok(_) => println!("[SUCCESS] Scene saved. 0 operations pending."),
                    Err(e) => println!("[ERROR] {:?}", e),
                }
            }
            Command::Revert { yes } => {
                if !yes && !state.log.is_empty() {
                    // This interactive prompt part should technically happen before execute if possible, 
                    // or we handle it here if it's the REPL.
                    // But in execute we just do it. Wait, the repl loop already handles input. 
                    // Let's assume repl loop handles the prompt, or we just do it here if possible.
                    // To keep `execute` simple and non-interactive, I'll let `repl.rs` intercept `Revert` without `yes` if there are changes.
                    // Actually, if we get here without `yes` and there are changes, it's an error in batch mode.
                    println!("[ERROR] Unsaved changes detected. Use --yes or -y to revert.");
                    return true;
                }
                
                let count = state.log.len();
                state.revert();
                println!("[INFO] Reverted {} unsaved operations. Current cursor is at {}.", count, state.cursor_path);
            }
            Command::Diff => {
                state.diff();
            }
            Command::Log { limit } => {
                if state.log.is_empty() {
                    println!("Log is empty. No unsaved operations.");
                } else {
                    let start = state.log.len().saturating_sub(*limit);
                    for (i, entry) in state.log.iter().enumerate().skip(start) {
                        println!("[{}] {}", i + 1, entry);
                    }
                }
            }
            Command::AddEntity(name) => {
                if let Err(e) = state.add_entity(name) {
                    println!("{}", e);
                } else {
                    let log_entry = format!("{} (cursor: {})", args_raw.join(" "), state.cursor_path);
                    state.log.push(log_entry);
                }
            }
            Command::DeleteEntity { force } => {
                if let Err(e) = state.delete_entity(*force) {
                    println!("{}", e);
                } else {
                    let log_entry = format!("{} (cursor: {})", args_raw.join(" "), state.cursor_path);
                    state.log.push(log_entry);
                }
            }
            Command::AddComponent { name, file, eol: _ } => {
                let data_str = if let Some(f) = file {
                    if f.starts_with("PAYLOAD:") {
                        f.trim_start_matches("PAYLOAD:").to_string()
                    } else {
                        match fs::read_to_string(f) {
                            Ok(s) => s,
                            Err(e) => {
                                println!("[ERROR] Could not read file: {}", e);
                                return true;
                            }
                        }
                    }
                } else {
                    println!("[ERROR] No data provided for component.");
                    return true;
                };

                let ron_str = if data_str.trim().starts_with(&format!("{}(", name)) {
                    data_str.clone()
                } else {
                    format!("{}({})", name, data_str.trim())
                };

                match ron::from_str::<SerializedComponent>(&ron_str) {
                    Ok(comp) => {
                        if let Err(e) = state.add_component(comp) {
                            println!("{}", e);
                        } else {
                            let log_entry = format!("{} (cursor: {})", args_raw.join(" "), state.cursor_path);
                            state.log.push(log_entry);
                        }
                    }
                    Err(e) => {
                        println!("[ERROR] Data parse error: {}", e);
                    }
                }
            }
            Command::DeleteComponent(name) => {
                if let Err(e) = state.delete_component(name) {
                    println!("{}", e);
                } else {
                    let log_entry = format!("{} (cursor: {})", args_raw.join(" "), state.cursor_path);
                    state.log.push(log_entry);
                }
            }
        }
        true // Continue execution
    }

    fn print_node(state: &AppState, node_id: u64, depth: u32, max_depth: u32) {
        if max_depth > 0 && depth > max_depth {
            return;
        }
        
        if let Some(node) = state.entities.get(&node_id) {
            let indent = "  ".repeat(depth as usize);
            println!("{}- {}", indent, node.entity.name);
            for child_id in &node.children {
                Self::print_node(state, *child_id, depth + 1, max_depth);
            }
        }
    }
}
