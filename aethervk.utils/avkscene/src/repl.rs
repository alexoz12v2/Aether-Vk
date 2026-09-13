use crate::commands::Command;
use crate::state::AppState;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::time::{Instant, Duration};

pub fn run_repl(state: &mut AppState) -> miette::Result<()> {
    let mut rl = DefaultEditor::new().map_err(|e| miette::miette!("Failed to initialize rustyline: {}", e))?;
    let mut last_ctrl_c = None;
    let mut last_command: Option<Command> = None;

    loop {
        let readline = rl.readline(crate::strings::REPL_PROMPT);
        match readline {
            Ok(line) => {
                rl.add_history_entry(line.as_str()).unwrap_or(false);
                let cmds = crate::cli::parse_line(&line);
                
                if cmds.is_empty() {
                    // If repeatable, execute last command
                    if let Some(cmd) = &last_command {
                        if !cmd.execute(state, &[]) {
                            break;
                        }
                    }
                    continue;
                }

                for arg_vec in cmds {
                    match Command::parse(&arg_vec) {
                        Ok(mut cmd) => {
                            if let Command::Revert { yes } = cmd {
                                if !yes && !state.log.is_empty() {
                                    let confirm = rl.readline("Unsaved changes detected. Revert? (y/n) ");
                                    if let Ok(ans) = confirm {
                                        if ans.trim().to_lowercase() != "y" {
                                            println!("Revert cancelled.");
                                            continue;
                                        }
                                        cmd = Command::Revert { yes: true }; // Override so execute bypasses error
                                    } else {
                                        println!("Revert cancelled.");
                                        continue;
                                    }
                                }
                            }
                            
                            if let Command::AddComponent { name: _, ref mut file, eol: Some(ref marker) } = cmd {
                                let mut payload = String::new();
                                loop {
                                    let heredoc_line = rl.readline("(avkscene) > ");
                                    match heredoc_line {
                                        Ok(l) => {
                                            if l.trim() == marker {
                                                break;
                                            }
                                            payload.push_str(&l);
                                            payload.push('\n');
                                        }
                                        Err(_) => {
                                            break;
                                        }
                                    }
                                }
                                // Package payload as a fake file path starting with PAYLOAD:
                                *file = Some(format!("PAYLOAD:{}", payload));
                            }

                            let is_repeatable = match cmd {
                                Command::Quit | Command::GetCursor => true,
                                _ => false,
                            };
                            
                            let should_continue = cmd.execute(state, &arg_vec);
                            
                            if is_repeatable {
                                last_command = Some(cmd);
                            } else {
                                last_command = None;
                            }
                            
                            if !should_continue {
                                return Ok(());
                            }
                        }
                        Err(e) => {
                            println!("{}", e);
                        }
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C
                let now = Instant::now();
                if let Some(last) = last_ctrl_c {
                    if now.duration_since(last) < Duration::from_secs(1) {
                        break;
                    }
                }
                println!("{}", crate::strings::REPL_PROMPT_EXIT_CONFIRM);
                last_ctrl_c = Some(now);
            }
            Err(ReadlineError::Eof) => {
                // Ctrl-D
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}
