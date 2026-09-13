#[derive(Debug, Default)]
pub struct Cli {
    pub path: Option<String>,
    pub create_directory: bool,
    pub batch: bool,
    pub file: Option<String>,
    pub command: Option<String>,
}

pub fn parse() -> Cli {
    let mut args = std::env::args().skip(1).peekable();
    let mut cli = Cli::default();
    
    // Parse global flags and positional path
    while let Some(arg) = args.next() {
        if arg == "-c" || arg == "--create-directory" {
            cli.create_directory = true;
        } else if arg == "--batch" {
            cli.batch = true;
            break; // Enter batch scope
        } else if !arg.starts_with('-') && cli.path.is_none() {
            cli.path = Some(arg);
        } else {
            // Ignore unknown globals for now or handle them
        }
    }
    
    // If we entered batch scope, parse batch-specific flags
    if cli.batch {
        while let Some(arg) = args.next() {
            if arg == "-c" || arg == "--command" {
                if let Some(cmd) = args.next() {
                    cli.command = Some(cmd);
                }
            } else if arg == "-f" || arg == "--file" {
                if let Some(file) = args.next() {
                    cli.file = Some(file);
                }
            } else if arg == ";" {
                break; // End of batch scope
            }
        }
    }
    
    // Parse any remaining arguments if needed (e.g., path after scope)
    while let Some(arg) = args.next() {
        if !arg.starts_with('-') && cli.path.is_none() {
            cli.path = Some(arg);
        }
    }
    
    cli
}

/// Helper to parse a single line of command in REPL or Batch mode.
/// Returns a vector of commands (split by `;`) where each is a vector of args.
pub fn parse_line(line: &str) -> Vec<Vec<String>> {
    let mut commands = Vec::new();
    let mut current_command = Vec::new();
    let mut current_arg = String::new();
    let mut in_quotes = false;
    let mut escape = false;

    for c in line.chars() {
        if escape {
            current_arg.push(c);
            escape = false;
        } else if c == '\\' {
            escape = true;
        } else if c == '"' {
            in_quotes = !in_quotes;
        } else if c == ';' && !in_quotes {
            if !current_arg.is_empty() {
                current_command.push(current_arg.clone());
                current_arg.clear();
            }
            if !current_command.is_empty() {
                commands.push(current_command.clone());
                current_command.clear();
            }
        } else if c.is_whitespace() && !in_quotes {
            if !current_arg.is_empty() {
                current_command.push(current_arg.clone());
                current_arg.clear();
            }
        } else {
            current_arg.push(c);
        }
    }

    if !current_arg.is_empty() {
        current_command.push(current_arg);
    }
    if !current_command.is_empty() {
        commands.push(current_command);
    }

    commands
}
