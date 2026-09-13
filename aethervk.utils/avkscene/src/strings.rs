//! Centralized strings for the CLI.

pub const CLI_ABOUT: &str = "Aether-Vk Scene Manipulation Utility";
pub const REPL_PROMPT: &str = "(avkscene) ";
pub const REPL_PROMPT_EXIT_CONFIRM: &str = "Unsaved work detected (or Ctrl-C pressed). Are you sure you want to exit? Press again within 1s to confirm.";
pub const ERR_DIR_NOT_FOUND: &str = "Error: Scene directory does not exist.";
pub const ERR_SCENE_FILE_MISSING: &str = "Error: scene.bin not found in directory.";
pub const ERR_INVALID_SCENE: &str = "Error: Invalid scene dump.";
pub const ERR_PATH_NOT_FOUND: &str = "Error: path '{}' doesn't exist.";

#[allow(dead_code)]
pub mod help {
    pub const CMD_QUIT: &str = "Exits the application.";
    pub const CMD_GET_CURSOR: &str = "Prints the current position inside the entity hierarchy.";
    pub const CMD_SET_CURSOR: &str = "Modifies the current position inside the entity hierarchy.";
    pub const CMD_SUMMARY: &str = "Prints a summary of the scene.";
    pub const CMD_PRINT_TREE: &str = "Pretty prints the hierarchy starting from the cursor.";
    pub const CMD_LIST_COMPONENTS: &str = "Prints the list of component names applied to the current entity.";
    pub const CMD_PRINT_COMPONENT: &str = "Prints the details of a specific component.";
}
