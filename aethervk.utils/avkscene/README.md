# `avkscene` utility binary specification

`avkscene` will be a command line utility to manipulate serialized `aethervk_core_rlib::scene::Scene` objects.

It will behave similarly to `gdb`, meaning it will have a small REPL to work on the scene directory.

## Command Line Ergonomics

We interpret this section as how should command line arguments be given by the user for the application. We loosely base ourselves on
[Abseil](https://abseil.io/docs/cpp/guides/flags), plus usual command line flags behavior of GNU `coreutils`. This means that

- We distinguish between *positional arguments* and *keyed arguments* (called hencefourth "*flags*")
- positional arguments, if mixed with flags, should come
  - before a keyed argument if the given flag acts on the specified argument, or before any positional argument if it is a global option

    ```sh
    # the -c flag "--create-directory" acts on the directory positional argument. Creates the directory if it doesn't exist
    # note: a "subcommand" is considered a positional argument
    avkscene path/to/scenedir -c
    # the "--log-verbosity" flag acts on the command as a whole, therefore should precede all positional arguments
    # directory not specified, current directory = scene directory assumed. Note that you can't use -c (not that you need to)
    avkscene --log-verbosity=trace
    ```

  - some positional arguments may be defaulted if absent
- keyed arguments may come in *extended form* (--), and *contracted form* (-). Contracted form needs to be 1 letter. all keyed arguments
  are of the form `--keyed-argument=value` or `-k=value` (contracted flag form allows for `-k value` too).
  - a special case are *binary keyed arguments*, ie whose allowed values are only `true` or `false`, and whose absence means `false`, because
    - their contracted forms can be *merged*. Example, to apply `-c` and `-v` together, assuming they are contracted forms of keyed
      arguments, you can write `-cv`
  - non binary flags can have a contracted form (not mandatory), but they cannot be merged

- some non binary flags can introduce new valid flags (*keyed subcommands*), and the first positional argument of the CLI command may be
  a *positional subcommand*. we've decided to go exclusively with *keyed subcommands*.
  - a *keyed subcommand* has a *scope* in which the "dictionary" of valid positional arguments and keyed arguments change.
    By default, once you enter the keyed subcommand flag, you've entered the keyed subcommand scope, it lasts until the eol. Therefore,
    if you need to give flags or arguments outside the scope, you either need to position them beforehand, or place
    a ";" (escaped/stringified) in the command line, which signals the end of the scope.

    ```sh
    avkscene this/scene -c --batch -c 'get-cursor ; add-entity -name="the-entity" ; quit'
    ```

    Note in the example how the inner `-c` flag (expanded `--command`)

- keyed subcommands can hijack and modify the behavior of the main command. Example, a `--help` keyed subcommand doesn't require a
  scene directory, cause its only job is to print the command syntax

## Generals and code style

- any string which needs to be displayed to the screen needs to be stored inside its own rust module, called `strings.rs`. The static
  strings in there can have formatting arguments, and may be nested into modules logically grouped.

## Timeline

Total Development time: 1 day per feature block, so pretty narrow. Use as much libraries as possible

### Block 1: basics (12/09/2026)

- [x] project structure
- [] initial readme (this file)
- [] implementation for CLI Ergonomics
- [] green threads library like tokio may give performance boost, try to use it (async rust)
- [] implement block "Basic Commands"
  - [] tab autocompletion in `set-cursor`

## Requirements and development plan

The `avkscene` binary needs to be able to open serialized scenes. Since it is a command utility, the first step of development would
be to find libraries and impelment "Command Line Ergonomics" efficiently. As an example, we don't want to have to manually edit the `--help` subcommand whenever we modify some syntax

The behavior of the main command is simple

- check whether any hijacking keyed subcommands have been given (only `--help` is now planned, more in future)
  - if `--help`, then print help (possibly positional argument to narrow down to some help sections, not strictly necessary to filter cause
    the user can `grep` the output, but to give more detail on a particular command)
- check whether the given directory exists.
  - If it does not
    - if `--create-directory` flag is `true` -> try to create directory and all its ancestors. If fail, report native error and exit
    - Otherwise: fail, report error missing directory (come up with a fancy error message later)

if all these checks pass, check for any non-hijacking keyed subcommand. Right now the only one we have is `--batch` (keyed subcommands
are not compressed, not by ergonomics rules, but by arbitrary choice) (we'll refer to this subcommand as *batch mode*).
- if `--batch` is present, then introduce `--file` and `--command` flags. They are _mutually exclusive_.
  - if file mode, then the list of commands to execute is contained in a file. End of command is either EOL, or ";".
    - file we handle are exclusively _UTF-8 no BOM_. If, while opening the file, a bom is detected, reject it and report error. Otherwise,
      assume UTF-8 and try the parse function
  - if command mode, then the list of command is given by a string given as value of the flag itself.

If we are not in batch mode, then we enter a gdb-style REPL with `(avkscene) ` as prompt. The following are the *REPL ergonomics*:

- CTRL + Z -> follow gdb behavior (when subprocess not running)
- CTRL + C -> ask the user for confirmation, and if user presses the combination again within 1 second, close the application. How is the
  terminal line handled in this case? Std out and then new repl line. Example:

  ```sh
  (avkscene)
  [CTRL + C] detected. Are you sure you want to exit?
  (avkscene)
  _>
  ```

  (we displayed terminal prompt cause supposedly the user pressed exit again)

## Starting up (*Scene Validation*)

After the application checks that the directory exists, before entering repl or batch mode, if the chosen *Scene Directory* is not empty,
then we assume it is a valid scene according to `aethervk_core_rlib::simulation_api::scene_dump`. Therefore, we try out a quick function
to assess whether
- the scene file exists and is readable by the current user
- the scene file is valid (there should be a header/version)
- there are no other "alien files" in the directory other than what scene serialization provides

(deepen this paragraph if more startup validation is needed)

- After startup and scene exists:
  - we need to load the scene hierarchy in memory for *Path Autocompletion* with TAB

## 2-phase execution

Execution for every command, whether in batch mode or in REPL mode, needs to happen in two phases

- Parsing phase
- Execution phase

In the parsing phase, we tokenize and validate each and every command
- is it a valid command?
- are all required parameters (including their default values and values injected by global state) filled out?
- do each parameter have a valid value? (validation function)
- we do not check for existance of a command, only syntax

If a command is well formed, then it is marshalled into an callable object which will be invoked in the second phase

In the execution phase, we start executing each command, stopping when a runtime error (eg trying to query components for a non existing
entity) and report it.
- If the command was executed from a file (batch, file mode), we need to report the file location in which the error happened, in the
  form filename:line, meaning that the parsing phase also need to attach metadata to each callable object

## Basic command syntax

- (REPL Only) pressing enter on a *Repeatable command* will let it execute again. This can happen only once
  - each command here is tagged with "REPL Repeatable"
- Needs save is for later
- Each command might manipulate a global state. Since we are designing and developing the CLI tool at the same time, we don't have a
  list of global states here, so the first declared command which manipulates this state will have a section dedicated to explain how
  it works

## Marker Paragraph

From this paragraph onwards we will describe what the various commands do, and whether they introduce some global state and how
to maintain it (eg *in-memory-scene*, *cursor*, *unsaved-operations-log*)

## Basic Commands

### `quit`, `exit`, `q`

- REPL repeatable: Yes
- Needs save: No
- Arguments: None

In REPL Mode, exits the application. In Batch Mode, it does the same, with the additional validation that it can only be the last command

- Hook registration on exit (for later, but needs to be there)
  - example: if you have unsaved work, display a message and then if the user confirms with another execution of the quit command
    (meaning another enter since repeatable)

    ```
    (avkscene) quit
    Unsaved work detected. Are you sure you want to exit?
    (avkscene)
    _>
    ```

    (The user here pressed Enter)

## `get-cursor`, `pc` (print cursor)

- REPL repeatable: Yes
- Needs save: No
- global state
  - name: *cursor*
  - affect type: Read
- Arguments: None

Prints to stdout the current position inside the entity hierarchy. Since we only have a single root entity, it is printed as `/`.
Otherwise, name of entities are printed as path inside the scene.

- *cursor* global state: `aethervk_core_rlil::scene::EntityId` and its associated `alloc::string::String` which represents its full path
 - if an entity subtree gets deleted from the in-memory state, the cursor gets snapped to the first still existing ancestor

## `set-cursor`, `sc` (set cursor)

- REPL repeatable: No
- Needs save: No
- global state
  - name: *cursor*
  - affect type: Write
- Arguments (positional):
  - `path`
    - required: yes
    - type: string (entity path)

Modifies the current position inside the entity hierarchy. *TAB Autocompletion* should work here. If you type something which doesn't
exist, print to stdout error path {} doesn't exist

- TAB Autocompletion: Should work by completing paths for you. It works with the in-memory state of the scene. (not the saved one)

## `summary` (`smy`)

- REPL repeatable: No
- Needs save: No
- Arguments: None

## `print-tree` (`pt`)

- REPL repeatable: No
- Needs save: No
- global state
  - name: *cursor*
  - affect type: Read
- Arguments (positional):
  - `max depth`
    - required: no
    - default: 0 (means infinite)
    - type: number (positive or zero, integer)

Pretty prints the hierarchy, where each node is the entity name, starting from the entity on which the *cursor* is positioned on

## `list-components` (`lc`)

- REPL repeatable: No
- Needs save: No
- global state
  - name: *cursor*
  - affect type: Read
- Arguments: None

Prints the list of component names applied to the current entity, in lexicographical order, of the entity the cursor is placed on

## `print-component` (`c`)

- REPL repeatable: No
- Needs save: No
- global state
  - name: *cursor*
  - affect type: Read
- Arguments (positional):
  - `component name`
    - required: true
    - type: string (enum, valid component name present on the entity)

Prints the details (like we do in avkSimulationContext_debugECSPrint). Implement any pretty print function for each component here

- TAB Autocompletion: on component names present in this entity

## Scene State: In-Memory vs File-Backed (Transaction Semantics)

`avkscene` operates on an *in-memory* copy of the scene graph, treating the directory's `scene.bin` as a persistent, file-backed state. This model works similarly to SQL transactions. 
Any modifications made using commands (e.g., adding/deleting entities or components) only mutate the in-memory state. 
- Opening or creating an empty directory initializes an in-memory scene with a single `root` entity, with no file-backing present until saved.
- **Save**: Writes the current in-memory state to disk (`scene.bin`), establishing a new file-backed state.
- **Revert**: Discards all in-memory changes, restoring the scene to the last file-backed state on disk.

---

## State & Tracking Commands

### `save`
- REPL repeatable: No
- Needs save: No
- Arguments: None

Commits the current in-memory scene to the file-backed directory (`scene.bin`). Empties the unsaved-operations log.
- **State Constraints**: If the scene fails validation (e.g., corrupted tree hierarchy), the save operation is aborted.
- **Edge Cases**: If directory permissions are read-only, fails immediately. If there are no unsaved changes in the in-memory state, the command succeeds but skips disk I/O.
- **Output Formatting**: Prints a success message such as `[SUCCESS] Scene saved. 0 operations pending.`
- **Error Messages**: 
  - `[ERROR] Permission denied: Cannot write to scene.bin.`
  - `[ERROR] Validation failed: <reason>. Save aborted.`

### `revert`
- REPL repeatable: No
- Needs save: No
- Flags:
  - `--yes`, `-y`: Bypasses the confirmation prompt when unsaved changes exist.
- Arguments: None

Rolls back the in-memory scene to match the last file-backed state on disk.
- **State Constraints**: Clears the unsaved-operations log entirely. If the cursor is positioned on an entity that is removed by the revert, the cursor snaps back to the `root` (`/`) entity.
- **Edge Cases**: If the directory is empty (no `scene.bin` exists yet), reverting will reset the in-memory state to a single empty `root` entity. If there are no unsaved changes, it safely does nothing.
- **Output Formatting**: `[INFO] Reverted <N> unsaved operations. Current cursor is at <path>.`
- **Error Messages**: (Non-failing, but if invoked in REPL mode with pending changes and without `-y`, it will prompt: `Unsaved changes detected. Revert? (y/n)`)

### `diff`
- REPL repeatable: No
- Needs save: No
- Arguments: None

Prints the difference between the last saved file-backed state and the current in-memory state.
- **Edge Cases**: If there are no unsaved operations, simply prints "No changes." Binary component properties that differ will not dump raw binary to the console; instead, they will display as `<Binary Data Modified>`.
- **Output Formatting**: Uses a unified diff-like structure. 
  - Additions are prefixed with `+` (colored green in terminal).
  - Deletions are prefixed with `-` (colored red in terminal).
  - Modified properties within components show `OldValue -> NewValue`.
  - Grouped by entity path: 
    ```text
    Entity [/root/Player]:
      + Added Component: Transform
      ~ Modified Component: Health [ hp: 100 -> 80 ]
    ```

### `log`
- REPL repeatable: No
- Needs save: No
- global state:
  - name: *unsaved-operations-log*
  - affect type: Read
- Flags:
  - `--limit`, `-n`: Limits the output to the last N commands (default: 50).
- Arguments: None

Prints a chronological list of mutating commands executed since the last `save` or since the CLI started.
- **Edge Cases**: If no modifications have been made, outputs `Log is empty. No unsaved operations.`
- **Output Formatting**: Numbered chronologically, displaying the command name, arguments, and the cursor path at the time of execution.
  ```text
  [1] add-entity "Player" (cursor: /root)
  [2] add-component "Transform" (cursor: /root/Player)
  ```

---

## Modification Commands

### `add-entity`
- REPL repeatable: No
- Needs save: Yes
- Arguments (positional):
  - `name`
    - required: true
    - type: string

Creates a new entity with the given name as a child of the entity the *cursor* is currently positioned at.
- **State Constraints**: The name cannot contain path-separator characters (e.g., `/` or `\`). 
- **Edge Cases**: If a sibling entity with the exact same name already exists under the current parent, the command will fail to prevent path ambiguity.
- **Output Formatting**: `[SUCCESS] Created entity at path /.../<name>.`
- **Error Messages**: 
  - `[ERROR] Invalid entity name: cannot contain '/' characters.`
  - `[ERROR] Entity with name '<name>' already exists under <current_cursor_path>.`

### `delete-entity`
- REPL repeatable: No
- Needs save: Yes
- Flags:
  - `--force`, `-f`: If present, deletes the entire subtree (entity, components, and all children).
- Arguments: None

Deletes the entity the *cursor* is currently positioned at. 
- **State Constraints**: Upon successful deletion, the cursor automatically snaps to the nearest surviving ancestor (usually the parent).
- **Edge Cases**: Trying to delete the `root` entity (`/`) is strictly prohibited and will always fail, even with `--force`. 
- **Error Messages**: 
  - `[ERROR] Cannot delete the root entity.`
  - `[ERROR] Entity has <X> children and <Y> components. Use --force to delete.`
- **Output Formatting**: `[SUCCESS] Deleted entity <path>. Cursor snapped to <new_path>.`

### `add-component`
- REPL repeatable: No
- Needs save: Yes
- Flags:
  - `--file`: Path to a JSON or RON file containing the component data.
  - `--eol`: Bash-like heredoc marker (e.g. `EOF`). Reads input lines until the marker is encountered. In REPL mode, when reading input via `--eol`, the prompt temporarily changes to `(avkscene) > ` until the EOF marker is reached.
- Arguments (positional):
  - `name`
    - required: true
    - type: string (component name)

Attaches a new component to the entity under the cursor using data from a file or inline multi-line input.
- **State Constraints**: The component name must map to a valid, registered component schema in the engine. All required fields in the parsed format must be present.
- **Edge Cases**: If the component already exists on the entity, the command will fail.
- **Error Messages**: 
  - `[ERROR] Component '<name>' already exists on this entity.`
  - `[ERROR] Unknown component type '<name>'.`
  - `[ERROR] Data parse error at line <L>: missing required field '<field>'.`
- **Output Formatting**: `[SUCCESS] Attached component '<name>' to <current_cursor_path>.`

### `delete-component`
- REPL repeatable: No
- Needs save: Yes
- Arguments (positional):
  - `name`
    - required: true
    - type: string (component name)

Removes the specified component from the entity under the cursor. Supports TAB autocompletion for currently attached components.
- **Edge Cases**: Attempting to delete a component that doesn't exist on the current entity immediately fails. If the engine architecture defines certain components as 'locked' or core to an entity (e.g. UUID), deleting them may also be forbidden.
- **Error Messages**:
  - `[ERROR] Component '<name>' not found on entity <current_cursor_path>.`
  - `[ERROR] Cannot delete locked/required component '<name>'.`
- **Output Formatting**: `[SUCCESS] Removed component '<name>' from <current_cursor_path>.`
