use aethervk_core_rlib::simulation_api::structs::{SerializedComponent, SerializedEntity};
use std::collections::HashMap;
use std::path::PathBuf;
use std::fs;
use miette::{IntoDiagnostic, Result};

#[derive(Debug, Clone)]
pub struct EntityNode {
    pub entity: SerializedEntity,
    pub children: Vec<u64>,
}

pub struct AppState {
    pub entities: HashMap<u64, EntityNode>,
    pub root_id: Option<u64>,
    pub cursor_id: Option<u64>,
    pub cursor_path: String,
    
    pub file_backed: Vec<SerializedEntity>,
    pub log: Vec<String>,
    pub scene_path: Option<PathBuf>,
}

impl AppState {
    pub fn new(dump: Vec<SerializedEntity>, scene_path: Option<PathBuf>) -> Self {
        let mut state = Self {
            entities: HashMap::new(),
            root_id: None,
            cursor_id: None,
            cursor_path: String::from("/"),
            file_backed: Vec::new(),
            log: Vec::new(),
            scene_path,
        };
        state.rebuild_from_dump(dump);
        state.file_backed = state.to_dump();
        state
    }
    
    pub fn rebuild_from_dump(&mut self, dump: Vec<SerializedEntity>) {
        self.entities.clear();
        let mut children_map: HashMap<u64, Vec<u64>> = HashMap::new();
        self.root_id = None;
        
        for entity in &dump {
            if entity.parent_ffi_id.is_none() || entity.name == "root" || entity.name == "/" {
                self.root_id = Some(entity.ffi_id);
            }
            if let Some(parent_id) = entity.parent_ffi_id {
                children_map.entry(parent_id).or_default().push(entity.ffi_id);
            }
        }
        
        if dump.is_empty() {
            let root = SerializedEntity {
                ffi_id: 1, // Start with 1, as 0 might be invalid
                name: "root".to_string(),
                parent_ffi_id: None,
                components: Vec::new(),
            };
            self.entities.insert(1, EntityNode { entity: root, children: Vec::new() });
            self.root_id = Some(1);
        } else {
            for entity in dump {
                let ffi_id = entity.ffi_id;
                let children = children_map.remove(&ffi_id).unwrap_or_default();
                self.entities.insert(ffi_id, EntityNode { entity, children });
            }
        }
        
        if self.root_id.is_none() {
            self.root_id = self.entities.keys().next().copied();
        }
        self.cursor_id = self.root_id;
        if let Some(id) = self.cursor_id {
            self.cursor_path = self.get_entity_path(id);
        } else {
            self.cursor_path = String::from("/");
        }
    }

    pub fn to_dump(&self) -> Vec<SerializedEntity> {
        let mut dump = Vec::new();
        for node in self.entities.values() {
            dump.push(node.entity.clone());
        }
        dump
    }

    pub fn save(&mut self) -> Result<()> {
        if let Some(path) = &self.scene_path {
            let dump = self.to_dump();
            // Validate? (For now, assume valid)
            let scene_file = path.join("scene.bin");
            let bytes = bincode::serde::encode_to_vec(&dump, bincode::config::standard())
                .map_err(|e| miette::miette!("Failed to serialize scene: {}", e))?;
            fs::write(scene_file, bytes).into_diagnostic()?;
            
            self.file_backed = dump;
            self.log.clear();
            Ok(())
        } else {
            Err(miette::miette!("No scene directory specified. Cannot save."))
        }
    }
    
    pub fn revert(&mut self) {
        self.rebuild_from_dump(self.file_backed.clone());
        self.log.clear();
    }
    
    pub fn diff(&self) {
        if self.log.is_empty() {
            println!("No changes.");
            return;
        }
        
        let old_map: HashMap<u64, &SerializedEntity> = self.file_backed.iter().map(|e| (e.ffi_id, e)).collect();
        let new_map: HashMap<u64, &EntityNode> = self.entities.iter().map(|(k, v)| (*k, v)).collect();
        
        for (id, new_node) in &new_map {
            let path = self.get_entity_path(*id);
            if let Some(old_entity) = old_map.get(id) {
                // Modified
                let mut added_comps = Vec::new();
                let mut removed_comps = Vec::new();
                
                let old_comps: HashMap<String, &SerializedComponent> = old_entity.components.iter().map(|c| (Self::get_comp_name(c), c)).collect();
                let new_comps: HashMap<String, &SerializedComponent> = new_node.entity.components.iter().map(|c| (Self::get_comp_name(c), c)).collect();
                
                for comp_name in new_comps.keys() {
                    if !old_comps.contains_key(comp_name) {
                        added_comps.push(comp_name);
                    }
                }
                for comp_name in old_comps.keys() {
                    if !new_comps.contains_key(comp_name) {
                        removed_comps.push(comp_name);
                    }
                }
                
                if !added_comps.is_empty() || !removed_comps.is_empty() {
                    println!("Entity [{}]:", path);
                    for c in added_comps {
                        println!("  \x1b[32m+ Added Component: {}\x1b[0m", c);
                    }
                    for c in removed_comps {
                        println!("  \x1b[31m- Removed Component: {}\x1b[0m", c);
                    }
                }
            } else {
                // Added
                println!("Entity [{}]:", path);
                println!("  \x1b[32m+ Added Entity\x1b[0m");
            }
        }
        
        for (id, _old_entity) in &old_map {
            if !new_map.contains_key(id) {
                println!("Entity [ID: {}]:", id);
                println!("  \x1b[31m- Removed Entity\x1b[0m");
            }
        }
    }
    
    pub fn get_comp_name(c: &SerializedComponent) -> String {
        let debug_str = format!("{:?}", c);
        debug_str.split('(').next().unwrap_or(&debug_str).to_string()
    }
    
    pub fn add_entity(&mut self, name: &str) -> Result<()> {
        if name.contains('/') || name.contains('\\') {
            return Err(miette::miette!("[ERROR] Invalid entity name: cannot contain '/' characters."));
        }
        
        if let Some(cursor_id) = self.cursor_id {
            if let Some(parent) = self.entities.get(&cursor_id) {
                for child_id in &parent.children {
                    if let Some(child) = self.entities.get(child_id) {
                        if child.entity.name == name {
                            return Err(miette::miette!("[ERROR] Entity with name '{}' already exists under {}.", name, self.cursor_path));
                        }
                    }
                }
            }
            
            let new_id = self.entities.keys().copied().max().unwrap_or(0) + 1;
            let new_entity = SerializedEntity {
                ffi_id: new_id,
                name: name.to_string(),
                parent_ffi_id: Some(cursor_id),
                components: Vec::new(),
            };
            
            self.entities.insert(new_id, EntityNode { entity: new_entity, children: Vec::new() });
            if let Some(parent) = self.entities.get_mut(&cursor_id) {
                parent.children.push(new_id);
            }
            
            println!("[SUCCESS] Created entity at path {}/{}.", if self.cursor_path == "/" { "" } else { &self.cursor_path }, name);
            Ok(())
        } else {
            Err(miette::miette!("Cursor is not valid."))
        }
    }
    
    pub fn delete_entity(&mut self, force: bool) -> Result<()> {
        if let Some(cursor_id) = self.cursor_id {
            if Some(cursor_id) == self.root_id {
                return Err(miette::miette!("[ERROR] Cannot delete the root entity."));
            }
            
            if let Some(node) = self.entities.get(&cursor_id) {
                if !force && (!node.children.is_empty() || !node.entity.components.is_empty()) {
                    return Err(miette::miette!("[ERROR] Entity has {} children and {} components. Use --force to delete.", node.children.len(), node.entity.components.len()));
                }
                
                let parent_id = node.entity.parent_ffi_id;
                
                // Delete subtree
                let mut to_delete = vec![cursor_id];
                let mut i = 0;
                while i < to_delete.len() {
                    let id = to_delete[i];
                    if let Some(n) = self.entities.get(&id) {
                        to_delete.extend(&n.children);
                    }
                    i += 1;
                }
                
                for id in &to_delete {
                    self.entities.remove(id);
                }
                
                if let Some(pid) = parent_id {
                    if let Some(parent) = self.entities.get_mut(&pid) {
                        parent.children.retain(|&x| x != cursor_id);
                    }
                    self.cursor_id = Some(pid);
                    self.cursor_path = self.get_entity_path(pid);
                    println!("[SUCCESS] Deleted entity. Cursor snapped to {}.", self.cursor_path);
                }
                
                Ok(())
            } else {
                Err(miette::miette!("Cursor points to invalid entity."))
            }
        } else {
            Err(miette::miette!("Cursor is not valid."))
        }
    }

    pub fn add_component(&mut self, comp: SerializedComponent) -> Result<()> {
        let name = Self::get_comp_name(&comp);
        if let Some(cursor_id) = self.cursor_id {
            if let Some(node) = self.entities.get_mut(&cursor_id) {
                if node.entity.components.iter().any(|c| Self::get_comp_name(c) == name) {
                    return Err(miette::miette!("[ERROR] Component '{}' already exists on this entity.", name));
                }
                node.entity.components.push(comp);
                println!("[SUCCESS] Attached component '{}' to {}.", name, self.cursor_path);
                Ok(())
            } else {
                Err(miette::miette!("Cursor points to invalid entity."))
            }
        } else {
            Err(miette::miette!("Cursor is not valid."))
        }
    }
    
    pub fn delete_component(&mut self, name: &str) -> Result<()> {
        if let Some(cursor_id) = self.cursor_id {
            if let Some(node) = self.entities.get_mut(&cursor_id) {
                let initial_len = node.entity.components.len();
                node.entity.components.retain(|c| Self::get_comp_name(c) != name);
                if node.entity.components.len() == initial_len {
                    return Err(miette::miette!("[ERROR] Component '{}' not found on entity {}.", name, self.cursor_path));
                }
                println!("[SUCCESS] Removed component '{}' from {}.", name, self.cursor_path);
                Ok(())
            } else {
                Err(miette::miette!("Cursor points to invalid entity."))
            }
        } else {
            Err(miette::miette!("Cursor is not valid."))
        }
    }
    
    pub fn get_entity_path(&self, id: u64) -> String {
        if Some(id) == self.root_id {
            return "/".to_string();
        }
        
        let mut path_parts = Vec::new();
        let mut current_id = Some(id);
        
        while let Some(curr) = current_id {
            if Some(curr) == self.root_id {
                break;
            }
            if let Some(node) = self.entities.get(&curr) {
                path_parts.push(node.entity.name.clone());
                current_id = node.entity.parent_ffi_id;
            } else {
                break;
            }
        }
        
        path_parts.reverse();
        format!("/{}", path_parts.join("/"))
    }
    
    pub fn resolve_path(&self, path: &str) -> Option<u64> {
        if path == "/" {
            return self.root_id;
        }
        
        let mut current_id = if path.starts_with('/') {
            self.root_id
        } else {
            self.cursor_id
        };
        
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        
        for part in parts {
            if part == "." {
                continue;
            }
            if part == ".." {
                if let Some(curr) = current_id {
                    if let Some(node) = self.entities.get(&curr) {
                        if let Some(parent) = node.entity.parent_ffi_id {
                            current_id = Some(parent);
                        }
                    }
                }
                continue;
            }
            
            if let Some(curr) = current_id {
                let mut found = false;
                if let Some(node) = self.entities.get(&curr) {
                    for child_id in &node.children {
                        if let Some(child_node) = self.entities.get(child_id) {
                            if child_node.entity.name == part {
                                current_id = Some(*child_id);
                                found = true;
                                break;
                            }
                        }
                    }
                }
                if !found {
                    return None;
                }
            } else {
                return None;
            }
        }
        
        current_id
    }
    
    pub fn set_cursor(&mut self, path: &str) -> bool {
        if let Some(id) = self.resolve_path(path) {
            self.cursor_id = Some(id);
            self.cursor_path = self.get_entity_path(id);
            true
        } else {
            false
        }
    }
}
