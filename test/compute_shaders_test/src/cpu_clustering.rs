use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub struct CollisionEvent {
  pub time_of_impact: f32, // Time of Impact (t_c) from 0.0 to 1.0
  pub entity_a: u32,
  pub entity_b: u32,
}

pub fn parse_gpu_packed_pairs(gpu_data: &[u32], count: usize) -> Vec<CollisionEvent> {
  let mut events = Vec::with_capacity(count);
  for i in 0..count {
    let base = 4 + i * 3;
    events.push(CollisionEvent {
      entity_a: gpu_data[base],
      entity_b: gpu_data[base + 1],
      time_of_impact: f32::from_bits(gpu_data[base + 2]),
    });
  }
  events
}

pub type CollisionCluster = Vec<CollisionEvent>;

pub fn group_and_cluster_collisions(
  mut collisions: Vec<CollisionEvent>,
  time_tolerance: f32,
) -> Vec<CollisionCluster> {
  collisions.sort_by(|a, b| {
    a.time_of_impact.partial_cmp(&b.time_of_impact).unwrap_or(std::cmp::Ordering::Equal)
  });

  let mut collided_entities: HashSet<u32> = HashSet::new();
  let mut resolved_clusters: Vec<CollisionCluster> = Vec::new();

  let mut current_group: Vec<CollisionEvent> = Vec::new();
  let mut current_time = -1.0;

  for col in collisions {
    if collided_entities.contains(&col.entity_a) || collided_entities.contains(&col.entity_b) {
      continue;
    }

    if current_group.is_empty() {
      current_group.push(col.clone());
      current_time = col.time_of_impact;
      continue;
    }

    if (col.time_of_impact - current_time).abs() <= time_tolerance {
      current_group.push(col);
    } else {
      let mut clusters = form_clusters(&current_group);
      resolved_clusters.append(&mut clusters);

      for c in &current_group {
        collided_entities.insert(c.entity_a);
        collided_entities.insert(c.entity_b);
      }

      current_group.clear();

      if collided_entities.contains(&col.entity_a) || collided_entities.contains(&col.entity_b) {
        continue;
      }

      current_time = col.time_of_impact;
      current_group.push(col);
    }
  }

  if !current_group.is_empty() {
    let mut clusters = form_clusters(&current_group);
    resolved_clusters.append(&mut clusters);
  }

  resolved_clusters
}

fn form_clusters(group: &[CollisionEvent]) -> Vec<CollisionCluster> {
  let mut adj_list: HashMap<u32, Vec<usize>> = HashMap::new();
  for (i, col) in group.iter().enumerate() {
    adj_list.entry(col.entity_a).or_default().push(i);
    adj_list.entry(col.entity_b).or_default().push(i);
  }

  let mut visited_collisions = vec![false; group.len()];
  let mut clusters = Vec::new();

  for i in 0..group.len() {
    if !visited_collisions[i] {
      let mut cluster = Vec::new();
      let mut stack = vec![i];
      visited_collisions[i] = true;

      while let Some(idx) = stack.pop() {
        let col = &group[idx];
        cluster.push(col.clone());

        if let Some(neighbors) = adj_list.get(&col.entity_a) {
          for &n_idx in neighbors {
            if !visited_collisions[n_idx] {
              visited_collisions[n_idx] = true;
              stack.push(n_idx);
            }
          }
        }

        if let Some(neighbors) = adj_list.get(&col.entity_b) {
          for &n_idx in neighbors {
            if !visited_collisions[n_idx] {
              visited_collisions[n_idx] = true;
              stack.push(n_idx);
            }
          }
        }
      }
      clusters.push(cluster);
    }
  }

  clusters
}
