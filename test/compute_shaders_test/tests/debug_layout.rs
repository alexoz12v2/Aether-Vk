use aethervk_core_rlib::gpu::compute_push_constants::*;

#[test]
fn print_layouts() {
  println!(
    "Size of BvhNodeAABBGpu: {}",
    std::mem::size_of::<BvhNodeAABBGpu>()
  );
  println!("Size of EntityGpu: {}", std::mem::size_of::<EntityGpu>());
  // Cannot use memoffset if not in scope, just let it fail or use standard pointers
  let d = std::mem::MaybeUninit::<EntityGpu>::uninit();
  let base = d.as_ptr() as usize;
  let root_index_offset = unsafe { (&(*d.as_ptr()).root_index as *const _ as usize) - base };
  let angular_offset = unsafe { (&(*d.as_ptr()).angular_velocity as *const _ as usize) - base };

  println!("Offset of root_index: {}", root_index_offset);
  println!("Offset of angular_velocity: {}", angular_offset);
}
