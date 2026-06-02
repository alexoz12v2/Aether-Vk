#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraDTO {
  pub is_orthographic: bool, // 1 byte
  // padding 3 bytes
  pub fov: f32, // 4 bytes
  pub aspect: f32, // 4 bytes
  pub near: f32, // 4 bytes
  pub far: f32, // 4 bytes
  pub ortho_scale_factor: f32, // 4 bytes
  pub focus_distance: f32, // 4 bytes
  pub proj: [f32; 16], // 64 bytes
}

fn main() {
    println!("CameraDTO size: {}", std::mem::size_of::<CameraDTO>());
    // Cannot use offset_of easily without nightly or a macro, but let's do this:
    let dto = CameraDTO {
        is_orthographic: false,
        fov: 0.0,
        aspect: 0.0,
        near: 0.0,
        far: 0.0,
        ortho_scale_factor: 0.0,
        focus_distance: 0.0,
        proj: [0.0; 16],
    };
    let base = &dto as *const _ as usize;
    println!("is_orthographic: {}", &dto.is_orthographic as *const _ as usize - base);
    println!("fov: {}", &dto.fov as *const _ as usize - base);
    println!("aspect: {}", &dto.aspect as *const _ as usize - base);
    println!("near: {}", &dto.near as *const _ as usize - base);
    println!("far: {}", &dto.far as *const _ as usize - base);
    println!("ortho_scale_factor: {}", &dto.ortho_scale_factor as *const _ as usize - base);
    println!("focus_distance: {}", &dto.focus_distance as *const _ as usize - base);
    println!("proj: {}", &dto.proj as *const _ as usize - base);
}
