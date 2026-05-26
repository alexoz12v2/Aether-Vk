use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::Vector;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use aethervk_oshal_rlib::math::matrix::Matrix4;

fn main() {
    let pos = 0.02 / f32::sqrt(3.0);
    let look_dir = Vec3f32::from_components(-pos, -pos, -pos).normalize();
    let up = Vec3f32::from_components(0.0, 0.0, 1.0);
    let right = look_dir.cross(up).normalize();
    let true_up = right.cross(look_dir).normalize();
    let mat = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::look_at_axes(right, look_dir, true_up, Vec3f32::from_components(pos, pos, pos));
    let rot = <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::to_quat_custom_frame(&mat);
    println!("rot: W={} X={} Y={} Z={}", rot.0.w(), rot.0.x(), rot.0.y(), rot.0.z());
}
