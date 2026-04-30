#[test]
fn test_alignment() {
    use aethervk_core_rlib::scene::particles::ParticleData;
    let p = ParticleData {
        id_low: 0,
        id_high: 0,
        age_low: 0,
        age_high: 0,
        position: [0.0; 3],
        mass: 0.0,
        velocity: [0.0; 3],
        active: 0,
    };
    let base = &p as *const ParticleData as usize;
    let id_low = &p.id_low as *const _ as usize - base;
    let position = &p.position as *const _ as usize - base;
    let velocity = &p.velocity as *const _ as usize - base;
    let age_low = &p.age_low as *const _ as usize - base;
    let mass = &p.mass as *const _ as usize - base;
    let active = &p.active as *const _ as usize - base;
    println!("size: {}", std::mem::size_of::<ParticleData>());
    println!("id_low: {}", id_low);
    println!("position: {}", position);
    println!("velocity: {}", velocity);
    println!("age_low: {}", age_low);
    println!("mass: {}", mass);
    println!("active: {}", active);
}