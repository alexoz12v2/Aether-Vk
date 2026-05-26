use alloc::sync::Arc;
#[test]
fn test_create_scene() {
    let ctx = crate::simulation_api::SimulationContext::new_idle().unwrap();
    match ctx.create_default_scene() {
        Ok(id) => println!("Success: {}", id),
        Err(e) => panic!("Error: {:?}", e),
    }
}
