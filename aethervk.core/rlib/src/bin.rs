use aethervk_core_rlib::simulation_api::SimulationContext;
use aethervk_core_rlib::simulation_api::structs::{RenderThreadParams, LogicThreadParams};

fn main() {
    let render_params = RenderThreadParams { dummy: 0 };
    let logic_params = LogicThreadParams { dummy: 0 };
    let ctx = SimulationContext::new_running(render_params, logic_params).unwrap();
    match ctx.create_default_scene() {
        Ok(id) => println!("Success: {}", id),
        Err(e) => println!("Error: {:?}", e),
    }
}
