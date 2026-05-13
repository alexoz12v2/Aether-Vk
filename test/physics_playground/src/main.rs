use aethervk_core_rlib::scene::{Scene, TransformComponent, EntityId, ReferenceFrameType};
use aethervk_core_rlib::gpu::DynamicBody;
use aethervk_core_rlib::physics::physics_scene::PhysicsScene;
use aethervk_core_rlib::physics::cpu_kernels::{CpuScalarKernels, CpuSimdKernels};
use aethervk_core_rlib::gpu_backends::simulation_step;
use aethervk_core_rlib::scene::particles::{ParticleSystemComponent, ParticleData};
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::os::pool::ThreadPool;
use aethervk_oshal_rlib::math::vector::{Vector3, Vector};
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use std::sync::Arc;
// For LogicThread test
use aethervk_core_rlib::simulation_api::structs::{LogicCommand, LogicThreadContext, LogicState, SimulationTaskManager, SimulationSceneData, TimeScale};
use aethervk_core_rlib::simulation_api::logic_thread::start_logic_thread;
use spin::RwLock;

fn setup_test_scene(scene: &mut Scene) -> (EntityId, EntityId, EntityId) {
    // 1. Root Macro Frame
    let macro_frame = scene.spawn_reference_frame(
        "MacroFrame",
        None,
        TransformComponent {
            position: Vec3f32::from_components(0.0, 0.0, 0.0),
            rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
            scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
        ReferenceFrameType::Macro,
        1.0,
        10000.0,
    );

    // 2. Micro Frame (e.g., a local area where physics happens)
    let micro_frame = scene.spawn_reference_frame(
        "MicroFrame",
        Some(macro_frame),
        TransformComponent {
            position: Vec3f32::from_components(10.0, 0.0, 0.0),
            rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
            scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
        ReferenceFrameType::Micro,
        0.1,
        100.0,
    );

    let particle_system = scene.spawn_entity("ParticleSystem");
    scene.set_parent(particle_system, Some(micro_frame));
    scene.add_component(particle_system, TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
    }).unwrap();
    
    let mut p1 = ParticleData {
        id_low: 0,
        id_high: 0,
        age_low: 0,
        age_high: 0,
        position: [0.0, 10.0, 0.0],
        mass: 5.0,
        velocity: [2.0, -1.0, 0.0],
        active: 1,
    };

    let mut p2 = ParticleData {
        id_low: 1,
        id_high: 0,
        age_low: 0,
        age_high: 0,
        position: [5.0, 10.0, 0.0],
        mass: 1.0,
        velocity: [-2.0, 0.0, 0.0],
        active: 1,
    };

    let mut sys = ParticleSystemComponent {
        particles: std::sync::Arc::new(spin::RwLock::new(vec![p1, p2])),
        bvh: None,
        accumulator: 0,
        next_id: 2,
    };
    sys.update_bvh(1.0);

    scene.add_component(particle_system, sys).unwrap();

    (macro_frame, micro_frame, particle_system)
}

fn run_simulation<K>(kernels: &K, thread_pool: Arc<ThreadPool>, duration_secs: f32) -> Vec<ParticleData>
where
    K: aethervk_core_rlib::gpu::Kernels,
{
    let mut scene = Scene::new(std::sync::Arc::new(aethervk_core_rlib::gpu::RwLock::new(aethervk_core_rlib::simulation::texture_cache::TextureCache::new("AetherVk"))));
    scene.register_all_crate_components();
    let (_macro_frame, _micro_frame, particle_system) = setup_test_scene(&mut scene);

    let mut ps = PhysicsScene::build_from_scene(&scene);

    let dt_us = 16_667; // 60 FPS
    let iterations = (duration_secs * 60.0) as usize;

    for _ in 0..iterations {
        // Manually trigger the IMEX loop steps
        let _ = simulation_step(kernels, &mut ps, &scene, 0, dt_us);

        // Handoffs
        aethervk_core_rlib::physics::handoff::SpheresOfInfluenceSystem::process_handoffs_par(
            &scene,
            &thread_pool,
        );
    }

    let sys = scene.with_component(particle_system, |sys: &ParticleSystemComponent| {
        sys.particles.read().clone()
    }).unwrap();
    sys
}

#[test]
fn test_kernels_5s() {
    let pool = Arc::new(ThreadPool::new(4).unwrap());
    
    let scalar_kernels = CpuScalarKernels {};
    let simd_kernels = CpuSimdKernels { thread_pool: pool.clone() };

    let state_scalar = run_simulation(&scalar_kernels, pool.clone(), 5.0);
    let state_simd = run_simulation(&simd_kernels, pool.clone(), 5.0);

    // Initial position was (0.0, 10.0, 0.0) with velocity (2.0, -1.0, 0.0)
    // After 5s, the particle should have moved
    assert!(state_scalar[0].position[0] > 0.1 || state_scalar[0].position[0] < -0.1);
    
    assert_eq!(state_scalar.len(), state_simd.len());
    for (a, b) in state_scalar.iter().zip(state_simd.iter()) {
        approx::assert_abs_diff_eq!(a.position[0], b.position[0], epsilon = 1e-4);
        approx::assert_abs_diff_eq!(a.position[1], b.position[1], epsilon = 1e-4);
        approx::assert_abs_diff_eq!(a.position[2], b.position[2], epsilon = 1e-4);
        approx::assert_abs_diff_eq!(a.velocity[0], b.velocity[0], epsilon = 1e-4);
    }
}

#[test]
fn test_async_time_manipulation() {
    let pool = Arc::new(ThreadPool::new(4).unwrap());
    let (logic_tx, logic_rx) = thingbuf::mpsc::channel(64);
    let (feedback_tx, _) = thingbuf::mpsc::channel(64);
    let (render_tx, _) = thingbuf::mpsc::channel(64);
    let task_manager = Arc::new(RwLock::new(SimulationTaskManager::new()));
    let logic_state = Arc::new(RwLock::new(LogicState::default()));
    let scenes = Arc::new(RwLock::new(SimulationSceneData::new()));
    
    // Add a dummy scene
    let mut scene = Scene::new(std::sync::Arc::new(aethervk_core_rlib::gpu::RwLock::new(aethervk_core_rlib::simulation::texture_cache::TextureCache::new("AetherVk"))));
    scene.register_all_crate_components();
    let root = scene.spawn_entity("Root");
    let scene_arc = Arc::new(scene);
    let scene_ctx = aethervk_core_rlib::simulation_api::structs::SceneContext::new_empty(scene_arc.clone(), root).with_physics_scene();
    scenes.write().insert(1, Arc::new(RwLock::new(scene_ctx)));

    let context = Arc::new(LogicThreadContext {
        logic_state,
        thread_pool: pool,
        logic_feedback_tx: feedback_tx,
        task_manager,
        scenes: scenes.clone(),
        ctx_ptr: aethervk_core_rlib::simulation_api::structs::SendPtrMut(std::ptr::null_mut()),
        render_tx,
    });

    let _thread_handle = start_logic_thread(logic_rx, context).unwrap();

    // 1. Initially it should be not playing (paused)
    {
        let scenes_read = scenes.read();
        let scene_ctx = scenes_read.get(&1).unwrap().read();
        assert!(!scene_ctx.time_state.read().is_playing);
    }

    // 2. Change time scale and Play the scene
    let _ = logic_tx.try_send(LogicCommand::SetSceneTimeScale { scene_id: 1, scale: TimeScale::OneDay });
    let _ = logic_tx.try_send(LogicCommand::PlayScene { scene_id: 1 });
    std::thread::sleep(std::time::Duration::from_millis(50));
    {
        let scenes_read = scenes.read();
        let scene_ctx = scenes_read.get(&1).unwrap().read();
        assert!(scene_ctx.time_state.read().is_playing);
    }

    // 3. Change time scale
    let _ = logic_tx.try_send(LogicCommand::SetSceneTimeScale { scene_id: 1, scale: TimeScale::OneWeek });
    std::thread::sleep(std::time::Duration::from_millis(50));
    {
        let scenes_read = scenes.read();
        let scene_ctx = scenes_read.get(&1).unwrap().read();
        assert_eq!(scene_ctx.time_state.read().current_scale, TimeScale::OneWeek);
    }

    // 4. Pause the scene
    let _ = logic_tx.try_send(LogicCommand::PauseScene { scene_id: 1 });
    std::thread::sleep(std::time::Duration::from_millis(50));
    {
        let scenes_read = scenes.read();
        let scene_ctx = scenes_read.get(&1).unwrap().read();
        assert!(!scene_ctx.time_state.read().is_playing);
    }
    
    // Shutdown
    let _ = logic_tx.try_send(LogicCommand::Shutdown);
}

fn main() {
    println!("Physics playground running. Execute 'cargo test' to run assertions.");
}