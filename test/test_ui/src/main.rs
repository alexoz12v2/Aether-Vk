use aethervk_core_rlib::gpu::PresentationEngineHandle;
use aethervk_core_rlib::scene::{EntityId, Scene};
use aethervk_core_rlib::scene::ui::{ScreenSpaceTextComponent, Transform2DComponent, UiComponent};
use aethervk_core_rlib::simulation_api::SimulationContext;
use aethervk_core_rlib::types::EngineResult;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3};
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use std::sync::Arc;
use test_utils::sim_app::{SimulationDelegate, run_simulation_app};
use winit::window::Window;

struct TestUiDelegate {
    font_atlas: Option<Arc<aethervk_core_rlib::scene::text::FontAtlas>>,
    font_hash: u64,
    camera_ext_entity: Option<EntityId>,
}

impl TestUiDelegate {
    fn new() -> Self {
        Self {
            font_atlas: None,
            font_hash: 0,
            camera_ext_entity: None,
        }
    }
}

impl SimulationDelegate for TestUiDelegate {
    fn create_scene(&mut self, ctx: &mut SimulationContext) -> EngineResult<u64> {
        let scene_id = ctx.create_empty_scene()?;
        Ok(scene_id)
    }

    fn on_setup(
        &mut self,
        ctx: &mut SimulationContext,
        scene_id: u64,
        pe_handle: PresentationEngineHandle,
        window: &Window,
    ) -> EngineResult<()> {
        let width = window.inner_size().width as f32;
        let height = window.inner_size().height as f32;

        let scene_ctx = ctx.get_scene(scene_id).unwrap();
        let scene_ctx_write = scene_ctx.write();
        let scene = &scene_ctx_write.scene;

        let ext_cam = scene.spawn_entity("camera");
        scene.add_component(ext_cam, aethervk_core_rlib::scene::TransformComponent {
            position: Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
        }).unwrap();
        scene.add_component(ext_cam, aethervk_core_rlib::scene::CameraComponent::new_ortho(0.0, width, 0.0, height, -1.0, 1.0)).unwrap();

        self.camera_ext_entity = Some(ext_cam);

        {
            let mut presentation_engines = scene_ctx_write.presentation_engines.write();
            let pe_data = presentation_engines.get_mut(&pe_handle).unwrap();
            pe_data.camera_entity = Some(ext_cam);
        }

        // Continue with UI setup

        let asset_dir = test_utils::cycle_get_asset_path_from_exe(true);
        let font_path = test_utils::get_monospace_font_path_from_asset_path(&asset_dir);
        let font_atlas = aethervk_core_rlib::scene::text::FontAtlas::from_path(font_path.to_str().unwrap(), 32.0).unwrap();
        let font_hash = font_atlas.hash_metadata();
        let font_arc = Arc::new(font_atlas);
        self.font_atlas = Some(font_arc.clone());
        self.font_hash = font_hash;

        // Background
        let bg_entity = scene.spawn_entity("Background");
        let mut bg_t2d = Transform2DComponent::default();
        bg_t2d.local_position = [0.0, 0.0];
        bg_t2d.size = [1200.0, 800.0];
        bg_t2d.global_bounds = [0.0, 0.0, 1200.0, 800.0];
        bg_t2d.global_clip = [-9999.0, -9999.0, 9999.0, 9999.0];
        bg_t2d.global_depth = 0;
        bg_t2d.local_z_index = -100;
        let mut bg_ui = UiComponent::default();
        bg_ui.color_start = [0.5, 0.8, 1.0, 1.0];
        bg_ui.color_end = [0.1, 0.4, 0.8, 1.0];
        bg_ui.gradient_dir = [0.0, 1.0];
        bg_ui.opacity = 1.0;
        bg_ui.texture_id = 0xFFFFFFFF;
        scene.add_component(bg_entity, bg_t2d).unwrap();
        scene.add_component(bg_entity, bg_ui).unwrap();

        // Main Weather Panel
        let main_panel = scene.spawn_entity("MainPanel");
        let mut main_t2d = Transform2DComponent::default();
        main_t2d.local_position = [100.0, 100.0];
        main_t2d.size = [400.0, 300.0];
        main_t2d.global_bounds = [100.0, 100.0, 400.0, 300.0];
        main_t2d.global_clip = [-9999.0, -9999.0, 9999.0, 9999.0];
        main_t2d.global_depth = 1;
        let mut main_ui = UiComponent::default();
        main_ui.color_start = [0.8, 0.9, 1.0, 0.8];
        main_ui.color_end = [0.5, 0.7, 1.0, 0.8];
        main_ui.border_radius = [20.0, 20.0, 20.0, 20.0];
        main_ui.color_shadow = [0.0, 0.0, 0.0, 0.3];
        main_ui.shadow_params = [0.0, 10.0, 20.0, 5.0];
        main_ui.texture_id = 0xFFFFFFFF;
        main_ui.opacity = 1.0;
        scene.add_component(main_panel, main_t2d).unwrap();
        scene.add_component(main_panel, main_ui).unwrap();

        // Title Text
        let title_text = scene.spawn_entity("TitleText");
        let mut title_t2d = Transform2DComponent::default();
        title_t2d.local_position = [120.0, 130.0]; // Pixel coordinates
        title_t2d.global_bounds = [120.0, 130.0, 0.0, 0.0];
        title_t2d.global_clip = [-9999.0, -9999.0, 9999.0, 9999.0];
        title_t2d.global_depth = 2;
        let title_ui = ScreenSpaceTextComponent {
            text: "San Francisco, CA".to_string(),
            font_atlas: font_arc.clone(),
            font_hash,
            color: [0.1, 0.1, 0.1, 1.0],
            points: 24.0,
        };
        scene.add_component(title_text, title_t2d).unwrap();
        scene.add_component(title_text, title_ui).unwrap();

        // Temperature Text
        let temp_text = scene.spawn_entity("TempText");
        let mut temp_t2d = Transform2DComponent::default();
        temp_t2d.local_position = [120.0, 180.0]; // Pixel coordinates
        temp_t2d.global_bounds = [120.0, 180.0, 0.0, 0.0];
        temp_t2d.global_clip = [-9999.0, -9999.0, 9999.0, 9999.0];
        temp_t2d.global_depth = 2;
        let temp_ui = ScreenSpaceTextComponent {
            text: "21°C".to_string(),
            font_atlas: font_arc.clone(),
            font_hash,
            color: [0.1, 0.1, 0.1, 1.0],
            points: 48.0,
        };
        scene.add_component(temp_text, temp_t2d).unwrap();
        scene.add_component(temp_text, temp_ui).unwrap();

        // 7-Day Forecast Panel
        let forecast_panel = scene.spawn_entity("ForecastPanel");
        let mut forecast_t2d = Transform2DComponent::default();
        forecast_t2d.local_position = [520.0, 100.0];
        forecast_t2d.size = [200.0, 140.0];
        forecast_t2d.global_bounds = [520.0, 100.0, 200.0, 140.0];
        forecast_t2d.global_clip = [-9999.0, -9999.0, 9999.0, 9999.0];
        forecast_t2d.global_depth = 1;
        let mut forecast_ui = UiComponent::default();
        forecast_ui.color_start = [0.9, 0.95, 1.0, 0.8];
        forecast_ui.color_end = [0.9, 0.95, 1.0, 0.8];
        forecast_ui.border_radius = [15.0, 15.0, 15.0, 15.0];
        forecast_ui.color_shadow = [0.0, 0.0, 0.0, 0.15];
        forecast_ui.shadow_params = [0.0, 5.0, 10.0, 0.0];
        forecast_ui.texture_id = 0xFFFFFFFF;
        forecast_ui.opacity = 1.0;
        scene.add_component(forecast_panel, forecast_t2d).unwrap();
        scene.add_component(forecast_panel, forecast_ui).unwrap();

        drop(scene_ctx_write); // drop write lock before calling ctx functions

        let _ = ctx.threads.logic_thread.tx().try_send(
            aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id }
        );

        Ok(())
    }

    fn on_resize(
        &mut self,
        ctx: &mut SimulationContext,
        scene_id: u64,
        width: u32,
        height: u32,
    ) {
        if let Some(scene_ctx) = ctx.get_scene(scene_id) {
            let scene_ctx = scene_ctx.write();
            let scene = &scene_ctx.scene;
            if let Some(bg_entity) = scene.get_entity_by_name("Background") {
                scene.with_component_mut::<Transform2DComponent, _, _>(bg_entity, |t2d| {
                    t2d.size = [width as f32, height as f32];
                    t2d.is_dirty = true;
                });
            }
        }
        }
}

fn main() {
    run_simulation_app("Weather UI Test", TestUiDelegate::new());
}