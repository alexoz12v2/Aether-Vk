import numpy as np
import pygame
from .physics import (
    PARTICLE_NUCLEUS_RESTITUTION,
    RigidBody,
    continuous_collision_detection,
    implicit_midpoint_step,
    calculate_total_energy,
    G,
    Particle,
    update_particles,
    emit_particle,
    rewind_and_resolve_collision,
    ANELASTIC_LOOP_COUNT_THRESHOLD,
    particle_implicit_midpoint_step,
)
from .ui import DebugGUI, Slider, Button

# --- Constants ---
SIM_WIDTH, SIM_HEIGHT = 800, 600
GUI_WIDTH = 400
WIDTH, HEIGHT = SIM_WIDTH + GUI_WIDTH, SIM_HEIGHT
WHITE, BLACK, RED, BLUE, GUI_BG, SEED_VERTEX_COLOR = (
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 0, 255),
    (240, 240, 240),
    (255, 165, 0),
)
FPS, DT = 60, 1 / 60

# --- Simulation Parameters ---
SUN_MASS = 10000.0
COMET_MASS_PER_VERTEX = 1.0
CAMERA_ZOOM = 0.5
CAMERA_POS = np.array([SIM_WIDTH / 2, SIM_HEIGHT / 2])

# --- Particle Parameters ---
PARTICLE_LIFETIME = 50.0  # seconds
PARTICLE_MASS = 0.01
PARTICLE_RADIUS = 1
PARTICLE_BETA = 2  # For radiation pressure
PARTICLE_EMISSION_RATE = 10  # particles per second
PARTICLE_VELOCITY_SCALE = 1.0
PARTICLE_CONE_APERTURE = np.pi / 8  # radians


# --- Main Loop ---
def main():
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("monospace", 12)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Implicit Midpoint Rigid Body Simulation on SO(3)")
    clock = pygame.time.Clock()

    # --- Comet Configuration ---
    initial_pos_com = [200.0, 0.0, 0.0]
    initial_ang_vel_raw = [0.0, 0.0, 0.5]
    # Stable elliptical orbit (Total Energy < 0)
    initial_vel = [0.0, 5.0, 0.0]
    # Escape trajectory (Total Energy >= 0)
    # initial_vel = [0.0, 10.0, 0.0]
    # Near-circular orbit (v approx sqrt(GM/r))
    # v_circ = np.sqrt(G * SUN_MASS / np.linalg.norm(initial_pos_com)) # Approx 7.07
    # initial_vel = [0.0, v_circ, 0.0]

    comet = RigidBody(
        initial_pos_com=initial_pos_com,
        initial_vel=initial_vel,
        initial_ang_vel_raw=initial_ang_vel_raw,
        raw_vertices=[
            [-20, -15, 0],
            [0, -15, 0],
            [20, -10, 0],
            [20, 10, 0],
            [0, 15, 0],
            [-20, 10, 0],
            [-10, 0, 0],
        ],
        mass_per_vertex=COMET_MASS_PER_VERTEX,
    )

    print("--- Body Frame & Inertia Analysis ---")
    print(f"CoM Offset from Body Origin: {comet.com_offset_body}")
    print(f"Principal Moments (I_body):\n{np.diagonal(comet.inertia_tensor_body)}")
    print(f"Principal Axes (R_to_principal):\n{comet.principal_axes_R}")
    print("------------------------------------")

    sun_pos = np.zeros(3)
    gui = DebugGUI(screen, font, SIM_WIDTH)
    time_slider = Slider(SIM_WIDTH + 20, HEIGHT - 40, 150, 20, 1, 20, 1, label="Speed")
    particle_radius_slider = Slider(
        SIM_WIDTH + 20, HEIGHT - 160, 150, 20, 1, 10, PARTICLE_RADIUS, label="Radius"
    )

    initial_energy = calculate_total_energy(comet, sun_pos, SUN_MASS)
    print("--- Orbit Prediction ---")
    print(f"Initial Total Energy: {initial_energy:.2f}")
    if initial_energy < 0:
        print("Prediction: BOUND ORBIT (Elliptical). Total energy is negative.")
    else:
        print(
            "Prediction: ESCAPE TRAJECTORY (Parabolic/Hyperbolic). Total energy is non-negative."
        )
    print("------------------------")

    particles = []
    particle_generation_on = True
    show_particles = True

    PARTICLE_COLORS = [(200, 200, 0), (0, 200, 200), (200, 0, 200)]
    current_color_index = 0

    def toggle_particle_generation():
        nonlocal particle_generation_on
        particle_generation_on = not particle_generation_on

    def kill_all_particles():
        nonlocal particles
        particles = []

    def cycle_particle_color():
        nonlocal current_color_index
        current_color_index = (current_color_index + 1) % len(PARTICLE_COLORS)

    buttons = [
        Button(
            SIM_WIDTH + 20,
            HEIGHT - 80,
            150,
            30,
            "Toggle Particles",
            font,
            toggle_particle_generation,
        ),
        Button(
            SIM_WIDTH + 20,
            HEIGHT - 120,
            150,
            30,
            "Kill All Particles",
            font,
            kill_all_particles,
        ),
        Button(
            SIM_WIDTH + 20,
            HEIGHT - 200,
            150,
            30,
            "Cycle Color",
            font,
            cycle_particle_color,
        ),
    ]

    emission_timer = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            time_slider.handle_event(event)
            particle_radius_slider.handle_event(event)
            for button in buttons:
                button.handle_event(event)

        # --- Particle Management ---
        newly_generated_particles = []
        if particle_generation_on:
            emission_timer += DT
            while emission_timer > 1.0 / PARTICLE_EMISSION_RATE:
                p = emit_particle(
                    comet,
                    PARTICLE_MASS,
                    particle_radius_slider.get_value(),
                    PARTICLE_COLORS[current_color_index],
                    PARTICLE_BETA,
                    PARTICLE_LIFETIME,
                    PARTICLE_VELOCITY_SCALE,
                    PARTICLE_CONE_APERTURE,
                )
                particles.append(p)
                newly_generated_particles.append(p)
                emission_timer -= 1.0 / PARTICLE_EMISSION_RATE

        dead_particles_this_step = {p for p in particles if not p.is_alive()}

        # --- Revised Main Loop Snippet ---
        for _ in range(time_slider.get_value()):
            # 1. Step the heavy rigid body FIRST (no rewinding)
            implicit_midpoint_step(comet, DT, sun_pos, SUN_MASS)
            
            # 2. Update particles independently
            for p in particles:
                if not p.is_alive():
                    continue
                    
                initial_pos = p.position.copy()
                initial_vel = p.velocity.copy()
                
                # Step particle (using your new IMR or Velocity Verlet)
                particle_implicit_midpoint_step(p, DT, sun_pos, SUN_MASS, comet)
                
                # 3. CCD against the already-moved comet
                toi, normal = continuous_collision_detection(p, comet, DT)
                
                if toi is not None:
                    # Revert ONLY this particle to TOI
                    sub_dt = DT * toi
                    p.position = initial_pos
                    p.velocity = initial_vel
                    particle_implicit_midpoint_step(p, sub_dt, sun_pos, SUN_MASS, comet)
                    
                    # Resolve collision using RELATIVE velocity
                    r_contact = p.position - comet.position
                    v_point_on_comet = comet.velocity + np.cross(comet.angular_velocity_body, r_contact)
                    
                    v_rel = p.velocity - v_point_on_comet
                    v_rel_n = np.dot(v_rel, normal) * normal
                    v_rel_t = v_rel - v_rel_n
                    
                    # Apply restitution to relative velocity
                    v_rel_new = v_rel_t - PARTICLE_NUCLEUS_RESTITUTION * v_rel_n
                    
                    # Convert back to absolute velocity
                    p.velocity = v_rel_new + v_point_on_comet
                    
                    # Step the remaining time
                    remaining_dt = DT * (1.0 - toi)
                    particle_implicit_midpoint_step(p, remaining_dt, sun_pos, SUN_MASS, comet)
                    
                p.update(DT)

        particles = [p for p in particles if p.is_alive()]

        # --- Drawing ---
        screen.fill(WHITE)
        gui.y = 10
        screen.fill(GUI_BG, (SIM_WIDTH, 0, GUI_WIDTH, HEIGHT))

        gui.draw(
            "Constants",
            {"Total Mass": comet.mass, "CoM Offset (Body)": comet.com_offset_body},
            BLACK,
        )
        gui.draw(
            "State",
            {
                "CoM Position": comet.position,
                "Body Origin": comet.body_origin_world,
                "Lin Momentum": comet.linear_momentum,
                "Ang Momentum (Body)": comet.Pi,
                "Ang Velocity (Body)": comet.angular_velocity_body,
            },
            BLUE,
        )
        current_energy = calculate_total_energy(comet, sun_pos, SUN_MASS)
        gui.draw(
            "Error", {"Energy Drift": f"{abs(current_energy - initial_energy):e}"}, RED
        )
        gui.draw("Particles", {"Count": len(particles)}, BLACK)

        time_slider.draw(screen, font)
        particle_radius_slider.draw(screen, font)
        for button in buttons:
            button.draw(screen)

        def world_to_screen(pos):
            p = (pos[:2] - CAMERA_POS[:2]) * CAMERA_ZOOM + np.array(
                [SIM_WIDTH / 2, SIM_HEIGHT / 2]
            )
            return p.astype(int)

        # Draw particles
        if show_particles:
            for p in particles:
                pygame.draw.circle(
                    screen, p.color, world_to_screen(p.position), int(p.radius * CAMERA_ZOOM)
                )

        pygame.draw.circle(screen, BLACK, world_to_screen(sun_pos), 10)
        world_vertices = comet.get_world_vertices()
        pygame.draw.polygon(
            screen, BLACK, [world_to_screen(v) for v in world_vertices], 1
        )
        pygame.draw.circle(
            screen, SEED_VERTEX_COLOR, world_to_screen(world_vertices[0]), 4
        )
        pygame.draw.circle(screen, RED, world_to_screen(comet.position), 4)
        pygame.draw.circle(screen, BLUE, world_to_screen(comet.body_origin_world), 4)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    pygame.font.quit()


if __name__ == "__main__":
    main()

