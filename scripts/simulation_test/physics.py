import numpy as np
import scipy.linalg
from .linalg import hat, vee, compute_com_and_tensor, qr_diagonalization


G = 1.0  # Scaled for simulation. Original is 6.674e-11.

def get_force_and_torque(body, sun_pos, sun_mass):
    force_world = gravitational_force(body.position, sun_pos, body.mass, sun_mass)
    torque_body = body.orientation.T @ np.zeros(3)
    return force_world, torque_body


def gravitational_force(pos, sun_pos, m, sun_m):
    r_vec = sun_pos - pos
    dist = np.linalg.norm(r_vec)
    if dist < 1.0:
        return np.zeros(3)
    return (G * m * sun_m / dist**3) * r_vec


def calculate_total_energy(body, sun_pos, sun_mass):
    r_dist = np.linalg.norm(sun_pos - body.position)
    potential_energy = -G * body.mass * sun_mass / r_dist if r_dist > 0 else 0
    trans_ke = 0.5 * body.mass * np.dot(body.velocity, body.velocity)
    rot_ke = 0.5 * np.dot(body.angular_velocity_body, body.Pi)
    return potential_energy + trans_ke + rot_ke


def particle_implicit_midpoint_step(p, dt, sun_pos, sun_mass, comet):
    x_n = p.position.copy()
    v_n = p.velocity.copy()
    
    # Initial guess (Explicit Euler)
    force_n = (1 - p.beta) * gravitational_force(x_n, sun_pos, p.mass, sun_mass) + \
              gravitational_force(x_n, comet.position, p.mass, comet.mass)
    
    v_guess = v_n + dt * (force_n / p.mass)
    x_guess = x_n + dt * v_n
    
    # Fixed-point iteration (3-5 iterations is usually enough for particles)
    for _ in range(3):
        x_mid = 0.5 * (x_n + x_guess)
        v_mid = 0.5 * (v_n + v_guess)
        
        force_mid = (1 - p.beta) * gravitational_force(x_mid, sun_pos, p.mass, sun_mass) + \
                    gravitational_force(x_mid, comet.position, p.mass, comet.mass)
        
        v_guess = v_n + dt * (force_mid / p.mass)
        x_guess = x_n + dt * v_mid

    p.position = x_guess
    p.velocity = v_guess


def implicit_midpoint_step(body, h, sun_pos, sun_mass):
    x_n, p_n, R_n, Pi_n = body.position, body.linear_momentum, body.orientation, body.Pi
    I_inv, m_inv = body.inertia_tensor_body_inv, 1.0 / body.mass
    F_n, tau_n = get_force_and_torque(body, sun_pos, sun_mass)
    x_guess, p_guess = x_n + h * (p_n * m_inv), p_n + h * F_n
    omega_n = I_inv @ Pi_n
    R_guess = R_n @ scipy.linalg.expm(h * hat(omega_n))
    Pi_guess = Pi_n + h * (np.cross(Pi_n, omega_n) + tau_n)
    for _ in range(10):
        x_mid, p_mid, Pi_mid = (
            0.5 * (x_n + x_guess),
            0.5 * (p_n + p_guess),
            0.5 * (Pi_n + Pi_guess),
        )
        omega_mid = I_inv @ Pi_mid
        R_mid = R_n @ scipy.linalg.expm(0.5 * h * hat(omega_mid))
        F_mid, tau_mid = get_force_and_torque(
            type(
                "obj",
                (object,),
                {"position": x_mid, "orientation": R_mid, "mass": body.mass},
            ),
            sun_pos,
            sun_mass,
        )
        x_res, p_res = x_guess - (x_n + h * p_mid * m_inv), p_guess - (p_n + h * F_mid)
        Pi_res = Pi_guess - (Pi_n + h * (np.cross(Pi_mid, omega_mid) + tau_mid))
        R_err_matrix = (R_n @ scipy.linalg.expm(h * hat(omega_mid))) @ R_guess.T
        r_res = vee(R_err_matrix - R_err_matrix.T)
        residual = np.concatenate([x_res, p_res, Pi_res, r_res])
        if np.linalg.norm(residual) < 1e-9:
            break
        J = np.zeros((12, 12))
        J[0:3, 0:3], J[0:3, 3:6] = np.eye(3), -0.5 * h * m_inv * np.eye(3)
        J[3:6, 3:6] = np.eye(3)
        J[6:9, 6:9] = (
            np.eye(3) - 0.5 * h * (hat(omega_mid) - hat(I_inv @ Pi_mid)) @ I_inv
        )
        J[9:12, 9:12] = np.eye(3)
        try:
            correction = np.linalg.solve(J, -residual)
        except np.linalg.LinAlgError:
            break
        dx, dp, dPi, d_theta = (
            correction[0:3],
            correction[3:6],
            correction[6:9],
            correction[9:12],
        )
        x_guess, p_guess, Pi_guess = x_guess + dx, p_guess + dp, Pi_guess + dPi
        R_guess = scipy.linalg.expm(hat(d_theta)) @ R_guess
    body.position, body.linear_momentum, body.orientation, body.Pi = (
        x_guess,
        p_guess,
        R_guess,
        Pi_guess,
    )


class RigidBody:
    def __init__(
        self,
        initial_pos_com,
        initial_vel,
        initial_ang_vel_raw,
        raw_vertices,
        mass_per_vertex,
    ):
        self.mass = mass_per_vertex * len(raw_vertices)
        self.com_offset_body, raw_I = compute_com_and_tensor(
            raw_vertices, mass_per_vertex
        )
        principal_moments, self.principal_axes_R = qr_diagonalization(raw_I)
        self.inertia_tensor_body = np.diag(principal_moments)
        self.inertia_tensor_body_inv = np.diag(1.0 / principal_moments)
        vertices_com_raw = raw_vertices - self.com_offset_body
        self.vertices_com_frame = (self.principal_axes_R.T @ vertices_com_raw.T).T
        self.position = np.array(initial_pos_com, dtype=float)
        self.linear_momentum = self.mass * np.array(initial_vel)
        self.orientation = self.principal_axes_R
        ang_vel_principal = self.principal_axes_R.T @ np.array(initial_ang_vel_raw)
        self.Pi = self.inertia_tensor_body @ ang_vel_principal

    @property
    def velocity(self):
        return self.linear_momentum / self.mass

    @property
    def angular_velocity_body(self):
        return self.inertia_tensor_body_inv @ self.Pi

    @property
    def body_origin_world(self):
        return self.position - self.orientation @ (
            self.principal_axes_R.T @ self.com_offset_body
        )

    def get_world_vertices(self):
        return (self.orientation @ self.vertices_com_frame.T).T + self.position

    def get_comet_triangles(self):
        # This assumes a convex shape and simple fan triangulation from the first vertex
        verts = self.get_world_vertices()
        if len(verts) < 3:
            return []
        return [
            (verts[0], verts[i], verts[i + 1]) for i in range(1, len(verts) - 1)
        ]


ANELASTIC_LOOP_COUNT_THRESHOLD = 10
PARTICLE_NUCLEUS_RESTITUTION = 0.8


def sphere_triangle_intersection(p, p_next, r, tri):
    v0, v1, v2 = tri
    # TODO: This is a placeholder. A real implementation is needed.
    # This is a complex geometric problem involving moving sphere vs moving triangle.
    # For now, we simplify by checking intersection at the end of the step.
    # A proper implementation would solve for the time of impact `t` in [0, 1].
    
    # Simple check: does the particle's next position intersect the triangle's plane?
    normal = np.cross(v1 - v0, v2 - v0)
    normal /= np.linalg.norm(normal)
    
    dist = np.dot(p_next - v0, normal)
    
    if abs(dist) < r:
        # Now check if the projection is inside the triangle (Barycentric coordinates)
        # This is a simplification and not a full continuous check.
        # A full implementation is beyond this scope.
        return 0.99, normal # Assume collision happens at the end of the step
    return None, None


def continuous_collision_detection(particle, comet, dt):
    p_initial = particle.position
    # Simplified: assume constant velocity over the timestep for the particle
    p_next = particle.position + particle.velocity * dt

    earliest_toi = float("inf")
    collision_normal = None

    for triangle in comet.get_comet_triangles():
        toi, normal = sphere_triangle_intersection(
            p_initial, p_next, particle.radius, triangle
        )
        if toi is not None and toi < earliest_toi:
            earliest_toi = toi
            collision_normal = normal

    if earliest_toi <= 1.0: # toi is a fraction of the timestep
        return earliest_toi, collision_normal
    return None, None

def rewind_and_resolve_collision(particle, comet, toi, normal, dt):
    # Rewind particle to time of impact
    particle.position -= particle.velocity * (1.0 - toi) * dt
    
    # Reflect velocity
    v_n = np.dot(particle.velocity, normal) * normal
    v_t = particle.velocity - v_n
    particle.velocity = v_t - PARTICLE_NUCLEUS_RESTITUTION * v_n
    
    # Mark as collided for this step
    particle.collided = True



class Particle:
    def __init__(self, position, velocity, mass, radius, color, beta, lifetime):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.radius = radius
        self.color = color
        self.beta = beta
        self.lifetime = lifetime
        self.collided = False

    def update(self, dt):
        self.lifetime -= dt

    def is_alive(self):
        return self.lifetime > 0


def emit_particle(
    comet,
    particle_mass,
    particle_radius,
    particle_color,
    particle_beta,
    particle_lifetime,
    particle_velocity_scale,
    particle_cone_aperture,
):
    # Emit from the first vertex
    emission_pos = comet.get_world_vertices()[0]

    # Start with the velocity of the emission point
    base_velocity = comet.velocity + np.cross(
        comet.angular_velocity_body, emission_pos - comet.position
    )

    # Random direction within a cone
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, particle_cone_aperture)

    # Spherical to Cartesian conversion for the random direction
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    random_dir = np.array([x, y, z])

    # Ensure the random direction is oriented away from the comet
    if np.dot(random_dir, emission_pos - comet.position) < 0:
        random_dir = -random_dir

    # Combine and scale velocity
    initial_velocity = (
        base_velocity
        + random_dir * particle_velocity_scale * np.random.uniform(0.8, 1.2)
    )

    return Particle(
        position=emission_pos,
        velocity=initial_velocity,
        mass=particle_mass,
        radius=particle_radius,
        color=particle_color,
        beta=particle_beta,
        lifetime=particle_lifetime,
    )


def update_particles(particles, comet, sun_pos, sun_mass, dt):
    collisions = []
    for p in particles:
        if p.collided:
            p.collided = False  # Reset for next frame
            continue

        # Store initial state for rewinding
        initial_position = p.position.copy()
        initial_velocity = p.velocity.copy()

        # Force from sun (gravity + radiation pressure)
        force_sun = (1 - p.beta) * gravitational_force(
            p.position, sun_pos, p.mass, sun_mass
        )

        # Force from comet
        force_comet = gravitational_force(
            p.position, comet.position, p.mass, comet.mass
        )

        total_force = force_sun + force_comet

        # IMR (without rotational mechanics)
        particle_implicit_midpoint_step(p, dt, sun_pos, sun_mass, comet)

        # --- Collision Detection ---
        toi, normal = continuous_collision_detection(p, comet, dt)

        if toi is not None:
            # Found a collision, revert and add to list
            p.position = initial_position
            p.velocity = initial_velocity
            collisions.append((p, toi, normal))
        else:
            # No collision, finalize update
            p.update(dt)

    return collisions
