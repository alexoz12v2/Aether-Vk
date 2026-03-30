
import numpy as np
import pygame

# --- Constants ---
SIM_WIDTH, SIM_HEIGHT = 800, 600
GUI_WIDTH = 350
WIDTH, HEIGHT = SIM_WIDTH + GUI_WIDTH, SIM_HEIGHT

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
GUI_BG = (240, 240, 240)

FPS = 60
DT = 1 / FPS

# --- Simulation Parameters ---
G = 6.67430e-3  # Adjusted for simulation scale
SUN_MASS = 10000.0
COMET_MASS_PER_VERTEX = 1.0
CAMERA_ZOOM = 0.5
CAMERA_POS = np.array([SIM_WIDTH/2, SIM_HEIGHT/2])

# --- Utility Functions for Frame of Reference and Inertia Tensor Computation ---
def compute_com_and_tensor(verts, m=1.0):
    """
    Computes the center of mass and the Body-Frame Inertia Tensor
    for a discrete set of point masses
    """
    # 1. Center of mass
    # Average of all vertices (they all have the same mass)
    com = np.mean(verts, axis=0)

    # 2. Shift vertices to the CoM frame
    verts_centered = verts - com 

    # 3. Compute the Inertia Tensor
    I = np.zeros((3,3))
    for v in verts_centered:
        x, y, z = v
        # Diagonal elements (Moments of Inertia)
        I[0, 0] += m * (y**2 + z**2)
        I[1, 1] += m * (x**2 + z**2)
        I[2, 2] += m * (x**2 + y**2)

        # Off-Diagonal elements (Products of Inertia)
        I[0, 1] -= m * (x * y)
        I[1, 0] -= m * (x * y)

        # these should be zero for 2.5D
        I[0, 2] -= m * (x * z)
        I[2, 0] -= m * (x * z)
        I[1, 2] -= m * (y * z)
        I[2, 1] -= m * (y * z)

    return com, I


def qr_diagonalization(A, tol=1e-10, max_iter=1000):
    """
    Computes the eigenvalues and eigenvectors of a symmetric matrix
    using a basic implementation of the QR algorithm
    """
    A_k = np.copy(A)
    # Initialize the eigenvector matrix as the identity matrix
    Q_total = np.eye(A.shape[0])

    for _ in range(max_iter):
        # QR Decomposition
        Q, R = np.linalg.qr(A_k)

        # recombine in reverse order to converge towards a diagonal matrix
        A_k = R @ Q

        # accumulate the orthogonal transformations to get eigenvectors
        Q_total = Q_total @ Q

        # check for convergence: are the off-diagonal elements close to zero?
        off_diagonal_mask = ~np.eye(A.shape[0], dtype=bool)
        if np.max(np.abs(A_k[off_diagonal_mask])) < tol:
            break

    # return the diagonal values (eigenvalues) and the transformation matrix,
    # whose columns are the eigenvectors
    eigenvalues = np.diagonal(A_k)
    eigenvectors = Q_total
    return eigenvalues, eigenvectors


# --- Utility Functions for Quaternions and Rotations (Lie Group SO(3)) ---

def quat_multiply(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quat_from_axis_angle(axis, angle):
    """Creates a quaternion from an axis and angle (Lie algebra to Lie group)."""
    axis = np.asarray(axis)
    # Ensure axis is a unit vector
    norm = np.linalg.norm(axis)
    if norm < 1e-9: return np.array([1.0, 0, 0, 0])
    axis = axis / norm
    w = np.cos(angle / 2.0)
    x, y, z = axis * np.sin(angle / 2.0)
    return np.array([w, x, y, z])

def quat_to_rotation_matrix(q):
    """Converts a quaternion into a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

# --- Physics Functions ---

def gravitational_force(pos_comet, pos_sun, m_comet, m_sun):
    """Calculates gravitational force vector exerted by the sun on the comet."""
    r_vec = pos_sun - pos_comet
    r_dist = np.linalg.norm(r_vec)
    if r_dist < 1.0:  # Avoid singularity
        return np.zeros(3)
    force_mag = G * (m_comet * m_sun) / (r_dist**2)
    return force_mag * (r_vec / r_dist)

def force_jacobian(pos, m, m_sun):
    """
    Computes the Jacobian of the gravitational force F(pos).
    F(pos) = -G * m * m_sun * pos / ||pos||^3
    """
    r_norm = np.linalg.norm(pos)
    if r_norm < 1e-6: return np.zeros((3,3))
    r5 = r_norm**5
    c = G * m * m_sun
    x, y, z = pos
    
    J = np.zeros((3, 3))
    J[0, 0] = c * (3*x*x - r_norm**2) / r5
    J[0, 1] = c * (3*x*y) / r5
    J[0, 2] = c * (3*x*z) / r5
    J[1, 0] = c * (3*y*x) / r5
    J[1, 1] = c * (3*y*y - r_norm**2) / r5
    J[1, 2] = c * (3*y*z) / r5
    J[2, 0] = c * (3*z*x) / r5
    J[2, 1] = c * (3*z*y) / r5
    J[2, 2] = c * (3*z*z - r_norm**2) / r5
    return -J

def calculate_total_energy(body, sun_pos, sun_mass):
    # Potential Energy
    r_dist = np.linalg.norm(sun_pos - body.position)
    potential_energy = -G * body.mass * sun_mass / r_dist

    # Translational Kinetic Energy
    trans_ke = 0.5 * body.mass * np.dot(body.velocity, body.velocity)

    # Rotational Kinetic Energy
    rot_matrix = quat_to_rotation_matrix(body.orientation)
    ang_vel_body = rot_matrix.T @ body.angular_velocity_world
    rot_ke = 0.5 * np.dot(ang_vel_body, body.inertia_tensor_body @ ang_vel_body)

    return potential_energy + trans_ke + rot_ke

# --- GUI Class ---
class DebugGUI:
    def __init__(self, screen, font, start_x=SIM_WIDTH, start_y=10):
        self.screen = screen
        self.font = font
        self.x = start_x
        self.y = start_y
        self.line_height = font.get_height() + 2

    def _format_np(self, item):
        if isinstance(item, np.ndarray):
            if item.ndim == 1:
                return f"[{', '.join(f'{x:8.2f}' for x in item)}]"
            else:
                return str(item)
        return str(item)

    def draw(self, title, data, color):
        # Draw title
        title_surf = self.font.render(title, True, BLACK)
        self.screen.blit(title_surf, (self.x + 5, self.y))
        self.y += self.line_height * 1.5

        # Draw data rows
        for key, value in data.items():
            text = f"{key:<18}: {self._format_np(value)}"
            text_surf = self.font.render(text, True, color)
            self.screen.blit(text_surf, (self.x + 10, self.y))
            self.y += self.line_height
        self.y += self.line_height # Add spacing

# --- Rigid Body Class ---

class RigidBody:
    def __init__(self, initial_pos_com, initial_vel, initial_ang_vel_body, raw_vertices, mass_per_vertex):
        self.raw_vertices = np.array(raw_vertices, dtype=float)
        self.mass = mass_per_vertex * len(self.raw_vertices)

        # 1. Calculate CoM and the RAW Body Frame Inertia Tensor
        self.com_offset_body, raw_inertia_tensor = compute_com_and_tensor(self.raw_vertices, mass_per_vertex)
        
        # 2. Diagonalize to find Principal Axes (eigenvectors) and Principal Moments (eigenvalues)
        principal_moments, principal_axes = qr_diagonalization(raw_inertia_tensor)
        
        # 3. Overwrite the inertia tensor with the cleanly diagonalized version
        self.inertia_tensor_body = np.diag(principal_moments)
        
        # Inverse of a diagonal matrix is just 1 over the diagonal elements
        self.inertia_tensor_body_inv = np.diag(1.0 / principal_moments)
        
        # 4. Define vertices relative to the CoM (still in the old unrotated frame)
        vertices_com_raw = self.raw_vertices - self.com_offset_body

        # 5. Rotate vertices into the new Principal Axes frame!
        # The columns of principal_axes are the new basis vectors. 
        # Matrix multiplying (N,3) by (3,3) effectively applies the transpose (inverse) 
        # of the rotation matrix, successfully moving the points INTO the new frame.
        self.vertices_com_frame = vertices_com_raw @ principal_axes

        # State variables
        self.position = np.array(initial_pos_com, dtype=float) # Position of the CoM
        self.linear_momentum = self.mass * np.array(initial_vel, dtype=float)
        
        self.orientation = quat_from_axis_angle([0,0,1], 0) # Assuming this function exists elsewhere
        
        # 6. Set angular momentum
        # NOTE: If initial_ang_vel_body was defined in the OLD raw vertex frame, 
        # you need to rotate it into the new frame just like the vertices:
        # initial_ang_vel_body = np.array(initial_ang_vel_body, dtype=float) @ principal_axes
        self.angular_momentum_body = self.inertia_tensor_body @ np.array(initial_ang_vel_body, dtype=float)

    @property
    def velocity(self):
        return self.linear_momentum / self.mass

    @property
    def rotation_matrix(self):
        return quat_to_rotation_matrix(self.orientation)
    
    @property
    def angular_velocity_world(self):
        # 1. Calculate angular velocity in the body frame: w_body = I_body^-1 * L_body
        angular_velocity_body = self.inertia_tensor_body_inv @ self.angular_momentum_body

        # 2. Rotate the resulting vector into the world frame
        return self.rotation_matrix @ angular_velocity_body

    @property
    def body_origin_world(self):
        # 1. The vector from CoM to the raw origin is the negative of the offset
        raw_offset_to_origin = -self.com_offset_body

        # 2. Convert this offset into the Principal Axes frame (using the same logic as the vertices)
        # Note: raw_offset_to_origin is a 1D array of shape (3,), so the @ operator works perfectly here
        offset_in_principal_frame = raw_offset_to_origin @ self.principal_axes

        # 3. Rotate to world space and add to the current CoM world position
        return self.position + self.rotation_matrix @ offset_in_principal_frame

    def get_world_vertices(self):
        # Rotate vertices (which are relative to CoM) and translate to world CoM position
        rotated_vertices = (self.rotation_matrix @ self.vertices_com_frame.T).T
        return rotated_vertices + self.position

# --- Main Simulation ---

def main():
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("monospace", 12)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2.5D Rigid Body Simulation")
    clock = pygame.time.Clock()

    comet_verts = [
        [-20, -15, 0], [0, -15, 0], [20, -10, 0], [20, 10, 0], 
        [0, 15, 0], [-20, 10, 0], [-10, 0, 0]
    ]
    
    initial_pos = [200.0, 0.0, 0.0]
    initial_vel = [0.0, 10.0, 0.0]
    initial_ang_vel_body = [0.0, 0.0, 0.5] # Spinning around Z

    comet = RigidBody(
        initial_pos_com=initial_pos, 
        initial_vel=initial_vel, 
        initial_ang_vel_body=initial_ang_vel_body, 
        raw_vertices=comet_verts, 
        mass_per_vertex=COMET_MASS_PER_VERTEX
    )
    # Set initial angular momentum based on initial angular velocity
    comet.angular_momentum_body = comet.inertia_tensor_body @ np.array(initial_ang_vel_body)

    # --- Initial Conditions Logging ---
    print("--- Initial Conditions ---")
    print(f"Position: {comet.position}")
    print(f"Velocity: {comet.velocity}")
    print(f"Linear Momentum: {comet.linear_momentum}")
    print(f"Angular Vel (Body): {initial_ang_vel_body}")
    print(f"Angular Momentum (Body): {comet.angular_momentum_body}")
    print(f"Inertia Tensor (Body):\n{comet.inertia_tensor_body}")
    print("--------------------------")

    sun_pos = np.zeros(3)
    gui = DebugGUI(screen, font)
    
    initial_energy = calculate_total_energy(comet, sun_pos, SUN_MASS)
    quat_norm_error = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Physics Update ---
        pos_curr, vel_curr, m = comet.position, comet.velocity, comet.mass
        pos_next = pos_curr + vel_curr * DT # Initial guess

        for _ in range(5):
            mid_pos = (pos_curr + pos_next) / 2.0
            force = gravitational_force(mid_pos, sun_pos, m, SUN_MASS)
            g = pos_next - pos_curr - DT * vel_curr - (DT**2 / (2 * m)) * force
            J_g = np.identity(3) - (DT**2 / (4 * m)) * force_jacobian(mid_pos, m, SUN_MASS)
            delta_pos = np.linalg.solve(J_g, -g)
            pos_next += delta_pos
            if np.linalg.norm(delta_pos) < 1e-6: break

        mid_pos_final = (pos_curr + pos_next) / 2.0
        final_force = gravitational_force(mid_pos_final, sun_pos, m, SUN_MASS)
        vel_next = vel_curr + DT / m * final_force

        comet.position = pos_next
        comet.linear_momentum = m * vel_next

        # No external torque, so angular momentum in body frame is constant
        # Rotational update:
        angle = np.linalg.norm(comet.angular_velocity_world) * DT
        if angle > 1e-9:
            axis = comet.angular_velocity_world / (angle/DT)
            delta_rotation = quat_from_axis_angle(axis, angle)
            comet.orientation = quat_multiply(delta_rotation, comet.orientation)
            
            quat_norm_error = abs(1.0 - np.linalg.norm(comet.orientation))
            comet.orientation /= np.linalg.norm(comet.orientation)

        # --- Drawing ---
        screen.fill(WHITE)
        
        # --- Draw GUI Panel ---
        gui.y = 10 # Reset Y position
        screen.fill(GUI_BG, (SIM_WIDTH, 0, GUI_WIDTH, HEIGHT))
        
        const_data = {
            "Total Mass": f"{comet.mass:.2f}",
            "Mass/Vertex": f"{comet.mass / len(comet.vertices_com_frame):.2f}",
            "I_body xx": f"{comet.inertia_tensor_body[0,0]:.2f}",
            "I_body yy": f"{comet.inertia_tensor_body[1,1]:.2f}",
            "I_body zz": f"{comet.inertia_tensor_body[2,2]:.2f}",
        }
        gui.draw("Constants", const_data, BLACK)

        state_data = {
            "Position": comet.position,
            "Lin Momentum": comet.linear_momentum,
            "Ang Momentum (Body)": comet.angular_momentum_body,
            "Ang Velocity (World)": comet.angular_velocity_world,
            "Quaternion": comet.orientation,
        }
        gui.draw("State Variables", state_data, BLUE)
        
        current_energy = calculate_total_energy(comet, sun_pos, SUN_MASS)
        energy_error = abs(current_energy - initial_energy)
        error_data = {
            "Energy Drift (Alg)": f"{energy_error:e}",
            "Quat Norm Err (FP)": f"{quat_norm_error:e}",
        }
        gui.draw("Error Estimation", error_data, RED)

        # --- Draw Simulation ---
        def world_to_screen(pos):
            pos_2d = pos[:2]
            screen_pos = (pos_2d - CAMERA_POS[:2]) * CAMERA_ZOOM + np.array([SIM_WIDTH/2, SIM_HEIGHT/2])
            return screen_pos.astype(int)

        pygame.draw.circle(screen, BLACK, world_to_screen(sun_pos), 10)

        world_verts = comet.get_world_vertices()
        screen_points = [world_to_screen(v) for v in world_verts]
        pygame.draw.polygon(screen, BLACK, screen_points, 1)

        pygame.draw.circle(screen, RED, world_to_screen(comet.position), 3)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    pygame.font.quit()

if __name__ == '__main__':
    main()
