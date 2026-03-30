import numpy as np
import pygame
import scipy.linalg

# --- Constants ---
SIM_WIDTH, SIM_HEIGHT = 800, 600
GUI_WIDTH = 400
WIDTH, HEIGHT = SIM_WIDTH + GUI_WIDTH, SIM_HEIGHT
WHITE, BLACK, RED, BLUE, GUI_BG = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (240, 240, 240)
FPS, DT = 60, 1 / 60

# --- Simulation Parameters ---
G = 1.0  # Scaled for simulation. Original is 6.674e-11.
SUN_MASS = 10000.0
COMET_MASS_PER_VERTEX = 1.0
CAMERA_ZOOM = 0.5
CAMERA_POS = np.array([SIM_WIDTH / 2, SIM_HEIGHT / 2])


# --- Utility Functions for Frame of Reference and Inertia Tensor Computation ---
def compute_com_and_tensor(verts, m=1.0):
    com = np.mean(verts, axis=0)
    verts_centered = verts - com 
    I = np.zeros((3,3))
    for v in verts_centered:
        x, y, z = v
        I[0, 0] += m * (y**2 + z**2)
        I[1, 1] += m * (x**2 + z**2)
        I[2, 2] += m * (x**2 + y**2)
        I[0, 1] -= m * (x * y); I[1, 0] -= m * (x * y)
        I[0, 2] -= m * (x * z); I[2, 0] -= m * (x * z)
        I[1, 2] -= m * (y * z); I[2, 1] -= m * (y * z)
    return com, I

def qr_diagonalization(A, tol=1e-10, max_iter=1000):
    A_k = np.copy(A)
    Q_total = np.eye(A.shape[0])
    for _ in range(max_iter):
        if not np.all(np.isfinite(A_k)): break
        Q, R = np.linalg.qr(A_k)
        A_k = R @ Q
        Q_total = Q_total @ Q
        if np.max(np.abs(A_k[~np.eye(A.shape[0], dtype=bool)])) < tol: break
    return np.diagonal(A_k), Q_total

# --- Lie Algebra/Group and Rotation Helpers ---
def hat(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def vee(S): return np.array([S[2, 1], S[0, 2], S[1, 0]])

# --- Physics Model ---
def get_force_and_torque(body, sun_pos, sun_mass):
    force_world = gravitational_force(body.position, sun_pos, body.mass, sun_mass)
    torque_body = body.orientation.T @ np.zeros(3)
    return force_world, torque_body

def gravitational_force(pos, sun_pos, m, sun_m):
    r_vec = sun_pos - pos
    dist = np.linalg.norm(r_vec)
    if dist < 1.0: return np.zeros(3)
    return (G * m * sun_m / dist**3) * r_vec

def calculate_total_energy(body, sun_pos, sun_mass):
    r_dist = np.linalg.norm(sun_pos - body.position)
    potential_energy = -G * body.mass * sun_mass / r_dist if r_dist > 0 else 0
    trans_ke = 0.5 * body.mass * np.dot(body.velocity, body.velocity)
    rot_ke = 0.5 * np.dot(body.angular_velocity_body, body.Pi)
    return potential_energy + trans_ke + rot_ke

# --- UI Classes ---
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
                return str(item).replace('\n', ' ')
        if isinstance(item, float):
            return f"{item:8.2f}"
        return str(item)

    def draw(self, title, data, color):
        title_surf = self.font.render(title, True, BLACK)
        self.screen.blit(title_surf, (self.x + 5, self.y))
        self.y += self.line_height * 1.5
        for key, value in data.items():
            text = f"{key:<22}: {self._format_np(value)}"
            text_surf = self.font.render(text, True, color)
            self.screen.blit(text_surf, (self.x + 10, self.y))
            self.y += self.line_height
        self.y += self.line_height

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, start_val):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val, self.max_val = min_val, max_val
        self.val = start_val
        self.dragging = False
        self.handle_w = 10
        self.update_handle_pos()

    def update_handle_pos(self):
        percent = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + percent * (self.rect.w - self.handle_w)
        self.handle_rect = pygame.Rect(handle_x, self.rect.y, self.handle_w, self.rect.h)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.val = self.min_val + (event.pos[0] - self.rect.x) / self.rect.w * (self.max_val - self.min_val)
            self.val = max(self.min_val, min(self.max_val, self.val))
            self.update_handle_pos()
    
    def get_value(self): return int(self.val)

    def draw(self, screen, font):
        pygame.draw.rect(screen, (200, 200, 200), self.rect)
        pygame.draw.rect(screen, (100, 100, 100), self.handle_rect)
        text_surf = font.render(f"Speed: {self.get_value()}x", True, BLACK)
        screen.blit(text_surf, (self.rect.right + 10, self.rect.y))

# --- ODE Solver ---
def implicit_midpoint_step(body, h, sun_pos, sun_mass):
    x_n, p_n, R_n, Pi_n = body.position, body.linear_momentum, body.orientation, body.Pi
    I_inv, m_inv = body.inertia_tensor_body_inv, 1.0 / body.mass
    F_n, tau_n = get_force_and_torque(body, sun_pos, sun_mass)
    x_guess, p_guess = x_n + h*(p_n*m_inv), p_n + h*F_n
    omega_n = I_inv @ Pi_n
    R_guess = R_n @ scipy.linalg.expm(h * hat(omega_n))
    Pi_guess = Pi_n + h * (np.cross(Pi_n, omega_n) + tau_n)
    for _ in range(10):
        x_mid, p_mid, Pi_mid = 0.5*(x_n+x_guess), 0.5*(p_n+p_guess), 0.5*(Pi_n+Pi_guess)
        omega_mid = I_inv @ Pi_mid
        R_mid = R_n @ scipy.linalg.expm(0.5 * h * hat(omega_mid))
        F_mid, tau_mid = get_force_and_torque(type('obj',(object,),{'position':x_mid,'orientation':R_mid,'mass':body.mass}), sun_pos, sun_mass)
        x_res, p_res = x_guess-(x_n+h*p_mid*m_inv), p_guess-(p_n+h*F_mid)
        Pi_res = Pi_guess - (Pi_n + h * (np.cross(Pi_mid, omega_mid) + tau_mid))
        R_err_matrix = (R_n @ scipy.linalg.expm(h * hat(omega_mid))) @ R_guess.T
        r_res = vee(R_err_matrix - R_err_matrix.T)
        residual = np.concatenate([x_res, p_res, Pi_res, r_res])
        if np.linalg.norm(residual) < 1e-9: break
        J = np.zeros((12, 12))
        J[0:3, 0:3], J[0:3, 3:6] = np.eye(3), -0.5*h*m_inv*np.eye(3)
        J[3:6, 3:6] = np.eye(3)
        J[6:9, 6:9] = np.eye(3) - 0.5*h*(hat(omega_mid) - hat(I_inv@Pi_mid)) @ I_inv
        J[9:12, 9:12] = np.eye(3)
        try: correction = np.linalg.solve(J, -residual)
        except np.linalg.LinAlgError: break
        dx, dp, dPi, d_theta = correction[0:3], correction[3:6], correction[6:9], correction[9:12]
        x_guess, p_guess, Pi_guess = x_guess+dx, p_guess+dp, Pi_guess+dPi
        R_guess = scipy.linalg.expm(hat(d_theta)) @ R_guess
    body.position, body.linear_momentum, body.orientation, body.Pi = x_guess, p_guess, R_guess, Pi_guess

# --- Rigid Body Class ---
class RigidBody:
    def __init__(self, initial_pos_com, initial_vel, initial_ang_vel_raw, raw_vertices, mass_per_vertex):
        self.mass = mass_per_vertex * len(raw_vertices)
        self.com_offset_body, raw_I = compute_com_and_tensor(raw_vertices, mass_per_vertex)
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
    def velocity(self): return self.linear_momentum / self.mass
    @property
    def angular_velocity_body(self): return self.inertia_tensor_body_inv @ self.Pi
    @property
    def body_origin_world(self): return self.position - self.orientation @ (self.principal_axes_R.T @ self.com_offset_body)
    def get_world_vertices(self): return (self.orientation @ self.vertices_com_frame.T).T + self.position

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
        initial_pos_com=initial_pos_com, initial_vel=initial_vel,
        initial_ang_vel_raw=initial_ang_vel_raw,
        raw_vertices=[[-20, -15, 0], [0, -15, 0], [20, -10, 0], [20, 10, 0], [0, 15, 0], [-20, 10, 0], [-10, 0, 0]],
        mass_per_vertex=COMET_MASS_PER_VERTEX
    )
    
    print("--- Body Frame & Inertia Analysis ---")
    print(f"CoM Offset from Body Origin: {comet.com_offset_body}")
    print(f"Principal Moments (I_body):\n{np.diagonal(comet.inertia_tensor_body)}")
    print(f"Principal Axes (R_to_principal):\n{comet.principal_axes_R}")
    print("------------------------------------")

    sun_pos = np.zeros(3)
    gui = DebugGUI(screen, font)
    time_slider = Slider(SIM_WIDTH + 20, HEIGHT - 40, 150, 20, 1, 20, 1)
    
    initial_energy = calculate_total_energy(comet, sun_pos, SUN_MASS)
    print("--- Orbit Prediction ---")
    print(f"Initial Total Energy: {initial_energy:.2f}")
    if initial_energy < 0:
        print("Prediction: BOUND ORBIT (Elliptical). Total energy is negative.")
    else:
        print("Prediction: ESCAPE TRAJECTORY (Parabolic/Hyperbolic). Total energy is non-negative.")
    print("------------------------")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            time_slider.handle_event(event)

        for _ in range(time_slider.get_value()):
            implicit_midpoint_step(comet, DT, sun_pos, SUN_MASS)

        screen.fill(WHITE)
        gui.y = 10
        screen.fill(GUI_BG, (SIM_WIDTH, 0, GUI_WIDTH, HEIGHT))
        
        gui.draw("Constants", {"Total Mass": comet.mass, "CoM Offset (Body)": comet.com_offset_body}, BLACK)
        gui.draw("State", {"CoM Position": comet.position, "Body Origin": comet.body_origin_world,
                           "Lin Momentum": comet.linear_momentum, "Ang Momentum (Body)": comet.Pi,
                           "Ang Velocity (Body)": comet.angular_velocity_body}, BLUE)
        current_energy = calculate_total_energy(comet, sun_pos, SUN_MASS)
        gui.draw("Error", {"Energy Drift": f"{abs(current_energy - initial_energy):e}"}, RED)
        time_slider.draw(screen, font)
        
        def world_to_screen(pos):
            p = (pos[:2] - CAMERA_POS[:2]) * CAMERA_ZOOM + np.array([SIM_WIDTH/2, SIM_HEIGHT/2])
            return p.astype(int)

        pygame.draw.circle(screen, BLACK, world_to_screen(sun_pos), 10)
        pygame.draw.polygon(screen, BLACK, [world_to_screen(v) for v in comet.get_world_vertices()], 1)
        pygame.draw.circle(screen, RED, world_to_screen(comet.position), 4)
        pygame.draw.circle(screen, BLUE, world_to_screen(comet.body_origin_world), 4)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    pygame.font.quit()

if __name__ == '__main__':
    main()
