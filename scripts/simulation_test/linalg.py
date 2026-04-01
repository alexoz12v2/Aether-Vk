import numpy as np

def compute_com_and_tensor(verts, m=1.0):
    com = np.mean(verts, axis=0)
    verts_centered = verts - com
    I = np.zeros((3, 3))
    for v in verts_centered:
        x, y, z = v
        I[0, 0] += m * (y**2 + z**2)
        I[1, 1] += m * (x**2 + z**2)
        I[2, 2] += m * (x**2 + y**2)
        I[0, 1] -= m * (x * y)
        I[1, 0] -= m * (x * y)
        I[0, 2] -= m * (x * z)
        I[2, 0] -= m * (x * z)
        I[1, 2] -= m * (y * z)
        I[2, 1] -= m * (y * z)
    return com, I


def qr_diagonalization(A, tol=1e-10, max_iter=1000):
    A_k = np.copy(A)
    Q_total = np.eye(A.shape[0])
    for _ in range(max_iter):
        if not np.all(np.isfinite(A_k)):
            break
        Q, R = np.linalg.qr(A_k)
        A_k = R @ Q
        Q_total = Q_total @ Q
        if np.max(np.abs(A_k[~np.eye(A.shape[0], dtype=bool)])) < tol:
            break
    return np.diagonal(A_k), Q_total

def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])
