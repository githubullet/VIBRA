"""Closed form maps."""
import torch
import torch.nn.functional as F


def ridge_fit(Z, Y, lam):
    """min ||[Z 1] W - Y||^2 + lam * (tr(Z^T Z) / d) * ||W||^2, bias unpenalized."""
    N, d = Z.shape
    Za = torch.cat([Z, torch.ones(N, 1, device=Z.device)], 1).double()
    A = Za.T @ Za
    reg = torch.eye(d + 1, device=Z.device, dtype=torch.float64) * lam * torch.trace(A[:d, :d]) / d
    reg[-1, -1] = 0
    Wb = torch.linalg.solve(A + reg, Za.T @ Y.double()).float()
    return Wb[:-1], Wb[-1]


def procrustes_fit(V_new, U_old):
    """Orthogonal map from the new to the old space; the smaller side is zero padded."""
    d_new, d_old = V_new.shape[1], U_old.shape[1]
    if d_new < d_old:
        V = F.pad(V_new, (0, d_old - d_new)).double()
        P, _, Qt = torch.linalg.svd(V.T @ U_old.double(), full_matrices=False)
        return (P @ Qt).float()[:d_new]
    U = F.pad(U_old, (0, d_new - d_old)).double()
    P, _, Qt = torch.linalg.svd(V_new.double().T @ U, full_matrices=False)
    return (P @ Qt).float()


def joint_procrustes(tr):
    """Procrustes map P (d_new x d_old) fitted jointly on the image and caption features of the training set."""
    R = procrustes_fit(torch.cat([tr["new_txt"], tr["new_img"]]), torch.cat([tr["old_txt"], tr["old_img"]]))
    return R[:, :tr["old_img"].shape[1]]
