import torch
import torch.nn.functional as F


def pad_conv2d(x, kernel):
    """Apply 2D conv with 'same' padding for fixed kernels."""
    c = x.shape[1]
    weight = kernel.to(x.device).expand(c, 1, *kernel.shape)
    return F.conv2d(x, weight, padding=kernel.shape[-1] // 2, groups=c)


def codelet_data(x, A, y, _params):
    # G_DATA = A^T(Ax - y)
    return A.A_adjoint(A(x) - y)


def codelet_robust_data(x, A, y, params):
    # w(t) = 1 / (1 + (t/tau)^p); G = A^T(w(|Ax-y|) ⊙ (Ax-y))
    tau = params.get("tau_rob", 0.1)
    p = params.get("p_rob", 1.0)
    r = A(x) - y
    w = 1.0 / (1.0 + (r.abs() / tau) ** p)
    return A.A_adjoint(w * r)


def codelet_precond_data(x, A, y, params):
    # G = sum_i c_i (K_i * g), g = A^T(Ax - y)
    g = A.A_adjoint(A(x) - y)
    coeffs = params.get("coeffs", [])
    kernels = params.get("kernels", [])
    if not coeffs or not kernels:
        return g
    acc = 0.0
    for c, k in zip(coeffs, kernels):
        acc = acc + c * pad_conv2d(g, k)
    return acc


def codelet_ares(x, A, y, params):
    # G = alpha * A^T A b + beta * A^T sign(b), b = A^T(Ax - y)
    alpha = params.get("alpha", 0.0)
    beta = params.get("beta", 0.0)
    b = A.A_adjoint(A(x) - y)
    term1 = alpha * A.A_adjoint(A(b))
    term2 = beta * A.A_adjoint(torch.sign(b))
    return term1 + term2


def codelet_l2(x, _params):
    return x


def codelet_l1(x, _params):
    return torch.sign(x)


def codelet_tv(x, _params):
    alpha = _params.get("alpha_tv", 1e-3)
    # Isotropic TV gradient approx with small smoothing alpha.
    dx = torch.diff(x, dim=3, append=x[:, :, :, -1:].clone())
    dy = torch.diff(x, dim=2, append=x[:, :, -1:, :].clone())
    mag = torch.sqrt(dx**2 + dy**2 + alpha**2)
    dx_norm = dx / mag
    dy_norm = dy / mag
    dx_back = torch.diff(dx_norm, dim=3, prepend=dx_norm[:, :, :, :1].clone())
    dy_back = torch.diff(dy_norm, dim=2, prepend=dy_norm[:, :, :1, :].clone())
    return dx_back + dy_back


def codelet_graph(x, _params):
    # Lx with fixed Laplacian L stored in params
    L = _params.get("L", None)
    if L is None:
        return x * 0.0
    # Flatten spatial dims, apply L, then reshape
    b, c, h, w = x.shape
    y = x.reshape(b, c, -1)
    out = torch.einsum("ij,bcj->bci", L.to(x.device), y)
    return out.reshape_as(x)


def codelet_nonlocal(x, _params):
    Lnl = _params.get("L_nl", None)
    if Lnl is None:
        return x * 0.0
    b, c, h, w = x.shape
    y = x.reshape(b, c, -1)
    out = torch.einsum("ij,bcj->bci", Lnl.to(x.device), y)
    return out.reshape_as(x)


def codelet_sparse_w(x, _params):
    W = _params.get("W", None)
    tau = _params.get("tau_sp", 0.01)
    if W is None:
        return x * 0.0
    b, c, h, w = x.shape
    y = x.reshape(b, c, -1)
    Wx = torch.einsum("ij,bcj->bci", W.to(x.device), y)
    Wx_soft = torch.sign(Wx) * torch.clamp(Wx.abs() - tau, min=0.0)
    diff = Wx_soft - Wx
    out = torch.einsum("ij,bcj->bci", W.t().to(x.device), diff)
    return out.reshape_as(x)


def codelet_nullspace(x, _params):
    U = _params.get("U", None)
    tau = _params.get("tau_n", 0.01)
    if U is None:
        return x * 0.0
    b, c, h, w = x.shape
    y = x.reshape(b, c, -1)
    coeff = torch.einsum("qi,bci->bcq", U.t().to(x.device), y)
    coeff_shrunk = torch.sign(coeff) * torch.clamp(coeff.abs() - tau, min=0.0)
    out = torch.einsum("iq,bcq->bci", U.to(x.device), coeff - coeff_shrunk)
    return out.reshape_as(x)


def codelet_bquad(x, _params):
    B = _params.get("B", None)
    if B is None:
        return x * 0.0
    b, c, h, w = x.shape
    y = x.reshape(b, c, -1)
    By = torch.einsum("ij,bcj->bci", B.to(x.device), y)
    out = torch.einsum("ij,bcj->bci", B.t().to(x.device), By)
    return out.reshape_as(x)
