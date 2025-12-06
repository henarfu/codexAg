import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


def _xp_from_array(arr):
    if _HAS_CUPY and isinstance(arr, cp.ndarray):
        return cp
    return np


def gradient(img, xp=None):
    """Forward differences with Neumann boundary conditions."""
    xp = xp or _xp_from_array(img)
    gx = xp.zeros_like(img)
    gy = xp.zeros_like(img)
    gx[:-1, :] = img[1:, :] - img[:-1, :]
    gy[:, :-1] = img[:, 1:] - img[:, :-1]
    return gx, gy


def divergence(gx, gy, xp=None):
    """Negative adjoint of the forward-difference gradient (so -div is the adjoint)."""
    xp = xp or _xp_from_array(gx)
    div = xp.zeros_like(gx)
    div[:-1, :] += gx[:-1, :]
    div[1:, :] -= gx[:-1, :]
    div[:, :-1] += gy[:, :-1]
    div[:, 1:] -= gy[:, :-1]
    return div


def laplacian(img, xp=None):
    xp = xp or _xp_from_array(img)
    # -div(grad) yields a positive semi-definite Laplacian.
    return -divergence(*gradient(img, xp=xp), xp=xp)


def shrink_iso(gx, gy, thresh: float, xp=None):
    """Isotropic soft-thresholding."""
    xp = xp or _xp_from_array(gx)
    mag = xp.sqrt(gx * gx + gy * gy)
    scale = xp.maximum(0.0, 1.0 - thresh / xp.maximum(mag, 1e-12))
    return gx * scale, gy * scale


def tv_norm(gx, gy, xp=None):
    xp = xp or _xp_from_array(gx)
    return xp.sum(xp.sqrt(gx * gx + gy * gy))


def _to_float(val, xp):
    if xp is np:
        return float(val)
    return float(cp.asnumpy(val))


def _cg_gpu(matvec, b, x0, maxiter, tol, xp):
    x = x0.copy()
    r = b - matvec(x)
    p = r.copy()
    rs_old = xp.vdot(r, r)
    for _ in range(maxiter):
        Ap = matvec(p)
        denom = xp.vdot(p, Ap)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = xp.vdot(r, r)
        if float(xp.sqrt(rs_new)) < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x


def tval3_admm(
    A,
    AT,
    y,
    shape,
    lam: float = 0.1,
    rho: float = 2.0,
    max_iters: int = 200,
    cg_iters: int = 60,
    cg_tol: float = 1e-5,
    tol: float = 1e-4,
    verbose: bool = False,
    xp=np,
):
    """
    TV minimization with an augmented Lagrangian / ADMM scheme (TVAL3-style).

    Args:
        A, AT: forward and adjoint operators on flattened arrays.
        y: measurements.
        shape: tuple (H, W) of the target image.
        lam: TV weight.
        rho: augmented Lagrangian penalty.
        max_iters: outer ADMM iterations.
        cg_iters: max iterations of the inner CG solve.
        cg_tol: tolerance for CG.
        tol: stopping tolerance for primal/dual residuals.
        verbose: print progress if True.
        xp: backend array module (numpy or cupy).

    Returns:
        x (2D array) and history dict with objectives and residuals.
    """
    h, w = shape
    n = h * w
    x = xp.zeros((h, w), dtype=xp.float64)
    dx = xp.zeros_like(x)
    dy = xp.zeros_like(x)
    ux = xp.zeros_like(x)
    uy = xp.zeros_like(x)

    history = {"objective": [], "primal_res": [], "dual_res": []}
    At_y = AT(y).reshape(shape)

    def linop(vec):
        img = vec.reshape(shape)
        Ax = A(vec)
        AtAx = AT(Ax)
        lap = laplacian(img, xp=xp).ravel()
        return AtAx + rho * lap

    use_gpu = xp is not np
    if not use_gpu:
        lin = LinearOperator((n, n), matvec=linop, dtype=np.float64)

    for k in range(1, max_iters + 1):
        rhs = At_y - rho * divergence(dx - ux, dy - uy, xp=xp)

        if use_gpu:
            x_flat = _cg_gpu(
                matvec=linop, b=rhs.ravel(), x0=x.ravel(), maxiter=cg_iters, tol=cg_tol, xp=xp
            )
            info = 0
        else:
            x_flat, info = cg(
                lin, rhs.ravel(), x0=x.ravel(), maxiter=cg_iters, rtol=cg_tol, atol=0.0
            )
        if info != 0 and verbose:
            print(f"[CG] iter={k} did not fully converge (info={info}).")
        x = x_flat.reshape(shape)

        gx, gy = gradient(x, xp=xp)
        tmpx, tmpy = gx + ux, gy + uy
        prev_dx, prev_dy = dx, dy
        dx, dy = shrink_iso(tmpx, tmpy, lam / rho, xp=xp)
        ux += gx - dx
        uy += gy - dy

        # Residuals and objective for monitoring.
        prim = xp.sqrt(xp.sum((gx - dx) ** 2 + (gy - dy) ** 2))
        dual = rho * xp.sqrt(xp.sum((dx - prev_dx) ** 2 + (dy - prev_dy) ** 2))
        obj = 0.5 * xp.linalg.norm(A(x.ravel()) - y) ** 2 + lam * tv_norm(gx, gy, xp=xp)

        history["objective"].append(_to_float(obj, xp))
        history["primal_res"].append(_to_float(prim, xp))
        history["dual_res"].append(_to_float(dual, xp))

        if verbose:
            norm_x = _to_float(xp.linalg.norm(x), xp)
            print(
                f"[{k:03d}] obj={history['objective'][-1]:.4e} "
                f"prim={history['primal_res'][-1]:.3e} "
                f"dual={history['dual_res'][-1]:.3e} "
                f"||x||={norm_x:.3e}"
            )

        if max(history["primal_res"][-1], history["dual_res"][-1]) < tol:
            break

    return x, history
