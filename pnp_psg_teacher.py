"""PnP-PSG baseline (0.1 compression, 64x64) using the fixed saved matrix A (AA.npy).

- Forward model: y = A x with A loaded from RESULTS/AA.npy (1228 x 12288).
- Algorithm: PnP proximal gradient (data gradient step + pretrained DnCNN).
- Denoiser: generic DeepInv DnCNN (downloaded weights), optionally fine-tuned checkpoint.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Tuple, List, Callable

import numpy as np
from PIL import Image
import torch

sys.path.append("/home/hdsp/Documents/Henry/pnp")
import deepinv as dinv  # type: ignore

from linear_ops import load_fixed_A, Ax, ATz, estimate_spectral_norm
from codeletsA.train_teacher02p import YOnlyTeacher
from codeletsA.unet_leon import UNetLeon
from codeletsA.train_netBonline import NetBonline
from codeletsA.train_net02Ip import Y0ProjUNet
from codelets import normalize_to_data


def load_image(path: Path, device: torch.device, size: int = 64) -> torch.Tensor:
    """Load RGB image, resize, and return tensor [1,3,H,W] in [0,1]."""
    arr = np.array(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def list_images(folder: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in exts:
            yield p


def make_denoiser(device: torch.device, sigma: float, ckpt_path: Path | None = None):
    model = dinv.models.DnCNN(in_channels=3, out_channels=3, pretrained="download_lipschitz").to(device).eval()
    sigma_tensor = torch.tensor([[sigma]], device=device, dtype=torch.float32)
    if ckpt_path is not None and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded finetuned denoiser from {ckpt_path}")

    def denoise_fn(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(x, sigma=sigma_tensor)

    return denoise_fn


def make_denoiser_bank(device: torch.device, base_dir: Path, sigmas: List[float]) -> List[Callable[[torch.Tensor], torch.Tensor]]:
    paths = [base_dir / "generaldenoiser.pth"]
    for i in range(1, 6):
        paths.append(base_dir / f"denoiser{i}.pth")
    denos = []
    for p, s in zip(paths, sigmas):
        denos.append(make_denoiser(device, s, p))
    return denos


def psnr(x: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((x - x_gt) ** 2).item()
    if mse <= eps:
        return 99.0
    return 10 * math.log10(1.0 / mse)


def pnp_psg_matrix(x_true: torch.Tensor, A: torch.Tensor, denoise_fn, eta: float, iters: int,
                     use_teacher: bool = False, B: torch.Tensor | None = None,
                     teacher_net=None, gamma: float = 1.0, y0_pred=None) -> Tuple[torch.Tensor, float]:
    """PnP-PSG: gradient step on data fidelity, then denoise, using explicit A."""
    Bsz = x_true.shape[0]
    y = Ax(x_true.reshape(Bsz, -1), A)
    x = ATz(y, A).reshape_as(x_true)
    with torch.no_grad():
        for _ in range(iters):
            r = Ax(x.reshape(Bsz, -1), A) - y
            g = ATz(r, A).reshape_as(x)

            # Add teacher gradient
            if use_teacher and B is not None and y0_pred is not None:
                Bx = torch.matmul(x.reshape(Bsz, -1), B.t())
                g_teacher = torch.matmul(Bx - y0_pred, B).reshape_as(x)
                g += gamma * g_teacher

            x = denoise_fn(x - eta * g).clamp(0.0, 1.0)
    return x, psnr(x, x_true)


def main():
    parser = argparse.ArgumentParser(description="PnP-PSG single-pixel baseline (0.1x, 64x64).")
    parser.add_argument("--img-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test", help="Folder with input images.")
    parser.add_argument("--size", type=int, default=64, help="Resize images to this square size.")
    parser.add_argument("--iters", type=int, default=4000, help="Number of PnP-PSG iterations.")
    parser.add_argument("--sigma", type=float, default=0.02, help="Sigma passed to pretrained DnCNN.")
    parser.add_argument("--eta", type=float, default=1.0, help="Step size (fixed).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    parser.add_argument("--denoiser-path", type=str, default="RESULTS/generaldenoiser.pth", help="Optional finetuned denoiser checkpoint.")
    parser.add_argument("--use-denoiser-bank", action="store_true", help="Use general + denoiser1-5 with fixed schedule across iterations.")
    parser.add_argument("--bank-sigmas", type=str, default="0.02,0.005,0.02,0.045,0.08,0.13", help="Comma-separated sigma for bank (general,1..5).")
    parser.add_argument("--save-dir", type=str, default=None, help="Optional output folder for reconstructions.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images.")
    # Teacher arguments
    parser.add_argument("--use-teacher", action="store_true", help="Enable teacher-based fidelity term.")
    parser.add_argument("--teacher-type", type=str, default="unet", choices=["unet", "netbonline"], help="Teacher variant.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for the teacher's contribution.")
    parser.add_argument("--teacher-path", type=str, default="RESULTS/teacher02p.pt", help="Path to the teacher model.")
    parser.add_argument("--B-path", type=str, default="/home/hdsp/RESULTS/B_teacher02.npy", help="Path to the B matrix.")
    parser.add_argument("--netbonline-path", type=str, default="RESULTS/netBonline_ep700.pt", help="Checkpoint for netBonline teacher.")
    parser.add_argument("--netbonline-block", type=int, default=256, help="Block size for netBonline inference over rows.")
    parser.add_argument("--normalize-teacher-grad", action="store_true", help="Normalize teacher grad RMS to data grad RMS before weighting.")
    # GPMT-PnP (teacher gradient with net02p) options
    parser.add_argument("--use-gpmt", action="store_true", help="Enable gradient-augmented PnP with net02p anchor.")
    parser.add_argument("--net02p-path", type=str, default="RESULTS/teacher02p_unet.pt", help="Checkpoint for net02p (y->y0).")
    parser.add_argument("--gpmt-beta0", type=float, default=0.1, help="Initial teacher weight beta.")
    parser.add_argument("--gpmt-beta-decay", type=float, default=4000.0, help="Decay constant for beta schedule (exp(-k/decay)).")
    parser.add_argument("--gpmt-beta-floor", type=float, default=0.0, help="Minimum teacher weight (beta) after decay.")
    parser.add_argument("--gpmt-frac-final", type=float, default=1.0, help="Final fraction of teacher rows kept (linear decay from 1.0 to this value).")
    parser.add_argument("--gpmt-w-unmeasured", action="store_true", help="If set, only apply teacher gradient on unmeasured rows (complement of A).")
    parser.add_argument("--gpmt-use-ip-weights", action="store_true", help="Weight teacher residual by net02Ip sensitivity (gradient) per component.")
    parser.add_argument("--gpmt-use-uncertainty", action="store_true", help="Weight teacher residual by net02p uncertainty via MC sampling.")
    parser.add_argument("--gpmt-unc-samples", type=int, default=8, help="Number of MC samples for uncertainty weighting.")
    parser.add_argument("--gpmt-unc-noise-std", type=float, default=0.01, help="Std of input noise added to y for uncertainty sampling.")
    # Refinement of net02p anchor with net02Ip (pre-PnP)
    parser.add_argument("--use-refine-y0", action="store_true", help="Refine net02p anchor with net02Ip consistency before PnP.")
    parser.add_argument("--net02Ip-path", type=str, default="RESULTS/net02Ip_unetproj.pt", help="Checkpoint for net02Ip (y0->y).")
    parser.add_argument("--refine-steps", type=int, default=2, help="Gradient steps for y0 refinement.")
    parser.add_argument("--refine-lr", type=float, default=1e-3, help="Step size for y0 refinement.")
    parser.add_argument("--refine-lambda", type=float, default=1e-3, help="Weight on ||net02Ip(y0)-y||^2 during refinement.")
    parser.add_argument("--refine-gate-sigma", type=float, default=0.05, help="Gating sigma for teacher weight based on net02Ip residual.")
    # net02Ip-based gating of teacher (component-wise trust)
    parser.add_argument("--use-y0-gate", action="store_true", help="Gate net02p anchor per-component using net02Ip gradients.")
    parser.add_argument("--y0-gate-mode", type=str, default="soft", choices=["soft", "hard"], help="Soft weights or hard mask.")
    parser.add_argument("--y0-gate-sigma", type=float, default=0.0, help="Sigma for gradient-based gating (0 => use mean |g|).")
    parser.add_argument("--y0-gate-thresh", type=float, default=0.5, help="Threshold for hard gating (ignored in soft mode).")
    # Per-image adaptive y0 shrink (cheap per-run regularizer)
    parser.add_argument("--use-y0-adapt", action="store_true", help="Per-image adapt a diagonal shrink on Bx toward net02p(y).")
    parser.add_argument("--y0-adapt-weight", type=float, default=0.1, help="Weight for adaptive y0 regularizer.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    img_dir = Path(args.img_dir)
    img_paths = list(list_images(img_dir))
    if args.max_images is not None:
        img_paths = img_paths[: args.max_images]
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    # Load fixed matrix A (shape [m,n] with m=0.1*n=3684).
    A = load_fixed_A().to(device)
    norm = estimate_spectral_norm(A)
    eta = args.eta if args.eta is not None else 0.9 / (norm**2 + 1e-8)

    # Load teacher model if requested
    # Load B (shared)
    B = torch.from_numpy(np.load(args.B_path)).float().to(device) if Path(args.B_path).exists() else None

    # Load teacher model if requested
    teacher_proj = False
    if args.use_teacher:
        # Common B load
        if B is None:
            B = torch.from_numpy(np.load(args.B_path)).float().to(device)
        teacher_net = None
        if args.teacher_type == "unet":
            teacher_ckpt = torch.load(args.teacher_path, map_location=device)
            if any("proj.weight" in k for k in teacher_ckpt["model"].keys()):
                from codeletsA.train_teacher02p_continued import YToImageUNet
                base_ch = teacher_ckpt.get("base_channel", 64)
                if base_ch == 64 and "unetproj" in args.teacher_path:
                    base_ch = 96
                teacher_net = YToImageUNet(m_in=A.shape[0], m_out=B.shape[0], base_channel=base_ch).to(device)
                teacher_net.load_state_dict(teacher_ckpt["model"], strict=False)
                teacher_proj = True
            else:
                teacher_net = UNetLeon(n_channels=3, base_channel=64).to(device)
                teacher_net.load_state_dict(teacher_ckpt["model"])
            teacher_net.eval()
            print(f"Loaded UNet teacher from {args.teacher_path}")
        elif args.teacher_type == "netbonline":
            ckpt = torch.load(args.netbonline_path, map_location=device)
            normalize_b = ckpt.get("normalize_b", False)
            if normalize_b:
                B = torch.nn.functional.normalize(B, dim=1)
            netbonline = NetBonline(
                m_y=A.shape[0],
                n=B.shape[1],
                pos_size=B.shape[0],
                y_dim=ckpt.get("y_dim", 192),
                b_dim=ckpt.get("b_dim", 96),
                pos_dim=ckpt.get("pos_dim", 64),
            ).to(device)
            netbonline.load_state_dict(ckpt["model"])
            netbonline.eval()
            teacher_net = netbonline
            print(f"Loaded netBonline teacher from {args.netbonline_path} (normalize_b={normalize_b})")
        else:
            raise ValueError(f"Unknown teacher_type {args.teacher_type}")
    else:
        teacher_net = None
    # GPMT / refinement: load net02p (and net02Ip if needed)
    if args.use_gpmt or args.use_refine_y0 or args.use_y0_gate or args.use_y0_adapt or args.gpmt_use_ip_weights:
        if B is None:
            B = torch.from_numpy(np.load(args.B_path)).float().to(device)
        B_norm = torch.nn.functional.normalize(B, dim=1)
        net02p_ckpt = torch.load(args.net02p_path, map_location=device)
        # Choose architecture based on checkpoint keys
        if any("proj.weight" in k for k in net02p_ckpt["model"].keys()):
            from codeletsA.train_teacher02p_continued import YToImageUNet
            base_ch = 96 if "teacher02p_unetproj" in args.net02p_path else 64
            net02p = YToImageUNet(m_in=A.shape[0], m_out=B.shape[0], base_channel=base_ch).to(device)
            net02p.load_state_dict(net02p_ckpt["model"], strict=False)
        else:
            net02p = UNetLeon(n_channels=3, base_channel=64).to(device)
            net02p.load_state_dict(net02p_ckpt["model"])
        net02p.eval()
        for p in net02p.parameters():
            p.requires_grad_(False)
        if args.use_refine_y0 or args.use_y0_gate or args.use_y0_adapt or args.gpmt_use_ip_weights:
            net02Ip_ckpt = torch.load(args.net02Ip_path, map_location=device)
            proj_dim = net02Ip_ckpt.get("proj_dim", 4096)
            base_channel = net02Ip_ckpt.get("base_channel", 64)
            net02Ip = Y0ProjUNet(m_in=B_norm.shape[0], m_out=A.shape[0], proj_dim=proj_dim, base_channel=base_channel).to(device)
            net02Ip.load_state_dict(net02Ip_ckpt["model"])
            net02Ip.eval()
            for p in net02Ip.parameters():
                p.requires_grad_(False)
    else:
        net02p = None
        B_norm = None
        net02Ip = None

    if args.use_denoiser_bank:
        base_dir = Path("RESULTS")
        sigma_bank = [float(s) for s in args.bank_sigmas.split(",")]
        if len(sigma_bank) != 6:
            raise ValueError("bank-sigmas must have 6 comma-separated values (general + 5 denoisers).")
        deno_paths = [base_dir / "generaldenoiser.pth"]
        for i in range(1, 6):
            deno_paths.append(base_dir / f"denoiser{i}.pth")
        # general first, then remaining denoisers sorted by sigma descending
        general_fn = make_denoiser(device, sigma_bank[0], deno_paths[0])
        rest = []
        for p, s in zip(deno_paths[1:], sigma_bank[1:]):
            rest.append((s, make_denoiser(device, s, p)))
        rest_sorted = [fn for _, fn in sorted(rest, key=lambda t: t[0], reverse=True)]
        deno_bank = [general_fn] + rest_sorted
        deno_desc = "bank(general then high→low sigma)"
        denoise_fn = None
    else:
        ckpt_path = Path(args.denoiser_path) if args.denoiser_path else None
        denoise_fn = make_denoiser(device, args.sigma, ckpt_path)
        deno_desc = ckpt_path.name if ckpt_path is not None else "vanilla"

    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    scores = []
    for p in img_paths:
        x_true = load_image(p, device, size=args.size)
        y_true = Ax(x_true.reshape(1, -1), A)
        x0 = ATz(y_true, A).reshape_as(x_true)

        # Precompute net02p anchor y0_p if enabled
        if args.use_gpmt or args.use_refine_y0 or args.use_y0_adapt or args.gpmt_use_ip_weights or args.gpmt_use_uncertainty:
            if B_norm is None:
                B_norm = torch.nn.functional.normalize(B, dim=1)
            with torch.no_grad():
                if args.gpmt_use_uncertainty:
                    preds = []
                    net02p.train()
                    for _ in range(max(1, args.gpmt_unc_samples)):
                        noise = args.gpmt_unc_noise_std * torch.randn_like(y_true)
                        preds.append(net02p(y_true + noise))
                    net02p.eval()
                    y_stack = torch.stack(preds, dim=0)
                    y0_anchor = y_stack.mean(dim=0)
                    y0_std = y_stack.std(dim=0)
                    s0 = torch.median(y0_std) + 1e-8
                    y0_unc_weights = torch.exp(- (y0_std / s0) ** 2).clamp(0.05, 1.0)
                else:
                    y0_anchor = net02p(y_true)
                    y0_unc_weights = None
            row_norms = torch.norm(B, dim=1, keepdim=True)
            y0_anchor = y0_anchor / (row_norms.t() + 1e-8)
            # Optional refinement of y0_anchor using net02Ip consistency
            gamma_local = args.gamma
            if args.use_refine_y0 and net02Ip is not None:
                y0_ref = y0_anchor.detach().clone()
                for _ in range(args.refine_steps):
                    y0_ref.requires_grad_(True)
                    scale = torch.sqrt(torch.mean(y0_ref * y0_ref, dim=1, keepdim=True) + 1e-8).detach()
                    y0_scaled = y0_ref / (scale + 1e-8)
                    y_hat = net02Ip(y0_scaled)
                    loss_anchor = 0.5 * torch.mean((y0_ref - y0_anchor) ** 2)
                    loss_cons = 0.5 * torch.mean((y_hat - y_true) ** 2)
                    loss = loss_anchor + args.refine_lambda * loss_cons
                    loss.backward()
                    with torch.no_grad():
                        y0_ref -= args.refine_lr * y0_ref.grad
                    y0_ref = y0_ref.detach()
                y0_anchor = y0_ref
                # Gate gamma based on net02Ip residual
                scale = torch.sqrt(torch.mean(y0_anchor * y0_anchor, dim=1, keepdim=True) + 1e-8)
                y0_scaled = y0_anchor / (scale + 1e-8)
                with torch.no_grad():
                    y_hat = net02Ip(y0_scaled)
                    r = torch.sqrt(torch.mean((y_hat - y_true) ** 2)).item()
                gate = math.exp(- (r / (args.refine_gate_sigma + 1e-8)) ** 2)
                gamma_local = args.gamma * gate
            # Optional per-component gating using net02Ip sensitivity
            if args.use_y0_gate and net02Ip is not None:
                y0_gate = y0_anchor.detach().clone()
                y0_gate.requires_grad_(True)
                scale = torch.sqrt(torch.mean(y0_gate * y0_gate, dim=1, keepdim=True) + 1e-8).detach()
                y0_scaled = y0_gate / (scale + 1e-8)
                y_hat = net02Ip(y0_scaled)
                loss_gate = 0.5 * torch.mean((y_hat - y_true) ** 2)
                loss_gate.backward()
                g = y0_gate.grad.detach()
                sigma = args.y0_gate_sigma if args.y0_gate_sigma > 0 else g.abs().mean().item()
                w = torch.exp(- (g.abs() / (sigma + 1e-8)) ** 2)
                if args.y0_gate_mode == "hard":
                    w = (w > args.y0_gate_thresh).float()
                y0_anchor = w * y0_anchor
            # Optional net02p uncertainty weights via MC noise; if already computed, keep, else None
            if not args.gpmt_use_uncertainty:
                y0_unc_weights = None
            # Optional net02Ip weights for teacher residual (confidence)
            if args.gpmt_use_ip_weights and net02Ip is not None:
                y0_for_ip = y0_anchor.detach().clone().requires_grad_(True)
                scale = torch.sqrt(torch.mean(y0_for_ip * y0_for_ip, dim=1, keepdim=True) + 1e-8).detach()
                y0_scaled = y0_for_ip / (scale + 1e-8)
                y_hat_ip = net02Ip(y0_scaled)
                loss_ip = 0.5 * torch.mean((y_hat_ip - y_true) ** 2)
                loss_ip.backward()
                g_ip = y0_for_ip.grad.detach().abs()
                s_ip = g_ip.median() + 1e-8
                w_ip = torch.exp(- (g_ip / s_ip) ** 2)
                y0_ip_weights = w_ip
            else:
                y0_ip_weights = None
            # Adaptive per-image y0 shrink: compute diagonal alpha so that alpha*Bx0 ~ y0_anchor at init
            if args.use_y0_adapt:
                Bx0 = torch.matmul(x0.reshape(1, -1), B_norm.t())
                eps = 1e-8
                alpha = (Bx0 * y0_anchor) / (Bx0 * Bx0 + eps)
                y0_adapt_target = alpha * Bx0  # per-component shrink
            else:
                y0_adapt_target = None
        else:
            y0_anchor = None
            gamma_local = args.gamma
            y0_adapt_target = None
            y0_ip_weights = None
            y0_unc_weights = None

        # Pre-calculate teacher prediction if needed
        if args.use_teacher and teacher_net is not None:
            with torch.no_grad():
                if args.teacher_type == "unet":
                    if teacher_proj:
                        y0_pred = teacher_net(y_true)
                    else:
                        Bsz = y_true.shape[0]
                        z_img = torch.zeros((Bsz, 3, 64, 64), device=device, dtype=y_true.dtype)
                        pad_len = 64 * 64
                        if y_true.shape[1] < pad_len:
                            padded = torch.zeros((Bsz, pad_len), device=device, dtype=y_true.dtype)
                            padded[:, : y_true.shape[1]] = y_true
                        else:
                            padded = y_true[:, :pad_len]
                        z_img[:, 0] = padded.reshape(Bsz, 64, 64)
                        y0_pred = teacher_net(z_img).reshape(Bsz, -1)
                        y0_pred = y0_pred[:, :B.shape[0]]
                elif args.teacher_type == "netbonline":
                    # Predict all Bx using netBonline given y_true
                    block = max(1, args.netbonline_block)
                    preds = []
                    B_rows = B
                    for start in range(0, B_rows.shape[0], block):
                        end = min(B_rows.shape[0], start + block)
                        b_block = B_rows[start:end]  # [k, n]
                        k = b_block.shape[0]
                        y_rep = y_true.expand(k, -1)
                        b_rep = b_block
                        idx = torch.arange(start, end, device=device)
                        pred_block = teacher_net(y_rep, b_rep, idx).view(1, k)
                        preds.append(pred_block)
                    y0_pred = torch.cat(preds, dim=1)
                else:
                    y0_pred = None
        else:
            y0_pred = None

        # PSG loop with optional denoiser bank and GPMT teacher grad
        x = x0.clone()
        if args.use_denoiser_bank:
            block = max(1, math.ceil(args.iters / len(deno_bank)))
        for k in range(args.iters):
            r = Ax(x.reshape(1, -1), A) - y_true
            g_data = ATz(r, A).reshape_as(x)
            g = g_data
            # Adaptive y0 regularizer (diagonal shrink) if available
            if args.use_y0_adapt and y0_adapt_target is not None and B is not None:
                Bx_curr = torch.matmul(x.reshape(1, -1), B.t())
                g_adapt = torch.matmul(Bx_curr - y0_adapt_target, B).reshape_as(x)
                # normalize to data grad
                g_adapt = normalize_to_data(g_adapt, g_data)
                g = g + args.y0_adapt_weight * g_adapt
            # Teacher gradient from net02p anchor
            if args.use_gpmt and B_norm is not None and y0_anchor is not None:
                Bx = torch.matmul(x.reshape(1, -1), B_norm.t())
                # row subsampling schedule: decay keep fraction from 1.0 to gpmt-frac-final over iterations
                frac_keep = max(args.gpmt_frac_final, 1.0 - (1.0 - args.gpmt_frac_final) * (k / max(1, args.iters)))
                if frac_keep < 1.0:
                    keep_rows = max(1, int(frac_keep * B_norm.shape[0]))
                    vals = torch.abs(y0_anchor)
                    _, idx = torch.topk(vals, keep_rows, dim=1)
                    mask_rows = torch.zeros_like(Bx)
                    mask_rows.scatter_(1, idx, 1.0)
                else:
                    mask_rows = torch.ones_like(Bx)
                # weight rows: all ones unless unmeasured-only requested
                if args.gpmt_w_unmeasured:
                    w_rows = mask_rows
                else:
                    w_rows = mask_rows
                residual = Bx - y0_anchor
                if args.gpmt_use_uncertainty and y0_unc_weights is not None:
                    residual = residual * y0_unc_weights
                if args.gpmt_use_ip_weights and y0_ip_weights is not None:
                    residual = residual * y0_ip_weights
                g_teacher = torch.matmul(residual * w_rows, B_norm).reshape_as(x)
                # normalize teacher grad to match data grad RMS
                g_teacher = normalize_to_data(g_teacher, g_data)
                beta_k = args.gpmt_beta_floor + (args.gpmt_beta0 - args.gpmt_beta_floor) * math.exp(-k / max(1e-6, args.gpmt_beta_decay))
                g = g + beta_k * g_teacher
            # Add optional classic teacher gradient
            if args.use_teacher and B is not None and y0_pred is not None:
                Bx_raw = torch.matmul(x.reshape(1, -1), B.t())
                g_teacher_raw = torch.matmul(Bx_raw - y0_pred, B).reshape_as(x)
                if args.normalize_teacher_grad:
                    g_teacher_raw = normalize_to_data(g_teacher_raw, g_data)
                g = g + gamma_local * g_teacher_raw
            # Denoise step
            if args.use_denoiser_bank:
                idx = min(k // block, len(deno_bank) - 1)
                x = deno_bank[idx](x - eta * g).clamp(0.0, 1.0)
            else:
                x = denoise_fn(x - eta * g).clamp(0.0, 1.0)

        x_rec = x
        score = psnr(x_rec, x_true)
        scores.append(score)
        extra = f"bank" if args.use_denoiser_bank else f"sigma={args.sigma}"
        if args.use_teacher:
            extra += f", teacher(g={args.gamma})"
        print(f"{p.name}: {score:.2f} dB (eta={eta:.4e}, {extra})")
        if out_dir is not None:
            arr = (x_rec.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / p.name)

    mean_psnr = sum(scores) / len(scores)
    print(f"Average PSNR over {len(scores)} images: {mean_psnr:.2f} dB ({deno_desc})")


if __name__ == "__main__":
    main()
