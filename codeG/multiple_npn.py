import deepinv as dinv
import torch    
import numpy as np
import matplotlib.pyplot as plt
from utils import * 
from algorithms import MyPGD, NPN_PGD, MultipleSNPN_PGD#, MultipleS_RW_NPN_PGD
import argparse
from deepinv.utils import plot
from models import *
# Configs


def main(args):
    # Validate arguments
    if not isinstance(args.cr_s, list):
        args.cr_s = [args.cr_s]  # Convert single value to list
    
    # Ensure compression ratios are in descending order for algorithm logic
    args.cr_s = sorted(args.cr_s, reverse=True)
    
    cr_h = args.cr_h
    n = args.n
    cr_s = args.cr_s  # reverse to have decreasing order

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    set_seed(seed=0)

    m_h = int(cr_h*n*n)

    # Fix: Remove incorrect normalization that breaks compressed sensing theory
    fixed_matrix = generate_matrix(m_h, n*n).to(device)
    
    spc_h = dinv.physics.CompressedSensing(m=m_h,img_size=(3,n,n),device=device,noise_model=dinv.physics.GaussianNoise(sigma=args.sigma),channelwise=True,fast=False)

    _A_dagger = torch.linalg.pinv(fixed_matrix)
    spc_h.register_buffer("_A", fixed_matrix)
    spc_h.register_buffer("_A_dagger", _A_dagger)
    spc_h.register_buffer("_A_adjoint", spc_h._A.conj().T.type(spc_h.dtype).to(device))





    exp_names =[ f'v2_spc_pre_train_crh{cr_h}_crs{cr_si}_sigma_x_{args.sigma_x}_epochs{args.num_epochs}' for cr_si in cr_s]
    
    checkpoints = [torch.load(f'results/{exp_name}/best_model.pth', map_location=device) for exp_name in exp_names]


    backbone = UNet(n_channels=3, base_channel=args.base_channel)


    fixed_matrix_ss = [checkpoint['matrix_s'].to(device) for checkpoint in checkpoints]

    spc_ss = []
    for i, fixed_matrix_s in enumerate(fixed_matrix_ss):
        # Fix: Use correct measurement size for each sensor
        m_s = fixed_matrix_s.shape[0]  # Get actual measurement size from matrix
        spc_s = dinv.physics.CompressedSensing(m=m_s,img_size=(3,n,n),device=device,channelwise=True,fast=False)
        _A_dagger = torch.linalg.pinv(fixed_matrix_s)
        spc_s.register_buffer("_A", fixed_matrix_s)
        spc_s.register_buffer("_A_dagger", _A_dagger)
        spc_s.register_buffer("_A_adjoint", spc_s._A.conj().T.type(spc_s.dtype).to(device))
        spc_ss.append(spc_s)
    
    models = []
    for i,checkpoint in enumerate(checkpoints):
        spc_s = spc_ss[i]
        model = GPredictor(backbone=backbone, physics_s=spc_s, device=device)
        model.to(device)
        model.load_state_dict(checkpoint['G_state_dict'])
        models.append(model)



    testloader = get_dataloaders_testing(args)

    x = next(iter(testloader)).to(device)


    y = spc_h(x)

    y_s = [modeli(spc_h.A_dagger(y)) for modeli in models]

    y_s_gt = [spc_s.A(x) for spc_s in spc_ss]

    # Verify closeness of predicted and true measurements
    for i, (y_s_i, y_s_gt_i) in enumerate(zip(y_s, y_s_gt)):
        print(f"Sensor {i}: {dinv.metric.PSNR()(y_s_i, y_s_gt_i).mean().item()}")

    data_fidelity = dinv.optim.L2()
    denoiser = dinv.models.DnCNN(
        device=device,pretrained='download_lipschitz')
    prior = dinv.optim.PnP(denoiser=denoiser)
    
    # Fix: Use more conservative stepsize calculation
    stepsize = 0.1 / spc_h.compute_norm(spc_h.A_adjoint(y), tol=1e-3).item()

    
    gamma = [0.1/spc_s.compute_norm(spc_s.A_dagger(y_s_i), tol=1e-3).item() for spc_s,y_s_i in zip(spc_ss,y_s)]

    gamma_npn = 1.8/spc_ss[0].compute_norm(spc_ss[0].A_dagger(y_s[0]), tol=1e-3).item()
    lambd = 0.005
    # Fix: Reduce iterations for better performance and avoid overfitting
    max_iter = 3000



    model = MyPGD(
        data_fidelity=data_fidelity,
        prior=prior,
        stepsize=stepsize,
        lambd=lambd,
        max_iter=max_iter,
    )

    model_npn = NPN_PGD(
        data_fidelity=data_fidelity,
        prior=prior,
        stepsize=stepsize,
        lambd=lambd,
        max_iter=max_iter,
        gamma=gamma_npn
    )

    model_multiple_s = MultipleSNPN_PGD(
        data_fidelity=data_fidelity,
        prior=prior,
        stepsize=stepsize,
        lambd=lambd,
        max_iter=max_iter,
        gamma=gamma,
        eps=1e-8

    )


    x0 = spc_h.A_dagger(y)

    x0_npn = spc_h.A_dagger(y)
    print("Running Multi-Sensor NPN-PGD...")
    print(y_s[-1].shape, y_s[0].shape)
    x_hat_npn_multi,x_hats_npn_multi = model_multiple_s(x0_npn, y, y_s, spc_h, spc_ss, type_select='seq')
    print("Running Multi-Sensor NPN-PGD with concatenated gradients...")
    x_hat_npn_multi_concat,x_hats_npn_multi_concat = model_multiple_s(x0_npn, y, y_s, spc_h, spc_ss, type_select='concat')
    print("Running Multi-Sensor NPN-PGD with random gradients...")
    x_hat_npn_multi_rand,x_hats_npn_multi_rand = model_multiple_s(x0_npn, y, y_s, spc_h, spc_ss, type_select='rand')

    print("Running Base PGD...")
    x_hat,x_hats = model(x0, y, spc_h)
    print("Running NPN-PGD...")
    x_hat_npn,x_hats_npn = model_npn(x0_npn, y, y_s[0], spc_h, spc_ss[0])

    psnr = dinv.metric.PSNR()

    psnrs = [psnr(x_k, x).mean().item() for x_k in x_hats]

    psnrs_npn = [psnr(x_k, x).mean().item() for x_k in x_hats_npn]

    psnrs_npn_multi = [psnr(x_k, x).mean().item() for x_k in x_hats_npn_multi]

    psnrs_npn_multi_concat = [psnr(x_k, x).mean().item() for x_k in x_hats_npn_multi_concat]

    psnrs_npn_multi_rand = [psnr(x_k, x).mean().item() for x_k in x_hats_npn_multi_rand]

    print("Final PSNRs:")
    print("PGD: ", psnrs[-1])
    print("NPN-PGD: ", psnrs_npn[-1])
    print("Multi-Sensor NPN-PGD: ", psnrs_npn_multi[-1])
    print("Multi-Sensor NPN-PGD with concatenated gradients: ", psnrs_npn_multi_concat[-1])
    print("Multi-Sensor NPN-PGD with random gradients: ", psnrs_npn_multi_rand[-1])

    plt.figure()
    plt.plot(psnrs)
    plt.plot(psnrs_npn)
    plt.plot(psnrs_npn_multi)
    plt.plot(psnrs_npn_multi_concat)
    plt.plot(psnrs_npn_multi_rand)
    plt.legend(['PnP-PGD','NPN-PGD','Scheduling NPN-PGD','All regs NPN-PGD (concat)','Random reg NPN-PGD'])
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Iteration")
    plt.grid()
    
    plt.text(0.55, 0.1, f'cr_h={cr_h}, cr_s={cr_s}', transform=plt.gca().transAxes,
             fontsize=12, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.savefig(f'figures_results/spc_npn_pgd_crh{cr_h}_crs{cr_s}_sigma{args.sigma}.png')
    plt.show()

    
    




    idx_imgs = [0,max_iter//4,max_iter//2,3*max_iter//4,max_iter-1]
    # plot_reconstructions(x, x_hat, x_hats, idx_imgs, psnrs, title="SPC Reconstruction using NPN-PGD",save_path=f'results/spc_npn_pgd_cr{cr_h}_sigma{args.sigma}.png')


parser = argparse.ArgumentParser(description="Single Pixel Camera Reconstruction")

# Physics parameters 
parser.add_argument('--cr_h', type=float, default=0.1, help='Compression ratio for horizontal SPC')
parser.add_argument('--cr_s', type=float, nargs='+', default=[0.6,0.55,0.5,0.45,0.4,0.3,0.2,0.1,0.05], help='Compression ratio for single pixel cameras (list)')
parser.add_argument('--sigma', type=float, default=0, help='Noise level for Gaussian noise model')

# Data parameters

parser.add_argument('--n', type=int, default=128, help='Image size (n x n)')
parser.add_argument('--dataset', type=str, default='celeba', help='Dataset to use (mnist, fashionmnist, cifar10, BSDS500, CelebA, ct)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loaders')
parser.add_argument('--grayscale', type=bool, default=False, help='Convert images to grayscale')


# Model parameters
parser.add_argument('--base_channel', type=int, default=32, help='Base channel for UNet backbone')
parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--sigma_x', type=float, default=0.1, help='Noise level for data augmentation')

args = parser.parse_args()


main(args)
