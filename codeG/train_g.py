import deepinv as dinv
import torch    
import numpy as np
import matplotlib.pyplot as plt
from utils import * 
from algorithms import MyPGD
from tqdm import tqdm
import argparse
from models import *
import wandb 



def main(args):
    # Initialize wandb
    
    
    # Configs
    cr_h = args.cr_h
    cr_s = args.cr_s
    n = args.n
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed=0)

    trainloader, testloader = get_dataloaders(args)

    fixed_matrix = generate_matrix(int(cr_h*n*n), n*n)
    physics_h = dinv.physics.CompressedSensing(
        m=int(cr_h*n*n),
        img_size=(3,n,n),
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=args.sigma),
        fast=False, channelwise=True
    )
    
    _A_dagger = torch.linalg.pinv(fixed_matrix)
    physics_h.register_buffer("_A", fixed_matrix)
    physics_h.register_buffer("_A_dagger", _A_dagger)
    physics_h.register_buffer("_A_adjoint", physics_h._A.conj().T.type(physics_h.dtype).to(device))

    fixed_matrix_s = nn.Parameter(generate_orthogonal_rows_qr(fixed_matrix, int(cr_s*n*n), device=device),requires_grad=False) 


    physics_s = dinv.physics.CompressedSensing(
        m=int(cr_s*n*n),
        img_size=(3,n,n),
        device=device,
        fast=False,channelwise=True)

    _A_dagger = torch.linalg.pinv(fixed_matrix_s)
    physics_s.register_buffer("_A", fixed_matrix_s)
    physics_s.register_buffer("_A_dagger", _A_dagger)
    physics_s.register_buffer("_A_adjoint", physics_s._A.conj().T.type(physics_s.dtype).to(device))

    psnr = dinv.metric.PSNR()

    backbone = UNet(n_channels=3, base_channel=args.base_channel)

    model = GPredictor(backbone=backbone, physics_s=physics_s, device=device)
    model.to(device)

    criterion = torch.nn.MSELoss()
    print("number of parameters in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(physics_s.parameters()), lr=args.learning_rate)
    best_psnr = -np.inf

    exp_name = f'v2_spc_pre_train_crh{cr_h}_crs{cr_s}_sigma_x_{args.sigma_x}_epochs{args.num_epochs}'
    
    path = f'results/{exp_name}'
    if not os.path.exists(path):
        os.makedirs(path)

    

    wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=exp_name
        )


    for epoch in range(args.num_epochs):
        model.train()
        data_loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=True)
        train_loss_predictor = AverageMeter()
        train_psnr = AverageMeter()

        for batch in data_loop:
            x = batch.to(device)

            optimizer.zero_grad()

            y = physics_h(x)
            ys = physics_s.A(x)


            ys_hat = model(physics_h.A_dagger(y))

            loss_predictor = criterion(ys_hat, ys)
            

            loss = loss_predictor
            
            train_loss_predictor.update(loss_predictor.item(), x.size(0))
        
            train_psnr.update(psnr(ys_hat, ys).mean().item(), x.size(0))

            loss.backward()
            optimizer.step()

            data_loop.set_postfix(loss=loss_predictor.item(), psnr=train_psnr.avg)

        # Log training metrics to wandb
        log_dict = {
            "epoch": epoch,
            "train/loss_predictor": train_loss_predictor.avg,
            "train/psnr": train_psnr.avg
        }

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = AverageMeter()
            val_psnr = AverageMeter()

            data_loop = tqdm(testloader, desc=f"Validation Epoch {epoch+1}/{args.num_epochs}", leave=True, colour='green')
            for batch in data_loop:
                x = batch.to(device)
                y = physics_h(x)
                ys = physics_s.A(x)
                ys_hat = model(physics_h.A_dagger(y))
        
                val_loss.update(criterion(ys_hat, ys).item(), x.size(0))
                val_psnr.update(psnr(ys_hat, ys).mean().item(), x.size(0))
                data_loop.set_postfix(val_loss=val_loss.avg, val_psnr=val_psnr.avg)

        # Log validation metrics to wandb
        log_dict.update({
            "val/loss": val_loss.avg,
            "val/psnr": val_psnr.avg
        })
        wandb.log(log_dict)

        if train_psnr.avg > best_psnr:
            best_psnr = train_psnr.avg
            torch.save({'G_state_dict': model.state_dict(),
                        'matrix_s': fixed_matrix_s}, 
                       f'{path}/best_model.pth')
            print(f"New best PSNR: {best_psnr:.2f} dB, model saved.")
            # Log best model info to wandb
            wandb.log({"best_psnr": best_psnr})

    # Finish wandb run
    wandb.finish()    



# put config parameters argparse

parser = argparse.ArgumentParser(description="Single Pixel Camera Reconstruction")

# Physics parameters 
parser.add_argument('--cr_h', type=float, default=0.1, help='Compression ratio for horizontal SPC')
parser.add_argument('--cr_s', type=float, default=0.1, help='Compression ratio for single pixel cameras')
parser.add_argument('--sigma', type=float, default=0.01, help='Noise level for Gaussian noise model')

# Data parameters

parser.add_argument('--n', type=int, default=128, help='Image size (n x n)')
parser.add_argument('--dataset', type=str, default='celeba', help='Dataset to use (mnist, fashionmnist, cifar10, BSDS500, CelebA, ct)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders')
parser.add_argument('--grayscale', type=bool, default=False, help='Convert images to grayscale')

# Model parameters for U-Net
parser.add_argument('--base_channel', type=int, default=32, help='Initial number of features for U-Net')

# Training parameters
parser.add_argument('--num_s', type=int, default=5, help='Number of single pixel cameras')
parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
parser.add_argument('--lambda_predictor', type=float, default=0.0, help='Weight for predictor loss')
parser.add_argument('--lambda_denoise', type=float, default=0.0, help='Weight for denoiser loss')
parser.add_argument('--lambda_orthogonal', type=float, default=0.1, help='Weight for orthogonal loss')
# Miscellaneous
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cpu or cuda)')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--sigma_x', type=float, default=0.1, help='Noise level for data augmentation')

# Wandb parameters
parser.add_argument('--wandb_project', type=str, default='multiple-s-ns', help='Wandb project name')

parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name (auto-generated if None)')


args = parser.parse_args()


cr_ss = [0.6,0.55,0.5,0.45]

for cr_s in cr_ss:
    print('Running for cr_s = ', cr_s)
    args.cr_s = cr_s
    main(args)






