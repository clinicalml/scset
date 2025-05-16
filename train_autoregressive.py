import torch
import torch.nn as nn
import scanpy as sc
import os
import time
import math
import json

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from utils import init_rnd_seeds, EarlyStopping


def train_autoregressive_diff(args, train_dataset, test_dataset, encoder, denoise_model, diffusion_process, checkpoint_dir, writer=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    encoder.to(device=device)
    denoise_model.to(device=device)
    diffusion_process.to(device=device)

    parameters = [
        {'params': encoder.parameters(), 'lr': args.pretrain_lr, 'weight_decay': args.pretrain_weight_decay},
        {'params': denoise_model.parameters(), 'lr': args.pretrain_lr, 'weight_decay': args.pretrain_weight_decay}
    ]
    
    optimizer = AdamW(params=parameters)
    
    print("Building train DataLoader")
    train_dl = DataLoader(train_dataset, batch_size=args.pretrain_batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers, worker_init_fn=init_rnd_seeds)
    print("Building test DataLoader")
    test_dl = DataLoader(test_dataset, batch_size=args.pretrain_batch_size, shuffle=False, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers, worker_init_fn=init_rnd_seeds)

    #initialize early stopping
    if args.earlystop_patience_pt > 0:
        early_stopping = EarlyStopping(patience=args.earlystop_patience_pt, min_delta=args.earlystop_mindelta)

    for epoch in range(args.pretrain_epochs):
        print(f"Epoch {epoch}")
        acc = train_epoch(args, encoder, denoise_model, diffusion_process, optimizer, train_dl, epoch, device, writer=writer)
        test_loss = evaluate(args, encoder, denoise_model, diffusion_process, test_dl, epoch, device, writer=writer)

        # save pretraining checkpoint every 10 epochs
        if epoch % 10 == 0:
            model_weights = {
                'encoder': encoder.cpu().state_dict(),
                'denoise': denoise_model.cpu().state_dict()
            }
            print("Saving checkpoint...")
            torch.save(model_weights, checkpoint_dir / f'pretrained.pt')
            encoder.to(device=device)
            denoise_model.to(device=device)
        

        # Check for early stopping
        if args.earlystop_patience_pt > 0:
            early_stopping(test_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered at epoch:", epoch)
                break

def train_epoch(args, encoder, denoise_model, diff_process, optimizer, dl, epoch, device, writer=None):

    avg_loss = 0
    max_norm_encoder = 0
    max_norm_denoise = 0
    sum_norm_encoder = 0
    sum_norm_denoise = 0

    for iteration, batch in enumerate(dl):
        if args.debug:
            print(f"Iteration {iteration}")
        step = epoch*len(dl) + iteration

        anchor_cells = batch['anchor']
        anchor_mask = batch['anchor_mask']
        target_cells = batch['target']

        #check for a nan in the data
        if torch.isnan(anchor_cells).any():
            print("NaN detected in anchor cells!")
        if torch.isnan(anchor_mask).any():
            print("NaN detected in anchor mask!")
        if torch.isnan(target_cells).any():
            print("NaN detected in target cells!")

        anchor_cells, anchor_mask, target_cells = anchor_cells.to(device=device), anchor_mask.to(device=device), target_cells.to(device=device)

        patient_emb = encoder(anchor_cells, X_mask=anchor_mask) # (batch_size, model_dim)
        patient_emb = patient_emb.unsqueeze(1).expand(-1, args.num_target_cells*args.num_steps_per_sample, -1) # (batch_size, num_target_cells, model_dim)
        patient_emb = patient_emb.reshape(-1, patient_emb.shape[-1]) # (batch_size*num_target_cells*num_steps_per_sample, model_dim)
        
        target_cells = target_cells.unsqueeze(2).expand(-1, -1, args.num_steps_per_sample, -1) # (batch_size, num_target_cells*num_steps_per_sample, input_dim)
        target_cells = target_cells.reshape(-1, target_cells.shape[-1])

        tidx = torch.randint(0, args.num_timesteps, (target_cells.shape[0],), dtype=torch.long).to(device=device)

        loss = diff_process.p_loss(denoise_model, target_cells, tidx, condition=patient_emb, loss_fn='l2')
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.pretrain_clip_grad)
        torch.nn.utils.clip_grad_norm_(denoise_model.parameters(), args.pretrain_clip_grad)

        norm_sq_encoder = 0
        for param in encoder.parameters():
            if param.grad is not None:
                norm_sq_encoder += torch.norm(param.grad).item()**2
        norm_encoder = math.sqrt(norm_sq_encoder)
        if args.debug:
            print(f"Norm encoder: {norm_encoder}")

        sum_norm_encoder += norm_encoder
        max_norm_encoder = max(max_norm_encoder, norm_encoder)
        if args.debug:
            print(f"Max norm encoder: {max_norm_encoder}")
        
        norm_sq_denoise = 0
        for param in denoise_model.parameters():
            if param.grad is not None:
                norm_sq_denoise += torch.norm(param.grad).item()**2
        norm_denoise = math.sqrt(norm_sq_denoise)

        sum_norm_denoise += norm_denoise
        max_norm_denoise = max(max_norm_denoise, norm_denoise)

        if writer is not None:
            writer.add_scalar('Pretraining/Train Loss (step)', loss.item(), step)
            writer.add_scalar('Grad Norm/Encoder (step)', norm_encoder, step)
            writer.add_scalar('Grad Norm/Denoise (step)', norm_denoise, step)

        optimizer.step()
        optimizer.zero_grad()

        avg_loss += loss.item()

    avg_loss /= len(dl)
    print(f"Epoch {epoch}: Train Loss: {avg_loss}, Encoder Max Grad Norm: {max_norm_encoder}, Denoise Max Grad Norm: {max_norm_denoise}")

    if writer is not None:
        writer.add_scalar('Pretraining/Train Loss (epoch)', avg_loss, epoch)

        writer.add_scalar('Grad Norm/Encoder (epoch max)', max_norm_encoder, epoch)
        writer.add_scalar('Grad Norm/Denoise (epoch max)', max_norm_denoise, epoch)

        writer.add_scalar('Grad Norm/Encoder (epoch avg)', sum_norm_encoder/len(dl), epoch)
        writer.add_scalar('Grad Norm/Denoise (epoch avg)', sum_norm_denoise/len(dl), epoch)

        writer.flush()

    return avg_loss


def evaluate(args, encoder, denoise_model, diff_process, dl, epoch, device, writer=None):

    avg_loss = 0

    for iteration, batch in enumerate(dl): 
        step = epoch*len(dl) + iteration

        anchor_cells = batch['anchor']
        anchor_mask = batch['anchor_mask']
        target_cells = batch['target']

        anchor_cells, anchor_mask, target_cells = anchor_cells.to(device=device), anchor_mask.to(device=device), target_cells.to(device=device)

        patient_emb = encoder(anchor_cells, X_mask=anchor_mask) # (batch_size, model_dim)
        patient_emb = patient_emb.unsqueeze(1).expand(-1, args.num_target_cells*args.num_steps_per_sample, -1) # (batch_size, num_target_cells, model_dim)
        patient_emb = patient_emb.reshape(-1, patient_emb.shape[-1]) # (batch_size*num_target_cells*num_steps_per_sample, model_dim)

        target_cells = target_cells.unsqueeze(2).expand(-1, -1, args.num_steps_per_sample, -1) # (batch_size, num_target_cells*num_steps_per_sample, input_dim)
        target_cells = target_cells.reshape(-1, target_cells.shape[-1])

        tidx = torch.randint(0, args.num_timesteps, (target_cells.shape[0],), dtype=torch.long).to(device=device)

        with torch.no_grad():
            loss = diff_process.p_loss(denoise_model, target_cells, tidx, condition=patient_emb, loss_fn='l2')

        if writer is not None:
            writer.add_scalar('Pretraining/Test Loss (step)', loss.item(), step)

        avg_loss += loss.item()

    avg_loss /= len(dl)
    print(f"Epoch {epoch}: Test Loss: {avg_loss}")

    if writer is not None:
        writer.add_scalar('Pretraining/Test Loss (epoch)', avg_loss, epoch)
        writer.flush()
    
    return avg_loss


def save_config(args, name):
    """
    Saves the config file for a given checkpoint name as a json dictionnary.
    """
    args_dict = vars(args)
    if not os.path.exists("./configs/autoregressive_diffusion/"): os.makedirs("./configs/autoregressive_diffusion/")
    
    with open(os.path.join("./configs/autoregressive_diffusion/", f"{name}.json"), "w") as f:
        json.dump(args_dict, f, indent=4)