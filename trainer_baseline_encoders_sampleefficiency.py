import os
import random
from pathlib import Path
import scanpy as sc
import gc
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import pandas as pd

from args import get_args
from utils import AverageValueMeter, set_random_seed, save, resume, get_scheduler, write_params, generate_run_name, save_config, write_results
from torch.utils.data import Dataset
from datasets import get_datasets
from datasets.RNASeq import CellBagDataset

from train_supervised import train_supervised
from train_autoregressive import train_autoregressive_diff
from train_contrastive import train_contrastive
from train_cellsorter import train_cellsorter
from train_clustermask import train_clustermask
from modules.encoder import build_encoder_model
from datasets.cxg_datasets import AutoregPatientDataset, ContrastivePatientDataset, CellSorterPatientDataset, PatientDataset
from modules.denoising_models import ConditionalDenosingMLP, build_denoising_model
from diffusion.diffusion_utils import build_diffusion_process
from baseline_modules import run_linear_probe_baselines, run_finetuning_baselines
from datetime import datetime

import atexit
import signal

import sys
import logging
import traceback

from modules.mlp import build_classifier_head, build_projection_head, build_sorter_head

def main():
    print("reading args")
    args = get_args()
    args.pretrain = "none"

    if args.debug:
        print("**RUNNING IN DEBUG MODE**")
        args.epochs = 2
        args.pretrain_epochs = 2
        args.val_interval = 1
        args.test_interval = 1
        args.log_interval = 1
        args.save_interval = 1
        args.run_kfold = False
        args.run_name = 'debug_run'
        args.seed = 0
        args.tag = 'debug'

        args.h5ad_loc = "/data/rna_rep_learning/smalladata_fordebugging_hlca_sikkema2023.h5ad"
        args.target_col = "disease"
        args.pid_col = "sample"
        args.adata_layer="scvi_cxgcensus_20240701"
    else:
        args.run_name = f'{generate_run_name()}-baselines-{args.tag}'
        print('Run name: ', args.run_name)
    
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.cur_time = cur_time
    
    print("set save dir and random seed")
    work_dir = Path(args.work_dir) #defaults to current directory
    log_dir = Path(args.log_dir)
    tensorboard_dir = Path(work_dir) / f"tensorboard_logs/{args.run_name}"
    checkpoint_dir = Path(log_dir) / f"checkpoints/{args.run_name}"
    results_dir = Path(log_dir) / f"results"
    params_dir = Path(log_dir) / f"params"
    exception_log_dir = Path(log_dir) / f"exceptions"
    
    if not Path(checkpoint_dir).exists(): Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    if not Path(tensorboard_dir).exists(): Path(tensorboard_dir).mkdir(exist_ok=True, parents=True)
    if not Path(results_dir).exists(): Path(results_dir).mkdir(exist_ok=True, parents=True)
    if not Path(params_dir).exists(): Path(params_dir).mkdir(exist_ok=True, parents=True)
    if not Path(exception_log_dir).exists(): Path(exception_log_dir).mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger()
    fh = logging.FileHandler(exception_log_dir /  f'{args.run_name}.log')
    logger.addHandler(fh)
 
    def log_exceptions(type, value, tb):
        for line in traceback.TracebackException(type, value, tb).format(chain=True):
           logging.exception(line)
        logging.exception(value)
        sys.__excepthook__(type, value, tb) # calls default excepthook
    sys.excepthook = log_exceptions

    def handle_exit():
        logging.exception('Exiting due to signal')  

    atexit.register(handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)

    print(f"checkpoint_dir: {checkpoint_dir}")
    print("Writing params...")
    save_config(args, params_dir)
    print("Done.")

    # Writer
    writer = SummaryWriter(tensorboard_dir)
    #cudnn.benchmark = True
    device = torch.device('cuda')
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    print("Reading adata...")
    adata = sc.read_h5ad(args.h5ad_loc)
    print("Done.")

    # if not args.skip_pseudobulk_baseline:
    #     print("Running pseudobulk baseline")

    celltypefracs_df = None
    if not args.skip_celltypefracs_baseline:
        print("Running cell type proportions baseline")
        args.encoder='celltypefracs' #to be used when writing results

        # calculate embeddings
        celltypefracs_df = pd.DataFrame(adata.obs.groupby(args.pid_col)[args.celltype_col].value_counts()).rename({'count':'cell_type_counts'},axis=1).reset_index().pivot(index=args.pid_col, columns=args.celltype_col, values="cell_type_counts")
        celltypefracs_df = celltypefracs_df.div(celltypefracs_df.sum(axis=1), axis=0)
        celltypefracs_df = celltypefracs_df.fillna(0)

        # get patient order, targets, and folds
        pt_adata = adata.obs[[args.pid_col, args.target_col, args.fold_col]].drop_duplicates()
        target_classes, targets = np.unique(pt_adata[args.target_col], return_inverse=True) 
        folds = pt_adata[args.fold_col].values

        # get patient embeddings in same patient order
        embs = celltypefracs_df.loc[pt_adata[args.pid_col]].values 

        # run linear probe
        test_mean_acc, test_std_acc, test_mean_auc, test_std_auc, test_mean_f1, test_std_f1 = run_linear_probe_baselines(args, embs, targets, folds, results_dir=results_dir, subsample_ns=[25, 50, 100], nreps=5)
        print(f'10-Fold Test Accuracy of Linear Probe: {test_mean_acc} +- {test_std_acc}')
        print(f'10-Fold Test AUC of Linear Probe: {test_mean_auc} +- {test_std_auc}')
        print(f'10-Fold Test Weighted-F1 of Linear Probe: {test_mean_f1} +- {test_std_f1}')

        # run MLP - for sample efficiency experiments, only run linear probe
        #run_finetuning_baselines(args, embs, targets, folds, results_dir, writer, checkpoint_dir, device)

    celltype_means_df = None
    if not args.skip_celltypemeans_baseline:
        print("Running mean embedding per cell type baseline")
        args.encoder='celltypemeans'

        # get mean embedding per sample+cell_type
        if args.latent_idx_tokeep is None:
            celltype_means_df = pd.DataFrame(adata.obsm[args.adata_layer], index=adata.obs.index).merge(adata.obs[[args.pid_col,args.celltype_col]], right_index=True, left_index=True).groupby([args.pid_col,args.celltype_col]).mean()
        else:
            celltype_means_df = pd.DataFrame(adata.obsm[args.adata_layer][:,args.latent_idx_tokeep], index=adata.obs.index).merge(adata.obs[[args.pid_col,args.celltype_col]], right_index=True, left_index=True).groupby([args.pid_col,args.celltype_col]).mean()
        celltype_means_df.columns = [str(col) for col in celltype_means_df.columns] # convert to strings
        celltype_means_df = celltype_means_df.fillna(0).reset_index() # cell types that dont exist in a given sample - fill 0 for mean expression
        celltype_means_df = celltype_means_df.pivot(index=args.pid_col, columns=args.celltype_col) # Pivot the dataframe
        celltype_means_df.columns = ['_'.join(col).strip() for col in celltype_means_df.columns.values] # Flatten the multi-level columns

        # get patient order, targets, and folds
        pt_adata = adata.obs[[args.pid_col, args.target_col, args.fold_col]].drop_duplicates()
        target_classes, targets = np.unique(pt_adata[args.target_col], return_inverse=True) 
        folds = pt_adata[args.fold_col].values

        # get patient embeddings in same patient order
        embs = celltype_means_df.loc[pt_adata[args.pid_col]].values 

        # run linear probe
        test_mean_acc, test_std_acc, test_mean_auc, test_std_auc, test_mean_f1, test_std_f1 = run_linear_probe_baselines(args, embs, targets, folds, results_dir=results_dir, subsample_ns=[25, 50, 100], nreps=5)
        print(f'10-Fold Test Accuracy of Linear Probe: {test_mean_acc} +- {test_std_acc}')
        print(f'10-Fold Test AUC of Linear Probe: {test_mean_auc} +- {test_std_auc}')
        print(f'10-Fold Test Weighted-F1 of Linear Probe: {test_mean_f1} +- {test_std_f1}')

        # run MLP
        #run_finetuning_baselines(args, embs, targets, folds, results_dir, writer, checkpoint_dir, device)

    if not args.skip_catfracsmeans_baseline:
        print("Running categorical proportions and mean embedding baseline")
        args.encoder='catfracsmeans'

        if celltypefracs_df is None:
            celltypefracs_df = pd.DataFrame(adata.obs.groupby(args.pid_col)[args.celltype_col].value_counts()).rename({'count':'cell_type_counts'},axis=1).reset_index().pivot(index=args.pid_col, columns=args.celltype_col, values="cell_type_counts")
            celltypefracs_df = celltypefracs_df.div(celltypefracs_df.sum(axis=1), axis=0)
            celltypefracs_df = celltypefracs_df.fillna(0)

        if celltype_means_df is None:
            if args.latent_idx_tokeep is None:
                celltype_means_df = pd.DataFrame(adata.obsm[args.adata_layer], index=adata.obs.index).merge(adata.obs[[args.pid_col,args.celltype_col]], right_index=True, left_index=True).groupby([args.pid_col,args.celltype_col]).mean()
            else:
                celltype_means_df = pd.DataFrame(adata.obsm[args.adata_layer][:,args.latent_idx_tokeep], index=adata.obs.index).merge(adata.obs[[args.pid_col,args.celltype_col]], right_index=True, left_index=True).groupby([args.pid_col,args.celltype_col]).mean()
            celltype_means_df.columns = [str(col) for col in celltype_means_df.columns] # convert to strings
            celltype_means_df = celltype_means_df.fillna(0).reset_index() # cell types that dont exist in a given sample - fill 0 for mean expression
            celltype_means_df = celltype_means_df.pivot(index=args.pid_col, columns=args.celltype_col) # Pivot the dataframe
            celltype_means_df.columns = ['_'.join(col).strip() for col in celltype_means_df.columns.values] # Flatten the multi-level columns

        cat_fracsmeans_df = celltype_means_df.merge(celltype_means_df, right_index=True, left_index=True)

        # get patient order, targets, and folds
        pt_adata = adata.obs[[args.pid_col, args.target_col, args.fold_col]].drop_duplicates()
        target_classes, targets = np.unique(pt_adata[args.target_col], return_inverse=True) 
        folds = pt_adata[args.fold_col].values

        # get patient embeddings in same patient order
        embs = cat_fracsmeans_df.loc[pt_adata[args.pid_col]].values 

        # run linear probe
        test_mean_acc, test_std_acc, test_mean_auc, test_std_auc, test_mean_f1, test_std_f1 = run_linear_probe_baselines(args, embs, targets, folds, results_dir=results_dir, subsample_ns=[25, 50, 100], nreps=5)
        print(f'10-Fold Test Accuracy of Linear Probe: {test_mean_acc} +- {test_std_acc}')
        print(f'10-Fold Test AUC of Linear Probe: {test_mean_auc} +- {test_std_auc}')
        print(f'10-Fold Test Weighted-F1 of Linear Probe: {test_mean_f1} +- {test_std_f1}')

        # run MLP
        #run_finetuning_baselines(args, embs, targets, folds, results_dir, writer, checkpoint_dir, device)
        

    # if not args.skip_kmeans_baseline:
    #     print("Running kmeans baseline")



if __name__ == '__main__':
    main()
