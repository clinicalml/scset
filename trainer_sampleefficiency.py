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

from args import get_args
from utils import AverageValueMeter, set_random_seed, save, resume, get_scheduler, write_params, generate_run_name, save_config, write_results, write_results_sampleefficiency
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
from linear_probe import run_linear_probe
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
        args.run_name = f'{generate_run_name()}-{args.encoder}-{args.pretrain}-{args.tag}'
        print('Run name: ', args.run_name)
    
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.cur_time = cur_time
    
    print("set save dir and random seed")
    work_dir = Path(args.work_dir) #defaults to current directory
    log_dir = Path(args.log_dir)
    tensorboard_dir = Path(work_dir) / f"tensorboard_logs/{args.run_name}"
    checkpoint_dir = Path(log_dir) / f"checkpoints/{args.run_name}"
    dataset_cache_dir = Path(log_dir) / f"torch_datasets"
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

    if args.latent_idx_tokeep is not None:
        args.sample_dim = len(args.latent_idx_tokeep)
    encoder = build_encoder_model(args, sample_dim=args.sample_dim)

    if args.pretrained_ckpt is not None:
        print(f"Loading pretrained model from {args.pretrained_ckpt}")
        model_weights = torch.load(args.pretrained_ckpt, weights_only=True)
        encoder.load_state_dict(model_weights['encoder'])

    elif args.pretrain == 'diffusion':
        if not args.load_cached_pretrain_dataset:
            print("Creating train AutoregPatientDataset")
            autoreg_train_dataset = AutoregPatientDataset(folder=args.pretrain_data_dir, pretrain_embedding=args.pretrain_embedding, min_num_cells = args.min_num_cells, max_num_cells=args.max_num_cells,
                                                        num_anchor_cells=args.num_anchor_cells, num_target_cells=args.num_target_cells, preload=args.no_preload, split_col=args.pretrain_fold_col, 
                                                        split_ids=[0, 1, 2, 3, 4, 5, 7, 8, 9], patient_csv=args.patient_csv, debug=args.debug, annots_dir=args.annots_dir, latent_idx_tokeep=args.latent_idx_tokeep)
            print("Creating test AutoregPatientDataset")
            autoreg_test_dataset = AutoregPatientDataset(folder=args.pretrain_data_dir, pretrain_embedding=args.pretrain_embedding, min_num_cells=args.min_num_cells, max_num_cells=args.max_num_cells,
                                                        num_anchor_cells=args.num_anchor_cells, num_target_cells=args.num_target_cells, preload=args.no_preload, split_col=args.pretrain_fold_col, 
                                                        split_ids=[6], patient_csv=args.patient_csv, debug=args.debug, annots_dir=args.annots_dir, latent_idx_tokeep=args.latent_idx_tokeep)
            # save to disk
            if args.pretrain_datasets_cache is None:
                args.pretrain_datasets_cache = args.run_name
            torch.save(autoreg_train_dataset, dataset_cache_dir / f'{args.pretrain_datasets_cache}_train.pt')
            torch.save(autoreg_test_dataset, dataset_cache_dir / f'{args.pretrain_datasets_cache}_test.pt')
        else:
            print("Loading cached AutoregPatientDataset")
            try:
                autoreg_train_dataset = torch.load(dataset_cache_dir / f'{args.pretrain_datasets_cache}_train.pt')
                autoreg_test_dataset = torch.load(dataset_cache_dir / f'{args.pretrain_datasets_cache}_test.pt')
            except:
                raise ValueError(f"Could not load cached datasets from {dataset_cache_dir / f'{args.pretrain_datasets_cache}_.pt'}")

        denoise_model = build_denoising_model(args, sample_dim=args.sample_dim)

        dp = build_diffusion_process(args)

        print("Running autoregressive pretraining")
        train_autoregressive_diff(args, autoreg_train_dataset, autoreg_test_dataset, encoder, denoise_model, dp, checkpoint_dir, writer=writer)
        model_weights = {
            'encoder': encoder.cpu().state_dict(),
            'denoise': denoise_model.cpu().state_dict()
        }
        torch.save(model_weights, checkpoint_dir / f'pretrained.pt')

    elif args.pretrain == 'contrastive':
        contrastive_train_dataset = ContrastivePatientDataset(folder=args.pretrain_data_dir, pretrain_embedding=args.pretrain_embedding, cells_per_view=args.cells_per_view, preload=args.no_preload, split_col=args.pretrain_fold_col, split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8], patient_csv=args.patient_csv, latent_idx_tokeep=args.latent_idx_tokeep)
        contrastive_test_dataset = ContrastivePatientDataset(folder=args.pretrain_data_dir, pretrain_embedding=args.pretrain_embedding, cells_per_view=args.cells_per_view, preload=args.no_preload, split_col=args.pretrain_fold_col, split_ids=[9], patient_csv=args.patient_csv, latent_idx_tokeep=args.latent_idx_tokeep)

        projection_head = build_projection_head(args, sample_dim=args.sample_dim)
        
        print("Running contrastive pretraining")
        train_contrastive(args, encoder, projection_head, contrastive_train_dataset, contrastive_test_dataset, writer=writer)

        model_weights = {
            'encoder': encoder.cpu().state_dict(),
            'projection_head': projection_head.cpu().state_dict()
        }
        torch.save(model_weights, checkpoint_dir / f'pretrained.pt')

    elif args.pretrain == 'cellsorter':

        sorter_head = build_sorter_head(args, sample_dim=args.sample_dim)

        cellsorter_train_dataset = CellSorterPatientDataset(args.pretrain_data_dir, pretrain_embedding=args.pretrain_embedding, preload=args.no_preload, n_positives_per_sample=args.cellsort_npospersample, min_num_cells=100, split_col=args.pretrain_fold_col, split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8], patient_csv=args.patient_csv, debug=args.debug, latent_idx_tokeep=args.latent_idx_tokeep)
        cellsorter_test_dataset = CellSorterPatientDataset(args.pretrain_data_dir, pretrain_embedding=args.pretrain_embedding, preload=args.no_preload, n_positives_per_sample=args.cellsort_npospersample, min_num_cells=100, split_col=args.pretrain_fold_col, split_ids=[9], patient_csv=args.patient_csv, debug=args.debug, latent_idx_tokeep=args.latent_idx_tokeep)

        print("Running cellsorter pretraining")
        train_cellsorter(args, encoder, sorter_head, cellsorter_train_dataset, cellsorter_test_dataset, writer=writer)

        model_weights = {
            'encoder': encoder.cpu().state_dict(),
            'sorter_head': sorter_head.cpu().state_dict()
        }
        torch.save(model_weights, checkpoint_dir / f'pretrained.pt')\
        
    elif args.pretrain == 'clustermask':
        train_dataset = PatientDataset(folder=args.pretrain_data_dir, pretrain_embedding=args.pretrain_embedding, preload=args.no_preload, split_col=args.pretrain_fold_col, split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8], patient_csv=args.patient_csv, max_num_cells=args.max_num_cells, latent_idx_tokeep=args.latent_idx_tokeep)
        test_dataset = PatientDataset(folder=args.pretrain_data_dir, pretrain_embedding=args.pretrain_embedding, preload=args.no_preload, split_col=args.pretrain_fold_col, split_ids=[9], patient_csv=args.patient_csv, max_num_cells=args.max_num_cells, latent_idx_tokeep=args.latent_idx_tokeep)

        print("Running clustermask pretraining")
        train_clustermask(args, encoder, train_dataset, test_dataset, writer=writer)

        model_weights = {
            'encoder': encoder.cpu().state_dict(),
        }
        torch.save(model_weights, checkpoint_dir / f'pretrained.pt')

    elif args.pretrain == 'none':
        print("Skipping pretraining")
        model_weights = {
            'encoder': encoder.cpu().state_dict(),
        }
        torch.save(model_weights, checkpoint_dir / f'pretrained.pt')

    else:
        raise ValueError(f"Pretrain method {args.pretrain} not recognized. Specify a valid pretraining or set to 'none' for supervised pretraining.")

    if (not args.skip_linear_probe) or (not args.skip_frozen_finetuning) or (not args.skip_endtoend_finetuning):
        print("Loading adata...")
        full_adata = sc.read_h5ad(args.h5ad_loc)
        print("Done.")

    # Run Linear Probe
    if not args.skip_linear_probe:    
        print("Running linear probe")
        test_mean_acc, test_std_acc, test_mean_auc, test_std_auc, test_mean_f1, test_std_f1 = run_linear_probe(args, encoder, full_adata, results_dir=results_dir, subsample_ns=[25, 50, 100], nreps=5)
        print(f'10-Fold Test Accuracy of Linear Probe: {test_mean_acc} +- {test_std_acc}')
        print(f'10-Fold Test AUC of Linear Probe: {test_mean_auc} +- {test_std_auc}')
        print(f'10-Fold Test Weighted-F1 of Linear Probe: {test_mean_f1} +- {test_std_f1}')
    else:
        print("Skipping linear probe")

    if not args.skip_frozen_finetuning:
        print("Running supervised training with frozen encoder")
        run_finetuning(args, full_adata, encoder, args.sample_dim, checkpoint_dir, results_dir, device, writer, freeze_encoder=True)
    else:
        print("Skipping frozen finetuning")
    
    if not args.skip_endtoend_finetuning:
        print("Running end-to-end supervised training (unfrozen encoder)")
        run_finetuning(args, full_adata, encoder, args.sample_dim, checkpoint_dir, results_dir, device, writer, freeze_encoder=False)
    else:
        print("Skipping end-to-end finetuning")

def run_finetuning(args, full_adata, encoder, sample_dim, checkpoint_dir, results_dir, device, writer, freeze_encoder=False, reps=10):
    if not args.dont_run_kfold: #by default, run kfold
        print("Running and evaluating supervised training with K fold cross validation.")
        all_folds = full_adata.obs[args.fold_col].unique()

        fold_counter = 0                
        for testfold in all_folds:
            fold_counter += 1
            print("Running model {} on test fold {} (run {}/{})".format(args.run_name, testfold, fold_counter, len(all_folds)))
            print("Loading pretrained model checkpoint")
            if args.pretrained_ckpt is not None:
                model_weights = torch.load(args.pretrained_ckpt, weights_only=True)
            else:
                model_weights = torch.load(checkpoint_dir / f'pretrained.pt', weights_only=True)
            encoder.load_state_dict(model_weights['encoder'])

            args.testfold = testfold
            args.valfold = all_folds[fold_counter % len(all_folds)] #this formulation doesn't break if args.debug removes some folds from adata

            print("Creating CellBagDatasets...")
            now = datetime.now()
            supervised_train_dataset = CellBagDataset(adata=full_adata, pid_col=args.pid_col, target_col=args.target_col, fold_col=args.fold_col, adata_layer=args.adata_layer, split_ids=list(np.setdiff1d(full_adata.obs[args.fold_col].unique(), [args.valfold, args.testfold])), max_num_cells=args.max_num_cells, latent_idx_tokeep=args.latent_idx_tokeep)
            supervised_val_dataset = CellBagDataset(adata=full_adata, pid_col=args.pid_col, target_col=args.target_col, fold_col=args.fold_col, adata_layer=args.adata_layer, split_ids=[args.valfold], max_num_cells=args.max_num_cells, latent_idx_tokeep=args.latent_idx_tokeep)
            supervised_test_dataset = CellBagDataset(adata=full_adata, pid_col=args.pid_col, target_col=args.target_col, fold_col=args.fold_col, adata_layer=args.adata_layer, split_ids=[args.testfold], max_num_cells=args.max_num_cells, latent_idx_tokeep=args.latent_idx_tokeep)

            print("n samples in full training dataset: ", len(supervised_train_dataset))
            # for sample efficiency experiment
            for n in [10, 20, 50, 100]:
                if n > len(supervised_train_dataset):
                    print("Skipping sample efficiency experiment for n={} as n is greater than length of training dataset {}".format(n, len(supervised_train_dataset)))
                    continue
                for rep in range(reps): #repeat 10 times with different random subsets of the training data
                    print("Running supervised training with n={} samples".format(n))
                    supervised_train_dataset = torch.utils.data.Subset(supervised_train_dataset, np.random.choice(len(supervised_train_dataset), n, replace=False))

                    num_groups = supervised_train_dataset.num_groups
                    classifier_head = build_classifier_head(args, sample_dim=sample_dim, output_dim=num_groups)

                    print("Running supervised training")
                    train_acc, train_auc, train_f1, train_loss, val_acc, val_auc, val_f1, val_loss, test_acc, test_auc, test_f1, test_loss = train_supervised(args, encoder, classifier_head, supervised_train_dataset, supervised_val_dataset, supervised_test_dataset, device, checkpoint_dir, writer=writer, freeze_encoder=freeze_encoder)

                    experiment = 'supervised_frozen' if freeze_encoder else 'supervised'

                    write_results_sampleefficiency(args, results_dir, experiment, 'train_acc', train_acc, args.testfold, n, rep)
                    write_results_sampleefficiency(args, results_dir, experiment, 'train_auc', train_auc, args.testfold, n, rep)
                    write_results_sampleefficiency(args, results_dir, experiment, 'train_weightedf1', train_f1, args.testfold, n, rep)
                    write_results_sampleefficiency(args, results_dir, experiment, 'train_loss', train_loss, args.testfold, n, rep)
                    write_results_sampleefficiency(args, results_dir, experiment, 'val_acc', val_acc, args.valfold, n, rep)
                    write_results_sampleefficiency(args, results_dir, experiment, 'val_auc', val_auc, args.valfold, n, rep)
                    write_results_sampleefficiency(args, results_dir, experiment, 'val_weightedf1', val_f1, args.valfold, n, rep)
                    write_results_sampleefficiency(args, results_dir, experiment, 'val_loss', val_loss, args.valfold, n, rep)
                    write_results_sampleefficiency(args, results_dir, experiment, 'test_acc', test_acc, args.testfold, n, rep)
                    write_results_sampleefficiency(args, results_dir, experiment, 'test_auc', test_auc, args.testfold, n, rep)
                    write_results_sampleefficiency(args, results_dir, experiment, 'test_weightedf1', test_f1, args.testfold, n, rep)
                    write_results_sampleefficiency(args, results_dir, experiment, 'test_loss', test_loss, args.testfold, n, rep)

                    #release memory before next fold
                    del supervised_train_dataset, supervised_val_dataset, supervised_test_dataset, classifier_head
                    torch.cuda.empty_cache()
                    gc.collect()

    else:
        if args.debug: #handle the case that the small adata doesn't have fold column specified in args
            args.testfold = full_adata.obs[args.fold_col].unique()[0]
            args.valfold = full_adata.obs[args.fold_col].unique()[1]
        print("Running and evaluating supervised training with a single fold.")
        print("trainfolds: ", list(np.setdiff1d(full_adata.obs[args.fold_col].unique(), [args.valfold, args.testfold])))
        print("valfolds: ", [args.valfold])
        print("testfolds: ", [args.testfold])

        print("Loading pretrained model checkpoint")
        if args.pretrained_ckpt is not None:
            model_weights = torch.load(args.pretrained_ckpt, weights_only=True)
        else:
            model_weights = torch.load(checkpoint_dir / f'pretrained.pt', weights_only=True)
        encoder.load_state_dict(model_weights['encoder'])

        now = datetime.now()
        print("Creating CellBagDatasets...")
        supervised_train_dataset = CellBagDataset(adata=full_adata, pid_col=args.pid_col, target_col=args.target_col, fold_col=args.fold_col, adata_layer=args.adata_layer, split_ids=list(np.setdiff1d(full_adata.obs[args.fold_col].unique(), [args.valfold, args.testfold])), max_num_cells=args.max_num_cells, latent_idx_tokeep=args.latent_idx_tokeep)
        supervised_val_dataset = CellBagDataset(adata=full_adata, pid_col=args.pid_col, target_col=args.target_col, fold_col=args.fold_col, adata_layer=args.adata_layer, split_ids=[args.valfold], max_num_cells=args.max_num_cells, latent_idx_tokeep=args.latent_idx_tokeep)
        supervised_test_dataset = CellBagDataset(adata=full_adata, pid_col=args.pid_col, target_col=args.target_col, fold_col=args.fold_col, adata_layer=args.adata_layer, split_ids=[args.testfold], max_num_cells=args.max_num_cells, latent_idx_tokeep=args.latent_idx_tokeep)
        print("...took ", datetime.now()-now)

        num_groups = supervised_train_dataset.num_groups
        if args.debug:
            print("num_groups: ", num_groups)

        classifier_head = build_classifier_head(args, sample_dim=args.sample_dim, output_dim=num_groups)
        print("Running supervised training")
        train_acc, train_auc, train_f1, train_loss, val_acc, val_auc, val_f1, val_loss, test_acc, test_auc, test_f1, test_loss = train_supervised(args, encoder, classifier_head, supervised_train_dataset, supervised_val_dataset, supervised_test_dataset, device, checkpoint_dir, writer=writer, freeze_encoder=freeze_encoder)

        experiment = 'supervised_frozen' if freeze_encoder else 'supervised'
        if args.debug:
            print(experiment)
            print("num_groups: ", num_groups)

        write_results(args, results_dir, experiment, 'train_acc', train_acc, args.testfold)
        write_results(args, results_dir, experiment, 'train_auc', train_auc, args.testfold)
        write_results(args, results_dir, experiment, 'train_weightedf1', train_f1, args.testfold)
        write_results(args, results_dir, experiment, 'train_loss', train_loss, args.testfold)
        write_results(args, results_dir, experiment, 'val_acc', val_acc, args.valfold)
        write_results(args, results_dir, experiment, 'val_auc', val_auc, args.valfold)
        write_results(args, results_dir, experiment, 'val_weightedf1', val_f1, args.valfold)
        write_results(args, results_dir, experiment, 'val_loss', val_loss, args.valfold)
        write_results(args, results_dir, experiment, 'test_acc', test_acc, args.testfold)
        write_results(args, results_dir, experiment, 'test_auc', test_auc, args.testfold)
        write_results(args, results_dir, experiment, 'test_weightedf1', test_f1, args.testfold)
        write_results(args, results_dir, experiment, 'test_loss', test_loss, args.testfold)

    return

if __name__ == '__main__':
    main()
