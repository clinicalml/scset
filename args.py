import argparse

def add_args(parser):

    # General options
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--device', default='cuda', help='Device to use for training / testing')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--tag', type=str, default=None, help='Tag for the experiment series')

    # Model architecture options

    ## Encoder options
    parser.add_argument('--encoder', type=str, default='set_transformer', choices=['set_transformer', 'abmil', 'cell_transformer', 'pseudo_bulk_classifier', 'pseudo_bulk', 'random_embedding', 'baseline'], help='Encoder to use.')
    parser.add_argument('--model_dim', type=int, default=None, help='Input dimension. Default depends on model type. CellTransformer: 256, else: 64')
    parser.add_argument('--feedforward_dim', type=int, default=512, help='Feedforward Dimension in Cell Transformer')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_seeds', type=int, default=100, help='Number of seeds (prev. referred to as inducing points) in PMA layers. Valid for SetTransformer')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='ABMIL and PseudoBulkClassifier contain an MLP. This is its number of hidden layers')
    parser.add_argument('--pseudobulk_project', action='store_true', help='Whether to project the pseudobulked embeddings to a lower dimension')
    
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function for MLP', choices=['relu', 'tanh', 'gelu', 'silu'])
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate for encoder.')
    parser.add_argument('--gated_attn', type=bool, default=True, help='Whether to use gated_attn for the ABMIL model')
    # parser.add_argument('--ln', action='store_true', help='Whether to use layer normalization in encoder') CURRENTLY ALWAYS USE LAYER NORM

    # Baseline embedding training options
    parser.add_argument('--skip_celltypefracs_baseline', action='store_true', help='Whether to skip cell type proportions baseline')
    parser.add_argument('--skip_celltypemeans_baseline', action='store_true', help='Whether to skip mean embedding per cell type baseline')
    parser.add_argument('--skip_catfracsmeans_baseline', action='store_true', help='Whether to skip categorical proportions and mean embedding baseline')
    parser.add_argument('--skip_kmeans_baseline', action='store_true', help='Whether to skip kmeans baseline')
    parser.add_argument('--kmeans_k', type=int, default=30, help='Number of clusters for kmeans baseline')

    ## Classifier head options
    parser.add_argument('--classifier_num_hidden_layers', type=int, default=2, help='Number of hidden layers in the classifier head. 0 equals linear head.')
    parser.add_argument('--classifier_activation', type=str, default="gelu", help='Activation used in the classifier head')
    parser.add_argument('--classifier_dropout', type=float, default=0., help='Dropout rate in the classifier head')
    parser.add_argument('--classifier_no_layernorm', action='store_false', help='Whether to use layernorm in the classifier head. Default yes if arg is not passed.')

    ## Denoising Model
    parser.add_argument('--time_emb_dim', type=int, default=128)
    parser.add_argument('--num_res_blocks', type=int, default=5)
    parser.add_argument('--denoising_act', type=str, default='silu')

    ## Projection Head Option - currently some of these are used both for contrastive and cellsorter pretraining
    parser.add_argument('--projection_num_hidden_layers', type=int, default=2, help='Number of hidden layers in the projection head. 0 equals linear head.')
    parser.add_argument('--projection_activation', type=str, default="gelu", help='Activation used in the projection head')
    parser.add_argument('--projection_dropout', type=float, default=0., help='Dropout rate in the projection head')
    parser.add_argument('--projection_no_layernorm', action='store_true', help='Whether to use layernorm in the projection head. Default yes if arg is not passed.')
    parser.add_argument('--projection_output_dim', type=int, default=64, help='Output dimension of the projection head')

    ## Clustermask Head Options
    parser.add_argument('--clustermask_num_hidden_layers', type=int, default=2, help='Number of hidden layers in the clustermask head. 0 equals linear head.')
    parser.add_argument('--clustermask_activation', type=str, default="gelu", help='Activation used in the clustermask head')
    parser.add_argument('--clustermask_dropout', type=float, default=0., help='Dropout rate in the clustermask head')
    parser.add_argument('--clustermask_no_layernorm', action='store_true', help='Whether to use layernorm in the clustermask head. Default yes if arg is not passed.')

    # Pre-Training Options
    parser.add_argument('--pretrain', type=str, default='none', help='Type of pretraining to use')
    parser.add_argument('--pretrain_epochs', type=int, default=200, help='Number of epochs to pretrain')
    parser.add_argument('--pretrain_batch_size', type=int, default=32, help='Batch size for pretraining')
    parser.add_argument('--pretrain_lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--pretrain_weight_decay', type=float, default=0.0)
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Path to the pretrained model checkpoint; will continue with fine-tuning/linear probe if provided.')
    parser.add_argument('--pretrained_ckpt_dir', type=str, default='/data/rna_rep_learning/scset/outputs/default_pretrained_ckpts/', 
                        help='Path to the directory containing default pretrained checkpoints. Expected models name in directory is pretrained-{encoder}-{pretrain}.pt; not used if pretrained_ckpt is provided.')
    parser.add_argument('--pretrain_data_dir', type=str, default='/data/rna_rep_learning/cellxgene_census_20240701', 
                        help='Path to base directory where pretraining cellxgene data lives. This directory should contain patient_index.csv, annotations directory and the embedding directory.')
    parser.add_argument('--pretrain_fold_col', type=str, default='grouped_fold', help='Name of column that contains fold information for pretraining.', choices=['grouped_fold', 'random_fold', 'debug_fold'])
    parser.add_argument('--no_preload', action='store_false', help='Load pretraining patients on the fly each batch, rather than loading them all into memory upfront')
    parser.add_argument('--load_cached_pretrain_dataset', action='store_true', help='Load cached dataset for pretraining')
    parser.add_argument('--pretrain_datasets_cache', type=str, default=None, help='Name to assign to processed torch Datasets when cached for fast loading, or name of cache to load  if args.load_cached_pretrain_dataset is true. _test and _train will be appended to this name.')


    ## Diffusion Options
    parser.add_argument('--num_anchor_cells', type=int, default=None, help='Number of anchor cells to use for diffusion. If None - uses all available cells except target cells.')
    parser.add_argument('--num_target_cells', type=int, default=16)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--num_steps_per_sample', type=int, default=16)
    parser.add_argument('--pretrain_clip_grad', type=float, default=0.1, help='Gradient norm clipping for pretraining. Important for diffusion training (currently only used there).')

    # Skip Options
    parser.add_argument('--skip_linear_probe', action='store_true', help='Whether to skip linear probe')
    parser.add_argument('--skip_frozen_finetuning', action='store_true', help='Whether to skip CxG pretraining')
    parser.add_argument('--skip_endtoend_finetuning', action='store_true', help='Whether to skip end-to-end fine-tuning')

    ## Contrastive Options
    parser.add_argument('--cells_per_view', type=int, default=100)

    ## Cellsorter Options
    parser.add_argument('--cellsort_npospersample', type=int, default=128, help='Number of positive examples per sample. Will create an equal number of negative examples.')

    # Fine-Tuning Options
    parser.add_argument('--epochs', default=100, type=int, help='Total epochs to train')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use', choices=['adam', 'adamax', 'sgd'])
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='exponential', help='Type of learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Length of learning rate warm-up')
    parser.add_argument('--exp_decay', type=float, default=1., help='Learning rate schedule exponential decay rate')
    parser.add_argument('--exp_decay_freq', type=int, default=1, help='Learning rate exponential decay frequency')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer.')

    # Early stopping options, used for both pretraining and fine-tuning
    parser.add_argument('--earlystop_patience_pt', type=int, default=0, help='Number of epochs to wait before early stopping during prectraining. Set to 0 to disable early stopping.')
    parser.add_argument('--earlystop_patience_ft', type=int, default=50, help='Number of epochs to wait before early stopping during finetuning. Set to 0 to disable early stopping.')
    parser.add_argument('--earlystop_mindelta', type=float, default=0.001, help='Minimum change in validation loss to be considered an improvement')

    # Data options
    parser.add_argument('--dataset_type', default='rnaseq', type=str, help='Dataset to train on, one of ShapeNet / MNIST / MultiMNIST / RNAseq', choices=['shapenet15k', 'mnist', 'multimnist', 'rnaseq'])
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading threads')

    # Dataset options
    parser.add_argument('--pretrain_embedding', type=str, default='scvi', help='Name of the embedding to use for CxG pretraining')
    parser.add_argument('--h5ad_loc', default=None, type=str, help='Path to the h5ad file containing your anndata object')
    parser.add_argument('--cache_dir', default=None, type=str, help='Path where processed data will be cached for fast loading. Defaults to current working dir.')
    parser.add_argument('--adata_layer', default='scvi_cxgcensus', type=str, help='Layer of the anndata object to use. Default pca.')
    parser.add_argument('--num_pcs', default=20, type=int, help='Number of principal components to use. Default 20.')
    parser.add_argument('--data_name', type=str, help='Name of RNAseq dataset.')
    parser.add_argument('--pid_col', type=str, default="pid", help='Name of column that contains patient identifiers.')
    parser.add_argument('--target_col', type=str, default=None, help='Name of column that contains patient group ID.')
    parser.add_argument('--celltype_col', type=str, default="cell_type", help='Name of column that contains cell types.')
    parser.add_argument('--min_num_cells', type=int, default=128, help='Filter for patients with at least this many cells during pretraining.')
    parser.add_argument('--max_num_cells', type=int, default=1024, help='Maximum number of cells to use for any given patient sample.')
    parser.add_argument('--patient_csv', type=str, default='patient_index.csv', help='Name of the csv file containing metadata for all cells (from CxG) to be included in pretraining. Must be located in args.pretrain_data_dir.')
    parser.add_argument('--annots_dir', type=str, default='annotations', help='Name of the directory containing annotations for all cells (from CxG) to be included in pretraining. Must be located in args.pretrain_data_dir.')
    parser.add_argument('--sample_dim', type=int, default=50, help='Number of gene features for each cell. Default is 50 corresponding to scVI in CxG.')
    parser.add_argument('--latent_idx_tokeep', type=int, nargs='+', default=None, help='Space-separated indices of latent dimensions to keep. Default is None, which keeps all dimensions.')
    parser.add_argument('--libnorm', action='store_true', help='Whether to library normalize the data (should pass for counts data). Normalizes anchor cells after pseudobulking them, as well as target cells.')
    parser.add_argument('--libnorm_maxfrac', type=float, default=.05, help='Exclude (very) highly expressed genes for the computation of the normalization factor (size factor) for each cell. A gene is considered highly expressed, if it has more than libnorm_maxfrac of the total counts.')
    parser.add_argument('--lognorm', action='store_true', help='Whether to log normalize the data (should pass for counts data). Normalizes anchor cells after pseudobulking them, as well as target cells.')

    # Logger options
    parser.add_argument('--log_dir', default="/data/rna_rep_learning/scset/outputs/", type=str, help="Name for the log dir where subfolders for saving checkpoints and summaries will be created (default = current directory).")
    parser.add_argument('--work_dir', default="./", type=str, help="Name for the work dir of the project. This is where tensorboard logs will be saved.")
    parser.add_argument('--model_name', default=None, type=str, help="Name of the model when saving checkpoints, tensorboard, etc")
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=10)

    # Validation options
    parser.add_argument('--dont_run_kfold', action='store_true', help='Whether to run kfold cross validation.')
    parser.add_argument('--fold_col', type=str, default='grouped_fold', help='Name of column that contains fold information.')
    parser.add_argument('--testfold', type=int, default=1)
    parser.add_argument('--valfold', type=int, default=0)

    # Resume options
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to the checkpoint to be loaded. Should end with .pth')

    return parser

def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='scset args')
    parser = add_args(parser)
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()

    if args.model_dim is None:
        if args.encoder == 'set_transformer':
            args.model_dim = 64
        elif args.encoder == 'abmil':
            args.model_dim = 64
        elif args.encoder == 'cell_transformer':
            args.model_dim = 256
        elif args.encoder == 'pseudo_bulk_classifier':
            args.model_dim = 64
        elif args.encoder == 'pseudo_bulk':
            args.model_dim = 64
        elif args.encoder == 'random_embedding':
            args.model_dim = 64
        else:
            raise ValueError(f"encoder {args.encoder} not supported")


    return args
