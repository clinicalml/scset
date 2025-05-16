#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

# define tag for this string of experiments
tag="hlcatrinary_flowdecoder_newmlp_lr1e-5_finetune_final"

# define finetuning task
h5ad_loc="/data/rna_rep_learning/hlca_sikkema2023/hlca_sikkema2023_covidIPFnormal_unbalanced_n369_cxgcensus20240701.h5ad"
fold_col="grouped_fold"
adata_layer="scvi_cxgcensus_20240701"
pid_col="sample"
target_col="disease"

# which latents to use
latent_idx_tokeep="2 6 14 20 24 27 28 31 34 35 36 37 43 46"

# results for pre-training with diffusion
pretrain='flow'

python trainer.py \
  --encoder cell_transformer \
  --adata_layer ${adata_layer} \
  --h5ad_loc ${h5ad_loc} \
  --pid_col ${pid_col} \
  --target_col ${target_col} \
  --fold_col ${fold_col} \
  --pretrain ${pretrain} \
  --tag ${tag} \
  --num_blocks 2 \
  --num_heads 4 \
  --pretrain_epochs 400 \
  --latent_idx_tokeep ${latent_idx_tokeep} \
  --pretrain_lr 1e-5 \
  --pretrained_ckpt '/data/rna_rep_learning/scset/outputs/checkpoints/terrific-red-penguin-985-cell_transformer-flow-hlcatrinary_flowdecoder_newmlp_lr1e-5/pretrained.pt' \


echo "Done"
exit 0  





