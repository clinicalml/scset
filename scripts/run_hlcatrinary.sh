#! /bin/bash

export CUDA_VISIBLE_DEVICES=1

# define tag for this string of experiments
tag="hlcatrinary_hvlatents"

# define finetuning task
h5ad_loc="/data/rna_rep_learning/hlca_sikkema2023/hlca_sikkema2023_covidIPFnormal_unbalanced_n369_cxgcensus20240701.h5ad"
fold_col="grouped_fold"
adata_layer="scvi_cxgcensus_20240701"
pid_col="sample"
target_col="disease"

# which latents to use
latent_idx_tokeep="2 6 14 20 24 27 28 31 34 35 36 37 43 46"

# results for pre-training with diffusion
pretrain='diffusion'

python trainer.py \
  --encoder abmil \
  --adata_layer ${adata_layer} \
  --h5ad_loc ${h5ad_loc} \
  --pid_col ${pid_col} \
  --target_col ${target_col} \
  --fold_col ${fold_col} \
  --pretrain ${pretrain} \
  --tag ${tag} \
  --lr 1e-3 \
  --classifier_num_hidden_layers 4 \
  --pretrain_epochs 400 \
  --latent_idx_tokeep ${latent_idx_tokeep} \

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

#train pseudobulk diffusion model for deconvolution application
python trainer.py \
  --encoder pseudo_bulk \
  --adata_layer ${adata_layer} \
  --h5ad_loc ${h5ad_loc} \
  --pid_col ${pid_col} \
  --target_col ${target_col} \
  --fold_col ${fold_col} \
  --pretrain ${pretrain} \
  --tag ${tag} \
  --pretrain_epochs 400 \
  --latent_idx_tokeep ${latent_idx_tokeep} \

# compare abmil and cell-transformer with no pretraining
pretrain='none'

python trainer.py \
  --encoder abmil \
  --adata_layer ${adata_layer} \
  --h5ad_loc ${h5ad_loc} \
  --pid_col ${pid_col} \
  --target_col ${target_col} \
  --fold_col ${fold_col} \
  --pretrain ${pretrain} \
  --tag ${tag} \
  --lr 1e-3 \
  --classifier_num_hidden_layers 4 \
  --latent_idx_tokeep ${latent_idx_tokeep} \

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
  --latent_idx_tokeep ${latent_idx_tokeep} \

# run baselines on downstream task

python trainer_baseline_encoders.py \
  --adata_layer ${adata_layer} \
  --h5ad_loc ${h5ad_loc} \
  --pid_col ${pid_col} \
  --target_col ${target_col} \
  --fold_col ${fold_col} \
  --tag ${tag} \
  --latent_idx_tokeep ${latent_idx_tokeep} \
  
echo "Done"
exit 0  





