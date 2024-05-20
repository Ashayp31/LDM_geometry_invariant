#!/bin/sh

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    /nfs/home/apatel/ddpm-main-xatt/train_ddpm.py \
  --output_dir=/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb \
  --model_name=ddpm_small_CT_spatial_conditioning_vq512_8_cond_dim_4 \
  --vqvae_checkpoint=/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/vqgan_ne512_dim8_CT/baseline_vqvae/checkpoints/checkpoint_epoch_result=34.pt \
  --training_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/37 \
  --validation_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov_diffusion_val \
  --spatial_encoding_path=/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/spatial_locations.csv \
  --is_grayscale=1 \
  --n_epochs=500 \
  --batch_size=6 \
  --eval_freq=10 \
  --checkpoint_every=25 \
  --cache_data=0  \
  --prediction_type=epsilon \
  --model_type=small \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=3 \
  --apply_coordConv=True \
  --input_has_coordconv=True \
  --dynamic_latent_pad=True \
  --spatial_conditioning=True \
  --conditioning_type=concat

