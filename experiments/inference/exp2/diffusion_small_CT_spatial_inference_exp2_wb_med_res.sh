#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/ddpm-main-xatt/sample_ddpm.py \
  --output_dir=/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb \
  --model_name=ddpm_small_CT_spatial_conditioning_vq512_8_cond_dim_4 \
  --vqvae_checkpoint=/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/vqgan_ne512_dim8_CT/baseline_vqvae/checkpoints/checkpoint_epoch_result=34.pt \
  --training_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/med_res_wb \
  --validation_ids=/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/med_res_wb \
  --spatial_encoding_path=/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_locations.csv \
  --is_grayscale=1 \
  --n_epochs=6000 \
  --batch_size=1 \
  --eval_freq=10 \
  --checkpoint_every=275 \
  --ddpm_checkpoint_epoch=500 \
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
  --conditioning_type=concat \
  --sample_postfix=_wb_med_res
