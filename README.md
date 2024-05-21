<h1 align="center">Resolution and Field of View Invariant Generative Modelling with Latent Diffusion Models</h1>
<p align="center">
</p>


<p align="center">
  <img width="800" height="300" src="https://github.com/Ashayp31/LDM_geometry_invariant/assets/62710884/393946f2-b3d5-43fe-853b-a28482f850ad">
</p>


## Intro
This codebase contains code for performing generative modelling of 3D medical imaging with diffusion models in addition to super-resolution tasks. It supports the use of DDPMs in addition to LDMs for higher dimensional data.
This work is the official implementation of [1] applied to 3D CT data. 

It supports the use of DDPMs as well as Latent Diffusion Models (LDM) for dealing with higher dimensional data.
It is based on work published in [1] and [2].

[1] [Resolution and Field of View Invariant Generative Modelling with Latent Diffusion Models]([https://arxiv.org/abs/2211.07740](https://openreview.net/pdf?id=VHfh2J8MQ6))

## Setup

### Install
Create a fresh virtualenv (this codebase was developed and tested with Python 3.8) and then install the required packages:

```pip install -r requirements.txt```

You can also build the docker image using the provided docker file.


## Run with LDM

### Train VQVAE
```bash
python3 run_vqvae.py run \
    --training_subjects=${path_to_training_subjects} \
    --validation_subjects=${path_to_validation_subjects}  \
    --project_directory=${project_directory_path} \
    --experiment_name='vqgan_ne512_dim8_CT' \
    --mode='training' \
    --device=0 \
    --amp=True \
    --deterministic=False \
    --cuda_benchmark=True \
    --seed=3 \
    --epochs=200 \
    --learning_rate=0.0001 \
    --gamma='auto' \
    --log_every=1 \
    --checkpoint_every=1 \
    --eval_every=1 \
    --loss='jukebox_perceptual' \
    --adversarial_component=True \
    --discriminator_network='baseline_discriminator' \
    --discriminator_learning_rate=0.0005 \
    --discriminator_loss='least_square' \
    --generator_loss='least_square' \
    --initial_factor_value=0 \
    --initial_factor_steps=25 \
    --max_factor_steps=50 \
    --max_factor_value=5 \
    --batch_size=3 \
    --normalize=False \
    --eval_batch_size=3 \
    --num_workers=8 \
    --prefetch_factor=8 \
    --starting_epoch=0 \
    --network='baseline_vqvae' \
    --use_subpixel_conv=False \
    --no_levels=3 \
    --no_res_layers=3 \
    --no_channels=256 \
    --codebook_type='ema' \
    --num_embeddings='(512,)' \
    --embedding_dim='(8,)' \
    --max_decay_epochs=50 \
    --dropout_penultimate=True \
    --dropout_enc=0.0 \
    --dropout_dec=0.0 \
    --act='LEAKYRELU' \
    --apply_coordConv=True \
    --input_has_coordConv=False \
    --cropping_type='without_padding' \
```

The VQVAE training code is DistributedDataParallel (DDP) compatible. For example to train with 4 GPUs run with: mpirun -np 4 --oversubscribe --allow-run-as-root python3 run_vqvae.py run \
    

### Train LDM
```bash
python train_ddpm.py \
  --output_dir=${output_root} \
  --model_name=ldm_ct_geometry_invariant \
  --vqvae_checkpoint=${output_root}/vqgan_ne512_dim8_CT/checkpoint.pth \
  --training_subjects=${path_to_training_subjects} \
  --validation_subjects=${path_to_validation_subjects}  \
  --is_grayscale=1 \
  --n_epochs=12000 \
  --batch_size=6 \
  --eval_freq=25 \
  --checkpoint_every=1000 \
  --cache_data=0  \
  --prediction_type=epsilon \
  --model_type=small \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=3 \
  --image_roi=[160,160,128] \
  --image_size=128
```

### Generate Samples
```bash
python3 sample_ddpm.py \
  --output_dir=${output_root} \
  --model_name=ldm_ct_geometry_invariant \
  --vqvae_checkpoint=${output_root}/vqgan_ne512_dim8_CT/checkpoint.pth \
  --validation_subjects=${path_to_validation_subjects}  \
  --spatial_encoding_path=${spatial_encoding_file_path} \
  --is_grayscale=1 \
  --n_epochs=6000 \
  --batch_size=6 \
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
  --conditioning_type=concat
````
For generating particular samples of a given field of view at a given resolution, a file of same size of the output image needs to be created with values along each dimension between 0-1 that correspond to the field-of-view.
These files are the ones stored as spatial_encoding_path and can also be used as the validation_subjects path to generate the starting dimension of the outputted generated image.

Example results from the paper can be seen below:
<p align="center">
  <img width="1200" height="600" src="https://github.com/Ashayp31/LDM_geometry_invariant/assets/62710884/0819a0d8-1325-4845-8729-98875cc77f33">
</p>


### Run Super-resolution
```bash
python3 super_res_ddpm.py \
  --output_dir=${output_root} \
  --model_name=ldm_ct_geometry_invariant \
  --vqvae_checkpoint=${output_root}/vqgan_ne512_dim8_CT/checkpoint.pth \
  --validation_subjects=${path_to_validation_subjects}  \
  --spatial_encoding_path=${spatial_encoding_file_path} \
  --is_grayscale=1 \
  --n_epochs=6000 \
  --batch_size=6 \
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
  --conditioning_type=concat
  --sample_postfix=_wb_lower_res \
  --num_steps_denoise=200
```
## Acknowledgements
Built with [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels) and [MONAI](https://github.com/Project-MONAI/MONAI).


## Citations
If you use this codebase, please cite
```bib
@InProceedings{AshayPatel_MIDL_2024,
    author    = {Patel, Ashay and Graham, Mark.S and Goh, Vicky, and Ourselin, Sebastien and Cardoso, M. Jorge},
    title     = {Resolution and Field of View Invariant Generative Modelling with Latent Diffusion Models},
    booktitle = {Proceedings of Machine Learning Research, Medical Imaging Deep Learning},
    month     = {July},
    year      = {2024},
}

}
```

