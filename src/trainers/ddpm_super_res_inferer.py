import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import nibabel as nib

from src.data.get_train_and_val_dataloader import get_training_data_loader
from src.utils.simplex_noise import generate_simplex_noise
from generative.networks.schedulers import PNDMScheduler

from .base import BaseTrainer
from monai.utils import first
import numpy as np

class DDPMInfererSuperRes(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        if args.quick_test:
            print("Quick test enabled, only running on a single train and eval batch.")
        self.quick_test = args.quick_test
        self.logger_train = SummaryWriter(log_dir=str(self.run_dir / "train"))
        self.logger_val = SummaryWriter(log_dir=str(self.run_dir / "val"))
        self.conditioning = False if args.spatial_conditioning == "False" else True
        self.image_conditioning = False if args.image_conditioning == "False" else True
        self.num_epochs = args.n_epochs
        self.output_dir = args.output_dir + "/" + args.model_name + "/" + "output"
        self.sample_name_extension = args.ddpm_checkpoint_epoch
        self.conditioning = args.spatial_conditioning
        self.spatial_bins = args.n_conditioning
        self.train_loader, self.val_loader = get_training_data_loader(
            batch_size=args.batch_size,
            training_ids=args.training_ids,
            validation_ids=args.validation_ids,
            augmentation=bool(args.augmentation),
            num_workers=args.num_workers,
            cache_data=bool(args.cache_data),
            is_grayscale=bool(args.is_grayscale),
            spatial_dimension=args.spatial_dimension,
            image_size=self.image_size,
            image_roi=args.image_roi,
            has_coordconv=args.input_has_coordconv,
            spatial_conditioning=args.spatial_conditioning,
            spatial_conditioning_file=args.spatial_encoding_path,
            image_conditioning_file=args.image_conditioning_file,
            image_conditioning=args.image_conditioning,
        )
        self.postfix = args.sample_postfix
        self.repetitions = args.num_repetitions
        self.denoising_steps = args.num_steps_denoise



    @torch.no_grad()
    def inference(self):
        progress_bar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            ncols=70,
            position=0,
            leave=True,
            desc="Validation",
        )

        for step, batch in progress_bar:

            for repeat_std in range(1):

                if repeat_std == 0:
                    images = batch["image"].to(self.device)
                    sample = images[0,0]

                    new_qform = [[1.6, 0, 0, 0],
                                 [0, 1.6, 0, 0],
                                 [0, 0, 2.5, 0],
                                 [0, 0, 0, 1]]

                    sample = sample.cpu().numpy()
                    sample_nii = nib.Nifti1Image(sample, affine=new_qform)
                    nib.save(sample_nii,
                             self.output_dir + "/sample_original_sample" + self.postfix + "_base_dim_96.nii.gz")

                images = self.vqvae_model.encode_stage_2_inputs(batch["image"].to(self.device))

                pndm_scheduler = PNDMScheduler(
                    num_train_timesteps=1000,
                    skip_prk_steps=True,
                    prediction_type=self.prediction_type,
                    schedule=self.beta_schedule,
                    beta_start=self.beta_start,
                    beta_end=self.beta_end,
                )

                img_size_x = images.shape[2]
                img_size_y = images.shape[3]
                img_size_z = images.shape[4]
                to_pad_x = (8 - img_size_x % 8) if img_size_x % 8 != 0 else 0
                to_pad_y = (8 - img_size_y % 8) if img_size_y % 8 != 0 else 0
                to_pad_z = (8 - img_size_z % 8) if img_size_z % 8 != 0 else 0
                to_pad_x_1 = to_pad_x // 2
                to_pad_x_2 = to_pad_x - to_pad_x_1
                to_pad_y_1 = to_pad_y // 2
                to_pad_y_2 = to_pad_y - to_pad_y_1
                to_pad_z_1 = to_pad_z // 2
                to_pad_z_2 = to_pad_z - to_pad_z_1

                padding = (to_pad_z_1, to_pad_z_2, to_pad_y_1, to_pad_y_2, to_pad_x_1, to_pad_x_2)

                inverse_pad = [-x for x in padding]

                conditioning = batch["spatial_latent"].long().to(self.device)
                num_samples = 1
                images = images[:num_samples]

                conditioning_orig = conditioning[:num_samples]

                if self.conditioning:

                    if self.dynamic_latent_pad:
                        images = F.pad(input=images, pad=padding, mode="constant", value=0)
                    if conditioning is not None:
                        conditioning_orig = F.pad(input=conditioning_orig, pad=padding, mode="constant", value=self.spatial_bins)

                    noise = torch.randn_like(images).to(self.device)

                else:
                    if self.dynamic_latent_pad:
                        images = F.pad(input=images, pad=padding, mode="constant", value=0)
                    noise = torch.randn((1, *tuple(images.shape[1:]))).to(self.device)
                    conditioning_orig = None

                ##### ADD NOISE AND DENOISE HERE #######
                pndm_scheduler.set_timesteps(1000)
                pndm_timesteps = pndm_scheduler.timesteps
                t_start = self.denoising_steps
                start_timesteps = torch.Tensor([t_start] * images.shape[0]).long()

    ###################################### GET ORIGINAL IMAGE DECODED #########################
                # if self.dynamic_latent_pad:
                #     samples = F.pad(
                #         input=images, pad=inverse_pad, mode="constant", value=0
                #     )
                # samples = self.vqvae_model.decode_stage_2_outputs(samples)
                #
                # new_qform = [[1.6, 0, 0, 0],
                #              [0, 1.6, 0, 0],
                #              [0, 0, 2.5, 0],
                #              [0, 0, 0, 1]]
                # sample = samples[0, 0]
                # sample = sample.cpu().numpy()
                # sample_nii = nib.Nifti1Image(sample, affine=new_qform)
                # nib.save(sample_nii, self.output_dir + "/sample_original_sample" + self.postfix + "_super_resolved_96_base_dim.nii.gz")
                #
                #
    #################################################################################################

                for reps in range(self.repetitions):
                    images = pndm_scheduler.add_noise(
                        original_samples=images * self.b_scale,
                        noise=noise,
                        timesteps=start_timesteps,
                    )

                    for step in pndm_timesteps[pndm_timesteps <= t_start]:
                        timesteps = torch.Tensor([step] * images.shape[0]).long()
                        model_output = self.model(
                            images, timesteps=timesteps.to(self.device), context=conditioning_orig
                        )
                        # 2. compute previous image: x_t -> x_t-1
                        images, _ = pndm_scheduler.step(
                            model_output, step, images
                        )

                    if reps == 0 and repeat_std > 0:
                        continue

                    if self.dynamic_latent_pad:
                        latent_samples = F.pad(
                            input=images, pad=inverse_pad, mode="constant", value=0
                        )

                    samples = self.vqvae_model.decode_stage_2_outputs(latent_samples)

                    new_qform = [[1.6, 0, 0, 0],
                         [0, 1.6, 0, 0],
                         [0, 0, 2.5, 0],
                         [0, 0, 0, 1]]

                    sample = samples[0,0]

                    sample = sample.cpu().numpy()
                    sample_nii = nib.Nifti1Image(sample, affine=new_qform)

                    if repeat_std == 0:
                        nib.save(sample_nii, self.output_dir + "/sample_" + str(reps) + self.postfix + "_super_resolved_" + str(self.denoising_steps) + "_steps_96_base_dim_result.nii.gz")
                    else:
                        nib.save(sample_nii, self.output_dir + "/sample_" + str(repeat_std) + "_" + str(reps) + self.postfix +"_super_resolved" + str(self.denoising_steps) + "_steps_96_base_dim_stdcalc.nii.gz")

        return


        

