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

from .base import BaseTrainer
from monai.utils import first
import numpy as np

class DDPMInferer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        if args.quick_test:
            print("Quick test enabled, only running on a single train and eval batch.")
        self.quick_test = args.quick_test
        self.logger_train = SummaryWriter(log_dir=str(self.run_dir / "train"))
        self.logger_val = SummaryWriter(log_dir=str(self.run_dir / "val"))
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
        )
        self.postfix = args.sample_postfix


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
            images = self.vqvae_model.encode_stage_2_inputs(batch["image"].to(self.device))

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

        sample_number = 0

        for vers in range(2):
            all_samples = []
            images_orig = images[:num_samples]
            conditioning_orig = conditioning[:num_samples]

            if self.conditioning:

                if self.dynamic_latent_pad:
                    images_orig = F.pad(input=images_orig, pad=padding, mode="constant", value=0)
                if conditioning is not None:
                    conditioning_orig = F.pad(input=conditioning_orig, pad=padding, mode="constant", value=self.spatial_bins)

                noise = torch.randn_like(images_orig).to(self.device)

            else:
                if self.dynamic_latent_pad:
                    images_orig = F.pad(input=images_orig, pad=padding, mode="constant", value=0)
                noise = torch.randn((1, *tuple(images_orig.shape[1:]))).to(self.device)
                conditioning = None

            latent_samples = self.inferer.sample(
                input_noise=noise,
                diffusion_model=self.model,
                conditioning=conditioning_orig if self.conditioning else None,
                scheduler=self.scheduler,
                verbose=True,
            )
            if self.dynamic_latent_pad:
                latent_samples = F.pad(
                    input=latent_samples, pad=inverse_pad, mode="constant", value=0
                )

            samples = self.vqvae_model.decode_stage_2_outputs(latent_samples)
            all_samples.append(samples)

            new_qform = [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]

            for i in range(len(all_samples)):

                sample = all_samples[i][0,0]

                sample = sample.cpu().numpy()
                sample_nii = nib.Nifti1Image(sample, affine=new_qform)

                nib.save(sample_nii, self.output_dir + "/sample_" + str(sample_number) + self.postfix +".nii.gz")
                sample_number += 1

        return


        

