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

class DDPMVQVAE_Inference(BaseTrainer):
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

        sample_num = 0
        for step, batch in progress_bar:


            images = self.vqvae_model.encode_stage_2_inputs(batch["image"].to(self.device))
            samples = self.vqvae_model.decode_stage_2_outputs(images)

            new_qform = [[1.6, 0, 0, 0],
                         [0, 1.6, 0, 0],
                         [0, 0, 2.5, 0],
                         [0, 0, 0, 1]]
            sample = samples[0, 0]
            sample = sample.cpu().numpy()
            sample_nii = nib.Nifti1Image(sample, affine=new_qform)
            nib.save(sample_nii, self.output_dir + "/vqvae_recon_" + self.postfix + "_sample_number_" + str(sample_num) + ".nii.gz")
            sample_num += 1

        return


        

