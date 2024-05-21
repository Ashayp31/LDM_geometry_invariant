import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.get_train_and_val_dataloader import get_training_data_loader
from src.utils.simplex_noise import generate_simplex_noise

from .base import BaseTrainer
from monai.utils import first

class DDPMTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        if args.quick_test:
            print("Quick test enabled, only running on a single train and eval batch.")
        self.quick_test = args.quick_test
        self.logger_train = SummaryWriter(log_dir=str(self.run_dir / "train"))
        self.logger_val = SummaryWriter(log_dir=str(self.run_dir / "val"))
        self.num_epochs = args.n_epochs
        self.conditioning = False if args.spatial_conditioning == "False" else True
        self.image_conditioning = False if args.image_conditioning == "False" else True
        print(self.image_conditioning)
        print(args.image_conditioning)
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

    def train(self, args):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            epoch_loss = self.train_epoch(epoch)
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

                self.save_checkpoint(
                    self.run_dir / "checkpoint.pth",
                    epoch,
                    save_message=f"Saving checkpoint for model with loss {self.best_loss}",
                )

            if args.checkpoint_every != 0 and (epoch + 1) % args.checkpoint_every == 0:
                self.save_checkpoint(
                    self.run_dir / f"checkpoint_{epoch+1}.pth",
                    epoch,
                    save_message=f"Saving checkpoint at epoch {epoch+1}",
                )

            if (epoch + 1) % args.eval_freq == 0:
                self.model.eval()
                self.val_epoch(epoch)
        print("Training completed.")
        if self.ddp:
            dist.destroy_process_group()

    def train_epoch(self, epoch):
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            ncols=70,
            position=0,
            leave=True,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = 0
        epoch_step = 0
        self.model.train()
        for step, batch in progress_bar:

            images = self.vqvae_model.encode_stage_2_inputs(batch["image"].to(self.device))


            if self.conditioning:
                conditioning = batch["spatial_latent"].long().to(self.device)
            else:
                conditioning = None

            if self.image_conditioning:
                image_condition = self.vqvae_model.encode_stage_2_inputs(batch["image_cond"].to(self.device))
            else:
                image_condition = None

            img_size_x = images.shape[2]
            img_size_y = images.shape[3]
            img_size_z = images.shape[4]
            to_pad_x = (8-img_size_x%8) if img_size_x%8 != 0 else 0
            to_pad_y = (8-img_size_y%8) if img_size_y%8 != 0 else 0
            to_pad_z = (8-img_size_z%8) if img_size_z%8 != 0 else 0
            to_pad_x_1 = to_pad_x//2
            to_pad_x_2 = to_pad_x - to_pad_x_1
            to_pad_y_1 = to_pad_y // 2
            to_pad_y_2 = to_pad_y - to_pad_y_1
            to_pad_z_1 = to_pad_z // 2
            to_pad_z_2 = to_pad_z - to_pad_z_1

            padding = (to_pad_z_1, to_pad_z_2, to_pad_y_1, to_pad_y_2, to_pad_x_1, to_pad_x_2)
            inverse_pad = [-x for x in padding]

            if self.dynamic_latent_pad:
                with torch.no_grad():

                    images = F.pad(input=images, pad=padding, mode="constant", value=0)
                    if conditioning is not None:
                        conditioning = F.pad(input=conditioning, pad=padding, mode="constant", value=self.spatial_bins)
                    if image_condition is not None:
                        image_condition = F.pad(input=image_condition, pad=padding, mode="constant", value=0)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                timesteps = torch.randint(
                    0,
                    self.inferer.scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=self.device,
                ).long()

                # noise images
                if self.simplex_noise:
                    noise = generate_simplex_noise(
                        self.simplex, x=images, t=timesteps, in_channels=images.shape[1]
                    )
                else:
                    noise = torch.randn_like(images).to(self.device)

                noisy_image = self.scheduler.add_noise(
                    original_samples=images * self.b_scale, noise=noise, timesteps=timesteps
                )
                noise_prediction = self.model(x=noisy_image, timesteps=timesteps, context=conditioning, image_context=image_condition)
                loss = F.mse_loss(noise_prediction.float(), noise.float())
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            epoch_loss += loss.item()
            self.global_step += images.shape[0]
            epoch_step += images.shape[0]
            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / epoch_step,
                }
            )

            self.logger_train.add_scalar(
                tag="loss", scalar_value=loss.item(), global_step=self.global_step
            )
            if self.quick_test:
                break
        epoch_loss = epoch_loss / epoch_step
        return epoch_loss

    @torch.no_grad()
    def val_epoch(self, epoch):
        progress_bar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            ncols=70,
            position=0,
            leave=True,
            desc="Validation",
        )
        epoch_loss = 0
        global_val_step = self.global_step
        val_steps = 0

        for step, batch in progress_bar:

            images = self.vqvae_model.encode_stage_2_inputs(batch["image"].to(self.device))

            if self.conditioning:
                conditioning = batch["spatial_latent"].long().to(self.device)
            else:
                conditioning = None

            if self.image_conditioning:
                image_condition = self.vqvae_model.encode_stage_2_inputs(batch["image_cond"].to(self.device))
            else:
                image_condition = None

            img_size_x = images.shape[2]
            img_size_y = images.shape[3]
            img_size_z = images.shape[4]
            to_pad_x = (8-img_size_x%8) if img_size_x%8 != 0 else 0
            to_pad_y = (8-img_size_y%8) if img_size_y%8 != 0 else 0
            to_pad_z = (8-img_size_z%8) if img_size_z%8 != 0 else 0
            to_pad_x_1 = to_pad_x//2
            to_pad_x_2 = to_pad_x - to_pad_x_1
            to_pad_y_1 = to_pad_y // 2
            to_pad_y_2 = to_pad_y - to_pad_y_1
            to_pad_z_1 = to_pad_z // 2
            to_pad_z_2 = to_pad_z - to_pad_z_1

            padding = (to_pad_z_1, to_pad_z_2, to_pad_y_1, to_pad_y_2, to_pad_x_1, to_pad_x_2)
            inverse_pad = [-x for x in padding]

            if self.dynamic_latent_pad:

                images = F.pad(input=images, pad=padding, mode="constant", value=0)

            if conditioning is not None:
                conditioning = F.pad(input=conditioning, pad=padding, mode="constant", value=self.spatial_bins)
            if image_condition is not None:
                image_condition = F.pad(input=image_condition, pad=padding, mode="constant", value=0)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                timesteps = torch.randint(
                    0,
                    self.inferer.scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=images.device,
                ).long()

                # noise images
                if self.simplex_noise:
                    noise = generate_simplex_noise(
                        self.simplex, x=images, t=timesteps, in_channels=images.shape[1]
                    )
                else:
                    noise = torch.randn_like(images).to(self.device)

                noisy_image = self.scheduler.add_noise(
                    original_samples=images * self.b_scale, noise=noise, timesteps=timesteps
                )
                noise_prediction = self.model(x=noisy_image, timesteps=timesteps, context=conditioning, image_context=image_condition)
                loss = F.mse_loss(noise_prediction.float(), noise.float())
            self.logger_val.add_scalar(
                tag="loss", scalar_value=loss.item(), global_step=global_val_step
            )
            epoch_loss += loss.item()
            val_steps += images.shape[0]
            global_val_step += images.shape[0]
            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / val_steps,
                }
            )

        # get some samples
        image_size = images.shape[2]
        if self.spatial_dimension == 2:
            if image_size >= 128:
                num_samples = 4
                fig, ax = plt.subplots(2, 2)
            else:
                num_samples = 8
                fig, ax = plt.subplots(2, 4)
        elif self.spatial_dimension == 3:
            num_samples = 6
            fig, ax = plt.subplots(2, 3)

        all_samples = []
        progress_bar = tqdm(
            enumerate(self.val_loader),
            total=1,
            ncols=2,
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

            if self.image_conditioning:
                image_condition = self.vqvae_model.encode_stage_2_inputs(batch["image_cond"].to(self.device))
                image_condition = image_condition[:num_samples]
            else:
                image_condition = None

            if self.conditioning:

                conditioning = batch["spatial_latent"].long().to(self.device)
                images = images[:num_samples]
                conditioning = conditioning[:num_samples]
                if self.dynamic_latent_pad:
                    images = F.pad(input=images, pad=padding, mode="constant", value=0)
                    if image_condition is not None:
                        image_condition = F.pad(input=image_condition, pad=padding, mode="constant", value=0)
                if conditioning is not None:
                    conditioning = F.pad(input=conditioning, pad=padding, mode="constant", value=self.spatial_bins)

                noise = torch.randn_like(images).to(self.device)

            else:

                if self.dynamic_latent_pad:
                    images = F.pad(input=images, pad=padding, mode="constant", value=0)
                    if image_condition is not None:
                        image_condition = F.pad(input=image_condition, pad=padding, mode="constant", value=0)
                noise = torch.randn((1, *tuple(images.shape[1:]))).to(self.device)
                conditioning = None

            latent_samples = self.inferer.sample(
                input_noise=noise,
                diffusion_model=self.model,
                conditioning=conditioning if self.conditioning else None,
                image_condition=image_condition,
                scheduler=self.scheduler,
                verbose=True,
            )
            if self.dynamic_latent_pad:
                latent_samples = F.pad(
                    input=latent_samples, pad=inverse_pad, mode="constant", value=0
                )

            samples = self.vqvae_model.decode_stage_2_outputs(latent_samples)
            all_samples.append(samples)
            if len(all_samples) == num_samples:
                break


        if self.spatial_dimension == 2:
            for i in range(len(ax.flat)):
                ax.flat[i].imshow(
                    np.transpose(samples[i, ...].cpu().numpy(), (1, 2, 0)), cmap="gray"
                )
                plt.axis("off")
        elif self.spatial_dimension == 3:
            for sample_num in range(len(all_samples)):
                slice = int(0.5 * all_samples[sample_num].shape[3])
                a = all_samples[sample_num]
                ax[sample_num//3][sample_num%3].imshow(
                    np.transpose(all_samples[sample_num][0, :, :, slice, :].cpu().numpy(), (1, 2, 0)),
                    cmap="gray",
                )

        self.logger_val.add_figure(tag="samples", figure=fig, global_step=self.global_step)

