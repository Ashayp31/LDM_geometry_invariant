import argparse
import ast
import os
from src.trainers import DDPMVQVAE_Inference


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location for models.")
    parser.add_argument("--model_name", help="Name of model.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--spatial_encoding_path", help="Location of csv file with spatial conditioning locations")
    parser.add_argument(
        "--spatial_dimension", default=2, type=int, help="Dimension of images: 2d or 3d."
    )
    parser.add_argument("--image_size", default=[272,272,288], help="Resize images.")
    parser.add_argument(
        "--image_roi",
        default=None,
        help="Specify central ROI crop of inputs, as a tuple, with -1 to not crop a dimension.",
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--latent_pad",
        default=None,
        help="Specify padding to apply to a latent, sometimes necessary to allow the DDPM U-net to work. Supply as a "
        "tuple following the 'pad' argument of torch.nn.functional.pad",
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--dynamic_latent_pad",
        default=False,
        help="Pad each image/latent to be a factor of 8 for a 3 layer Unet",
    )
    # model params
    parser.add_argument(
        "--vqvae_checkpoint",
        default=None,
        help="Path to a VQ-VAE model checkpoint, if you wish to train an LDM.",
    )

    parser.add_argument("--vqvae_network", default="baseline_vqvae")
    parser.add_argument("--use_subpixel_conv", default= False)
    parser.add_argument("--no_levels", default=3)
    parser.add_argument("--no_res_layers", default=3)
    parser.add_argument("--no_channels", default=256)
    parser.add_argument("--num_embeddings", default=(512,))
    parser.add_argument("--embedding_dim", default=(8,))
    parser.add_argument("--dropout_penultimate", default=True)
    parser.add_argument("--dropout_enc", default=0.0)
    parser.add_argument("--dropout_dec", default=0.0)
    parser.add_argument("--output_act", default=None)
    parser.add_argument("--downsample_parameters", default=(
        (4, 2, 1, 1),
        (4, 2, 1, 1),
        (4, 2, 1, 1),
    ))
    parser.add_argument("--upsample_parameters", default=(
        (4, 2, 1, 0, 1),
        (4, 2, 1, 0, 1),
        (4, 2, 1, 0, 1),
    ))
    parser.add_argument("--commitment_cost", default=(0.25,))
    parser.add_argument("--decay", default=(0.99,))
    parser.add_argument("--apply_coordConv", default=False)
    parser.add_argument("--input_has_coordconv", default=False)
    parser.add_argument("--spatial_conditioning", default=False)
    parser.add_argument("--conditioning_dim", default=4)
    parser.add_argument("--conditioning_type", default=None, help="Can be cross_attention, concat or None")
    parser.add_argument("--n_conditioning", default=8000, help="number of different tokens for spatial conditioning")

    parser.add_argument(
        "--prediction_type",
        default="epsilon",
        help="Scheduler prediction type to use: 'epsilon, sample, or v_prediction.",
    )
    parser.add_argument(
        "--model_type",
        default="small",
        help="Small or big model.",
    )
    parser.add_argument(
        "--beta_schedule",
        default="linear_beta",
        help="Linear_beta or scaled_linear_beta.",
    )
    parser.add_argument(
        "--beta_start",
        type=float,
        default=1e-4,
        help="Beta start.",
    )
    parser.add_argument(
        "--beta_end",
        type=float,
        default=2e-2,
        help="Beta end.",
    )
    parser.add_argument(
        "--b_scale",
        type=float,
        default=1,
        help="Scale the data by a factor b before noising.",
    )
    parser.add_argument(
        "--snr_shift",
        type=float,
        default=1,
        help="Shift the SNR of the noise scheduler by a factor to account for it increasing at higher resolution.",
    )
    parser.add_argument(
        "--simplex_noise",
        type=int,
        default=0,
        help="Use simplex instead of Gaussian noise.",
    )
    # training param
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs to train.")
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        help="Number of epochs to between evaluations.",
    )
    parser.add_argument(
        "--augmentation",
        type=int,
        default=1,
        help="Use of augmentation, 1 (True) or 0 (False).",
    )

    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument(
        "--cache_data",
        type=int,
        default=1,
        help="Whether or not to cache data in dataloaders.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="Save a checkpoint every checkpoint_every epochs.",
    )
    parser.add_argument(
        "--ddpm_checkpoint_epoch",
        default=None,
        help="If resuming, the epoch number for a specific checkpoint to resume from. If not specified, defaults to the best checkpoint.",
    )
    parser.add_argument("--is_grayscale", type=int, default=0, help="Is data grayscale.")
    parser.add_argument(
        "--quick_test",
        default=0,
        type=int,
        help="If True, runs through a single batch of the train and eval loop.",
    )
    parser.add_argument(
        "--sample_postfix",
        default="",
        type=str,
        help="Postfix to add to file name of samples generated",
    )

    args = parser.parse_args()
    return args


# to run using DDP, run torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0  train_ddpm.py --args
if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir + "/" + args.model_name + "/" + "output"):
        os.mkdir(args.output_dir + "/" + args.model_name + "/" + "output")
    else:
        print("Output Path Exists")
    inferer = DDPMVQVAE_Inference(args)
    inferer.inference()
