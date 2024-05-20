from generative.networks.nets import DiffusionModelUNet
from torch import nn
import torch

class DiffusionWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        ddpm_out_channels = args.embedding_dim[0]
        ddpm_in_channels = args.embedding_dim[0]
        if args.conditioning_type == "concat":
            ddpm_in_channels += args.conditioning_dim
        self.conditioning_type = args.conditioning_type

        if args.model_type == "small":
            self.model = DiffusionModelUNet(
                spatial_dims=args.spatial_dimension,
                in_channels=ddpm_in_channels,
                out_channels=ddpm_out_channels,
                num_channels=(128, 256, 256),
                attention_levels=(False, False, True),
                num_res_blocks=1,
                num_head_channels=8,
                with_conditioning=args.conditioning_type == "cross_attention",
                cross_attention_dim=args.conditioning_dim if args.conditioning_type == "cross_attention" else None,
            )
        elif args.model_type == "medium":
            self.model = DiffusionModelUNet(
                spatial_dims=args.spatial_dimension,
                in_channels=ddpm_in_channels,
                out_channels=ddpm_out_channels,
                num_channels=(256, 256, 256),
                attention_levels=(False, True, True),
                num_res_blocks=2,
                num_head_channels=8,
                with_conditioning=args.conditioning_type == "cross_attention",
                cross_attention_dim=args.conditioning_dim if args.conditioning_type == "cross_attention" else None,
            )
        elif args.model_type == "big":
            self.model = DiffusionModelUNet(
                spatial_dims=args.spatial_dimension,
                in_channels=ddpm_in_channels,
                out_channels=ddpm_out_channels,
                num_channels=(256, 512, 768),
                attention_levels=(True, True, True),
                num_res_blocks=2,
                num_head_channels=8,
                with_conditioning=args.conditioning_type == "cross_attention",
                cross_attention_dim=args.conditioning_dim if args.conditioning_type == "cross_attention" else None,
            )
        else:
            raise ValueError(f"Do not recognise model type {args.model_type}")

        if args.spatial_conditioning:
            self.spatial_xatt_emb = nn.Embedding(args.n_conditioning+1, args.conditioning_dim)

    def forward(self, x, timesteps, context=None, image_context=None):

        if context is None and image_context is None:
            out = self.model(x=x, timesteps=timesteps)
        else:
            if context is not None:
                context = torch.squeeze(context, 1)
                c_embd = self.spatial_xatt_emb(context)
                c_embd = torch.moveaxis(c_embd, -1, 1)
            if image_context is not None:
                x = torch.cat((x, image_context), dim=1)

            if self.conditioning_type == "cross_attention" and context is not None:
                out = self.model(x=x, timesteps=timesteps, context=c_embd)
            else:
                if context is not None:
                    x = torch.cat((x, c_embd), dim=1)
                out = self.model(x=x, timesteps=timesteps)

        return out
