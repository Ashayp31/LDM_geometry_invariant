"""
Model downloaded from https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view

using model
resnet_50_23dataset

This loading strategy was taken from MedNet3D codebase since their checkpoints did not include the segmentation
  heads so they need to be copied from the initialized network.
  https://github.com/Tencent/MedicalNet/blob/master/model.py#L87        net_dict = network.state_dict()
Cleaning the name of the methods due to the fact that the weights were saved from a DDP/DP model instead of
  the underlying .module attribute
"""

import torch
from monai import transforms
from monai.data import Dataset
from piq import FID
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.networks.med3d import MedNet3D, Bottleneck

model = MedNet3D(
    layers=[3, 4, 6, 3],
    block=Bottleneck,
    shortcut_type='B'
)
net_dict = model.state_dict()

pretrain_dict = torch.load(
    "/nfs/home/apatel/CT_PET_FDG/med_3d/MedicalNet_pytorch_files2/pretrain/resnet_50_23dataset.pth")[
    "state_dict"]
pretrain_dict = {k.replace("module.", ""): v for k, v in pretrain_dict.items()}

pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in net_dict.keys()}
net_dict.update(pretrain_dict)

device = torch.device("cuda")
model.load_state_dict(net_dict)
model = model.to(device)
model.eval()


data_dicts = []
for file in os.listdir("/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered"):
    data_dicts.append({"image": "/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered/" + file})

resize_transform = (
    transforms.Resized(keys=["image"], spatial_size=(184,152,192))
)

original_data_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=['image']),
        transforms.EnsureChannelFirstd(keys=['image']),
        resize_transform,
        transforms.ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
        transforms.ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
        transforms.ToTensord(keys=['image'])
    ]
)


original_data_df = Dataset(
    data=data_dicts,
    transform=original_data_transforms,
)
original_data_loader = DataLoader(
    original_data_df,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    drop_last=False,
    pin_memory=False,
)

original_feats = []
pbar = tqdm(enumerate(original_data_loader), total=len(original_data_loader))
for step, x in pbar:
    img = x["image"].to(device)
    with torch.no_grad():
        feats = model(img.to(device))
    original_feats.append(feats.cpu())
    pbar.update()

original_feats = torch.cat(original_feats, axis=0)



output_data_dicts = []

for file in os.listdir("/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb/ddpm_small_CT_wb/output"):
    output_data_dicts.append({"image" : "/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb/ddpm_small_CT_wb/output/" + file})



eval_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=['image']),
        transforms.EnsureChannelFirstd(keys=['image']),
        transforms.ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
        transforms.ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
        transforms.ToTensord(keys=['image'])
    ]
)
eval_ds = Dataset(
    data=output_data_dicts,
    transform=eval_transforms,
)
eval_loader = DataLoader(
    eval_ds,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    drop_last=False,
    pin_memory=False,
)

eval_feats = []
pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
for step, x in pbar:
    img = x["image"].to(device)
    with torch.no_grad():
        feats = model(img.to(device))
    eval_feats.append(feats.cpu())
    pbar.update()

eval_feats = torch.cat(eval_feats, axis=0)

fid_metric = FID()
score = fid_metric(original_feats, eval_feats)
print("Score here")
print(score)
