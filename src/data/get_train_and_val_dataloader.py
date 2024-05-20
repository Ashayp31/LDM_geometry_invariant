import pandas as pd
import torch.distributed as dist
from monai import transforms
from monai.data import CacheDataset, Dataset, ThreadDataLoader, partition_dataset, DataLoader
from monai.utils import first
import random
import os
import numpy as np
from typing import Tuple, Union

from monai.data.utils import pad_list_data_collate
# from src.data.pad_collate import pad_list_data_collate

from monai.transforms import MapTransform
from monai.config import KeysCollection
import torch
import torch.nn as nn



class RotateImages(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        img = d["image"]

        img = torch.rot90(img, k=1, dims=[2, 3])

        d["image"] = img
        d["spatial_latent"] = img
        return d

class RandResizeImg(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        rand_val = random.uniform(0, 1)
        img = d["image"]

        if rand_val < 1.0:
            rand_resolution_change = 1 + random.uniform(0, 1)
            x_len = img.shape[1]
            y_len = img.shape[2]
            z_len = img.shape[3]
            new_x_len = int(16 * ((x_len / rand_resolution_change) // 16))
            new_y_len = int(16 * ((y_len / rand_resolution_change) // 16))
            new_z_len = int(16 * ((z_len / rand_resolution_change) // 16))

            resize_transform = transforms.Resize(spatial_size=[new_x_len, new_y_len, new_z_len], mode="trilinear")
            img = resize_transform(img)
            np.nan_to_num(img, copy=False)
            d["image"] = img
            d["spatial_latent"] = img
        else:
            d["image"] = img
            d["spatial_latent"] = img

        return d


class CropWithoutPadding(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        rand_val_x = random.uniform(0, 1)
        rand_val_y = random.uniform(0, 1)
        rand_val_z = random.uniform(0, 1)
        image_data = d["image"]
        x_len = image_data.shape[1]
        y_len = image_data.shape[2]
        z_len = image_data.shape[3]

        if rand_val_x > 0.3:

            first_crop_x = random.randint(0, x_len - 96) if x_len > 96 else 0
            second_crop_x = random.randint(first_crop_x, x_len)
            image_size_x = second_crop_x - first_crop_x
            new_image_size_x = max(image_size_x, 96)
            second_crop_x = min(x_len,first_crop_x + new_image_size_x)
            new_image_size_x = min(192,((second_crop_x - first_crop_x)//16) * 16)
            second_crop_x = first_crop_x + new_image_size_x
            image_data = image_data[:,first_crop_x:second_crop_x,:,:]
        else:
            if x_len > 192:
                first_crop_x = random.randint(0, x_len - 192)
                second_crop_x = first_crop_x + 192
                image_data = image_data[:,first_crop_x:second_crop_x,:,:]

        if rand_val_y > 0.3:


            first_crop_y = random.randint(0, y_len - 96) if y_len > 96 else 0
            second_crop_y = random.randint(first_crop_y, y_len)
            image_size_y = second_crop_y - first_crop_y
            new_image_size_y = max(image_size_y, 96)
            second_crop_y = min(y_len,first_crop_y + new_image_size_y)
            new_image_size_y = min(192,((second_crop_y - first_crop_y)//16) * 16)
            second_crop_y = first_crop_y + new_image_size_y
            image_data = image_data[:,:,first_crop_y:second_crop_y,:]
        else:
            if y_len > 192:
                first_crop_y = random.randint(0, y_len - 192)
                second_crop_y = first_crop_y + 192
                image_data = image_data[:,:,first_crop_y:second_crop_y,:]

        if rand_val_z > 0.3:

            first_crop_z = random.randint(0, z_len - 96) if z_len > 96 else 0
            second_crop_z = random.randint(first_crop_z, z_len)
            image_size_z = second_crop_z - first_crop_z
            new_image_size_z = max(image_size_z, 96)
            second_crop_z = min(z_len,first_crop_z + new_image_size_z)
            new_image_size_z = min(192,((second_crop_z - first_crop_z)//16) * 16)
            second_crop_z = first_crop_z + new_image_size_z
            image_data = image_data[:,:,:,first_crop_z:second_crop_z]

        else:
            if z_len > 192:
                first_crop_z = random.randint(0, z_len - 192)
                second_crop_z = first_crop_z + 192
                image_data = image_data[:,:,:,first_crop_z:second_crop_z]

        d["image"] = image_data
        d["spatial_latent"] = image_data
        return d


avg_pool = nn.AvgPool3d(8, stride=8)
class GetCoordLatentVals(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        img = d["image"]

        spatial_size = img.shape
        coord_channels = img[1:]
        coord_channels = torch.unsqueeze(coord_channels, 0)

        new_img_coords = torch.zeros((3,spatial_size[1]//8, spatial_size[2]//8, spatial_size[3]//8))
        new_img_coords[0] = avg_pool(coord_channels[:,0])
        new_img_coords[1] = avg_pool(coord_channels[:,1])
        new_img_coords[2] = avg_pool(coord_channels[:,2])
        new_img_coords = torch.floor(new_img_coords*19)
        new_img_coords[new_img_coords<0] = 0
        new_img_coords[new_img_coords>19] = 19
        quantized_img_coords = torch.zeros((spatial_size[1]//8, spatial_size[2]//8, spatial_size[3]//8))

        for i in range(quantized_img_coords.shape[0]):
            for j in range(quantized_img_coords.shape[1]):
                for k in range(quantized_img_coords.shape[2]):
                    quantized_img_coords[i,j,k] = new_img_coords[0,i,j,k] + 20*new_img_coords[1,i,j,k] + 20*20*new_img_coords[2,i,j,k]
        quantized_img_coords = torch.unsqueeze(quantized_img_coords,0)
        d["spatial_latent"] = quantized_img_coords

        return d


def get_data_dicts(ids_path: Union[str, Tuple[str, ...]], spatial_encodings_file_path:str = None,
                   image_conditioning_file_path:str = None, shuffle: bool = False, first_n=False):

    """Get data dicts for data loaders."""
    if isinstance(ids_path, str):
        subjects_file_path = [ids_path]
    else:
        subjects_file_path = list(ids_path)

    subjects_files = []
    for path in subjects_file_path:
        if os.path.isdir(path):
            subjects_files.append([os.path.join(path, os.fsdecode(f)) for f in os.listdir(path)])
        elif os.path.isfile(path):
            if path.endswith(".csv"):
                subjects_files.append(
                    pd.read_csv(filepath_or_buffer=path, sep=",")["path"].to_list()
                )
            elif path.endswith(".tsv"):
                subjects_files.append(
                    pd.read_csv(filepath_or_buffer=path, sep="\t")["path"].to_list()
                )
        else:
            raise ValueError(
                "Path is neither a folder (to source all the files inside) or a csv/tsv with file paths inside."
            )

    if spatial_encodings_file_path:

        if os.path.isfile(spatial_encodings_file_path):
            if spatial_encodings_file_path.endswith(".csv"):
                spatial_encodings_file = pd.read_csv(
                    filepath_or_buffer=spatial_encodings_file_path, sep=","
                )
            elif spatial_encodings_file_path.endswith(".tsv"):
                spatial_encodings_file = pd.spatial_encodings_file_path(
                    filepath_or_buffer=spatial_encodings_file_path, sep="\t"
                )
        else:
            raise ValueError("Spatial Encoding Path is not a csv/tsv with file paths inside.")

    if image_conditioning_file_path:

        if os.path.isfile(image_conditioning_file_path):
            if image_conditioning_file_path.endswith(".csv"):
                image_conditioning_file = pd.read_csv(
                    filepath_or_buffer=image_conditioning_file_path, sep=","
                )
            elif image_conditioning_file_path.endswith(".tsv"):
                image_conditioning_file = pd.image_conditioning_file_path(
                    filepath_or_buffer=image_conditioning_file_path, sep="\t"
                )
        else:
            raise ValueError("Spatial Encoding Path is not a csv/tsv with file paths inside.")


    data_dicts = []
    mia_subjects = 0

    for file_list in subjects_files:
        for file in file_list:
            valid_subject = True
            subject_name = os.path.basename(file)
            subject = {"image": file}

            if spatial_encodings_file_path:

                try:
                    spatial_encoding_subject = spatial_encodings_file.loc[
                        spatial_encodings_file["subject"] == subject_name, "spatial"
                    ].values[0]
                except IndexError:

                    print("Cannot find Spatial Encoding npy file for ", subject_name)
                    mia_subjects += 1
                    valid_subject = False
                    continue

                subject["spatial_latent"] = spatial_encoding_subject

            if image_conditioning_file_path:

                try:
                    image_conditioning_subject = image_conditioning_file.loc[
                        image_conditioning_file["subject"] == subject_name, "image_conditioning"
                    ].values[0]
                except IndexError:

                    print("Cannot find Image Conditioning nii.gz file for ", subject_name)
                    mia_subjects += 1
                    valid_subject = False
                    continue

                subject["image_cond"] = image_conditioning_subject

            if valid_subject:
                data_dicts.append(subject)
    if shuffle:
        random.seed(a=1)
        random.shuffle(data_dicts)


    if first_n is not False:
        data_dicts = data_dicts[:first_n]

    print(f"Found {len(data_dicts)} subjects.")
    if dist.is_initialized():
        print(dist.get_rank())
        print(dist.get_world_size())
        return partition_dataset(
            data=data_dicts,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            seed=0,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]
    else:
        return data_dicts


def get_training_data_loader(
    batch_size: int,
    training_ids: Union[str, Tuple[str, ...]],
    validation_ids: str,
    only_val: bool = False,
    augmentation: bool = True,
    drop_last: bool = False,
    num_workers: int = 8,
    num_val_workers: int = 3,
    cache_data=True,
    first_n=None,
    is_grayscale=False,
    add_vflip=False,
    add_hflip=False,
    image_size=None,
    image_roi=None,
    spatial_dimension=2,
    has_coordconv=True,
    spatial_conditioning=False,
    spatial_conditioning_file=None,
    image_conditioning_file=None,
    image_conditioning=False,

):
    spatial_conditioning = False if spatial_conditioning == "False" else False if not spatial_conditioning else True
    image_conditioning = False if image_conditioning == "False" else False if not image_conditioning else True
    has_coordconv = False if has_coordconv == "False" else False if not has_coordconv else True

    # Define transformations


    keys = ["image"]
    if spatial_conditioning:
        keys += ["spatial_latent"]
    if image_conditioning:
        keys += ["image_cond"]
    image_keys = ["image"] if not image_conditioning else ["image","image_cond"]


    resize_transform = (transforms.Resized(keys=keys, spatial_size=(96,96,104)))
    resize_transform_2 = (transforms.Resized(keys=keys, spatial_size=(272,272,288)))

    elastic_transform = transforms.Rand3DElasticd(
        keys=image_keys,
        prob=0.5,  # 0.8,
        sigma_range=[1.0, 2.0],
        magnitude_range=[2.0, 5.0],
        rotate_range=[0, 0, 0.0],
        translate_range=[6, 6, 0],
        scale_range=[0.05, 0.05, 0],
        padding_mode="zeros"
    )

    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=image_keys, channel_dim='no_channel') if not has_coordconv else transforms.EnsureChannelFirstd(keys=image_keys, channel_dim=-1),
            # transforms.EnsureChannelFirstd(keys=image_keys, channel_dim=-1),
            elastic_transform,
            # resize_transform,
            # resize_transform_2,
            # RotateImages(keys=keys),
            # GetCoordLatentVals(keys=keys),
            transforms.SignalFillEmptyd(keys=keys),
            transforms.ThresholdIntensityd(keys=image_keys, threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=image_keys, threshold=0, above=True, cval=0),
            transforms.ToTensord(keys=keys),
        ]
    )

    # no augmentation for now
    if augmentation:
        train_transforms = val_transforms
    else:
        train_transforms = val_transforms

    val_dicts = get_data_dicts(validation_ids, spatial_conditioning_file, image_conditioning_file, shuffle=False, first_n=first_n)
    if first_n:
        val_dicts = val_dicts[:first_n]

    if cache_data:
        val_ds = CacheDataset(
            data=val_dicts,
            transform=val_transforms,
        )
    else:
        val_ds = Dataset(
            data=val_dicts,
            transform=val_transforms,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=num_val_workers,
        collate_fn=pad_list_data_collate,
        drop_last=drop_last,
        pin_memory=False,
    )

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(training_ids, spatial_conditioning_file, image_conditioning_file, shuffle=False, first_n=first_n)

    if cache_data:
        train_ds = CacheDataset(
            data=train_dicts,
            transform=train_transforms,
        )
    else:
        train_ds = Dataset(
            data=train_dicts,
            transform=train_transforms,
        )

    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate,
        drop_last=drop_last,
        pin_memory=False)

    return train_loader, val_loader
