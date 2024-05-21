import numpy as np
import os
from monai import transforms
import shutil

from monai.transforms import MapTransform
from monai.config import KeysCollection

import random

import nibabel as nib
from monai.data import DataLoader, Dataset
from monai.utils import first
import pandas as pd

import torch.nn as nn
import torch


class Resize_img(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            x: list,
            new_res: list,
            allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.new_dim = x
        self.new_resolution = new_res

    def __call__(self, data):
        d = dict(data)
        img = d["image"]
        resolution_val = d["resolution"]

        d["resolution"] = self.new_resolution

        resize_transform = transforms.Resize(spatial_size=[self.new_dim[0], self.new_dim[1], self.new_dim[2]], mode="trilinear")
        img = resize_transform(img)
        np.nan_to_num(img, copy=False)
        d["image"] = img
        d["spatial_latent"] = img
        return d


class CropWithoutPaddingUB(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        image_data = d["image"]
        x_len = image_data.shape[1]
        y_len = image_data.shape[2]
        z_len = image_data.shape[3]
        print(image_data.shape)

        new_z_len = int(8 * ((z_len / 2) // 8))
        print(new_z_len)
        image_data = image_data[:,:,:,z_len-new_z_len:]

        d["image"] = image_data
        d["spatial_latent"] = image_data
        return d

class CropWithoutPaddingLB(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        image_data = d["image"]
        x_len = image_data.shape[1]
        y_len = image_data.shape[2]
        z_len = image_data.shape[3]
        print(image_data.shape)

        new_x_len = int(8 * ((x_len / 2) // 8))
        image_data = image_data[:,x_len-new_x_len:,:,:]

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
        img = d["spatial_latent"]
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
        d["spatial_latent"] = quantized_img_coords

        return d

def main():

    keys = ["image", "spatial_latent"]
    keys_image = ["image", "spatial_latent", "resolution"]

    ct_directory = "/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_CoordConv/"

    all_cts = os.listdir(ct_directory)
    all_cts.reverse()

    # os.mkdir(new_ct_directory)
    # os.mkdir(spatial_info_directory)
    num_samples = 0
    for i in range(3):
        if i == 0 or i==1:
            continue
        ct_file = ct_directory + all_cts[i]

        data_dicts = [
            {"image": ct_file, "spatial_latent": ct_file, "resolution": [1.6,1.6,2.5]}]
        training_subjects = [data_dicts[0]]
        print(ct_file)
        print(okay)


# (was 8 before)
        for x in range(1):
            resize_transform_1 = (Resize_img(keys=keys_image, x=[272,272,288],new_res=[1.6,1.6,2.5])) # nothing needs to be done with this one
            resize_transform_2 = (Resize_img(keys=keys_image, x=[200,200,216],new_res=[2.2,2.2,3.4]))
            resize_transform_3 = (Resize_img(keys=keys_image, x=[144,144,160],new_res=[3,3,4.7]))

            crop_UB_transform = CropWithoutPaddingUB(keys=keys)
            crop_LB_transform = CropWithoutPaddingLB(keys=keys)

            #  WB Original High Res Sample
            t1 = transforms.Compose(
                [
                    # load 4 Nifti images and stack them together
                    transforms.LoadImaged(keys=keys),
                    transforms.EnsureChannelFirstd(keys=keys),
                    transforms.SignalFillEmptyd(keys=keys),
                    transforms.ThresholdIntensityd(keys=keys, threshold=1, above=False, cval=1.0),
                    transforms.ThresholdIntensityd(keys=keys, threshold=0, above=True, cval=0),
                    GetCoordLatentVals(keys=keys)
                ])

            #  WB Middle Res Sample
            t2 = transforms.Compose(
                [
                    # load 4 Nifti images and stack them together
                    transforms.LoadImaged(keys=keys),
                    transforms.EnsureChannelFirstd(keys=keys),
                    resize_transform_2,
                    transforms.SignalFillEmptyd(keys=keys),
                    transforms.ThresholdIntensityd(keys=keys, threshold=1, above=False, cval=1.0),
                    transforms.ThresholdIntensityd(keys=keys, threshold=0, above=True, cval=0),
                    GetCoordLatentVals(keys=keys)
                ])

            #  WB Low Res Sample
            t3 = transforms.Compose(
                [
                    # load 4 Nifti images and stack them together
                    transforms.LoadImaged(keys=keys),
                    transforms.EnsureChannelFirstd(keys=keys),
                    resize_transform_3,
                    transforms.SignalFillEmptyd(keys=keys),
                    transforms.ThresholdIntensityd(keys=keys, threshold=1, above=False, cval=1.0),
                    transforms.ThresholdIntensityd(keys=keys, threshold=0, above=True, cval=0),
                    GetCoordLatentVals(keys=keys)
                ])

            #  WB Middle Res Sample UB
            t4 = transforms.Compose(
                [
                    # load 4 Nifti images and stack them together
                    transforms.LoadImaged(keys=keys),
                    transforms.EnsureChannelFirstd(keys=keys),
                    resize_transform_2,
                    crop_UB_transform,
                    transforms.SignalFillEmptyd(keys=keys),
                    transforms.ThresholdIntensityd(keys=keys, threshold=1, above=False, cval=1.0),
                    transforms.ThresholdIntensityd(keys=keys, threshold=0, above=True, cval=0),
                    GetCoordLatentVals(keys=keys)
                ])

            #  WB Middle Res Sample LB
            t5 = transforms.Compose(
                [
                    # load 4 Nifti images and stack them together
                    transforms.LoadImaged(keys=keys),
                    transforms.EnsureChannelFirstd(keys=keys),
                    resize_transform_2,
                    crop_LB_transform,
                    transforms.SignalFillEmptyd(keys=keys),
                    transforms.ThresholdIntensityd(keys=keys, threshold=1, above=False, cval=1.0),
                    transforms.ThresholdIntensityd(keys=keys, threshold=0, above=True, cval=0),
                    GetCoordLatentVals(keys=keys)
                ])


            num_samples += 1

            ds1 = Dataset(data=training_subjects, transform=t1)
            ds2 = Dataset(data=training_subjects, transform=t2)
            ds3 = Dataset(data=training_subjects, transform=t3)
            ds4 = Dataset(data=training_subjects, transform=t4)
            ds5 = Dataset(data=training_subjects, transform=t5)
            loader_1 = DataLoader(ds1, batch_size=1)
            loader_2 = DataLoader(ds2, batch_size=1)
            loader_3 = DataLoader(ds3, batch_size=1)
            loader_4 = DataLoader(ds4, batch_size=1)
            loader_5 = DataLoader(ds5, batch_size=1)

            data_1 = first(loader_1)
            ct_img_1 = data_1["image"][0]
            spatial_encoding_1 = data_1["spatial_latent"][0]
            res_1 = data_1["resolution"]
            x_new_dim = res_1[0]
            y_new_dim = res_1[1]
            z_new_dim = res_1[2]
            new_qform = [[x_new_dim, 0, 0, 0],
                         [0, y_new_dim, 0, 0],
                         [0, 0, z_new_dim, 0],
                         [0, 0, 0, 1]]
            ct_img_1 = np.array(ct_img_1)

            np.save("/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_res/spatial_encoding_wb_high_res.npy", spatial_encoding_1)
            ct_nii_img = nib.Nifti1Image(ct_img_1, affine=new_qform)
            nib.save(ct_nii_img, "/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/high_res_wb/high_res_wb.nii.gz")



            ############################### SAMPLE 2 ###########################

            data_2 = first(loader_2)
            ct_img_2 = data_2["image"][0]
            spatial_encoding_2 = data_2["spatial_latent"][0]
            res_2 = data_2["resolution"]
            x_new_dim = res_2[0]
            y_new_dim = res_2[1]
            z_new_dim = res_2[2]
            new_qform = [[x_new_dim, 0, 0, 0],
                         [0, y_new_dim, 0, 0],
                         [0, 0, z_new_dim, 0],
                         [0, 0, 0, 1]]
            ct_img_2 = np.array(ct_img_2)

            np.save("/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_res/spatial_encoding_wb_med_res.npy", spatial_encoding_2)
            ct_nii_img = nib.Nifti1Image(ct_img_2, affine=new_qform)
            nib.save(ct_nii_img, "/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/med_res_wb/med_res_wb.nii.gz")


            ############################### SAMPLE 3 ###########################

            data_3 = first(loader_3)
            ct_img_3 = data_3["image"][0]
            spatial_encoding_3 = data_3["spatial_latent"][0]
            res_3 = data_3["resolution"]
            x_new_dim = res_3[0]
            y_new_dim = res_3[1]
            z_new_dim = res_3[2]
            new_qform = [[x_new_dim, 0, 0, 0],
                         [0, y_new_dim, 0, 0],
                         [0, 0, z_new_dim, 0],
                         [0, 0, 0, 1]]
            ct_img_3 = np.array(ct_img_3)

            np.save("/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_res/spatial_encoding_wb_low_res.npy", spatial_encoding_3)
            ct_nii_img = nib.Nifti1Image(ct_img_3, affine=new_qform)
            nib.save(ct_nii_img, "/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/low_res_wb/low_res_wb.nii.gz")

            ############################### SAMPLE 4 ###########################

            data_4 = first(loader_4)
            ct_img_4 = data_4["image"][0]
            spatial_encoding_4 = data_4["spatial_latent"][0]
            res_4 = data_4["resolution"]
            x_new_dim = res_4[0]
            y_new_dim = res_4[1]
            z_new_dim = res_4[2]
            new_qform = [[x_new_dim, 0, 0, 0],
                         [0, y_new_dim, 0, 0],
                         [0, 0, z_new_dim, 0],
                         [0, 0, 0, 1]]
            ct_img_4 = np.array(ct_img_4)

            np.save("/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_res/spatial_encoding_ub_med_res.npy", spatial_encoding_4)
            ct_nii_img = nib.Nifti1Image(ct_img_4, affine=new_qform)
            nib.save(ct_nii_img, "/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/med_res_ub/med_res_ub.nii.gz")

            ############################### SAMPLE 5 ###########################

            data_5 = first(loader_5)
            ct_img_5 = data_5["image"][0]
            spatial_encoding_5 = data_5["spatial_latent"][0]
            res_5 = data_5["resolution"]
            x_new_dim = res_5[0]
            y_new_dim = res_5[1]
            z_new_dim = res_5[2]
            new_qform = [[x_new_dim, 0, 0, 0],
                         [0, y_new_dim, 0, 0],
                         [0, 0, z_new_dim, 0],
                         [0, 0, 0, 1]]
            ct_img_5 = np.array(ct_img_5)

            np.save("/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_res/spatial_encoding_lb_med_res.npy", spatial_encoding_5)
            ct_nii_img = nib.Nifti1Image(ct_img_5, affine=new_qform)
            nib.save(ct_nii_img, "/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/med_res_leftb/med_res_leftb.nii.gz")


    data = []
    data.append(["high_res_wb.nii.gz", "/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_res/spatial_encoding_wb_high_res.npy"])
    data.append(["med_res_wb.nii.gz", "/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_res/spatial_encoding_wb_med_res.npy"])
    data.append(["low_res_wb.nii.gz", "/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_res/spatial_encoding_wb_low_res.npy"])
    data.append(["med_res_ub.nii.gz", "/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_res/spatial_encoding_ub_med_res.npy"])
    data.append(["med_res_leftb.nii.gz", "/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_res/spatial_encoding_lb_med_res.npy"])

    subjects_df = pd.DataFrame(data, columns=["subject","spatial"])
    subjects_df.to_csv("/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/spatial_locations.csv")

if __name__ == "__main__":
    main()
