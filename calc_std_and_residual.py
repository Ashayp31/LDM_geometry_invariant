import nibabel as nib
import numpy as np
#
# original_image = nib.load("/nfs/home/apatel/Data/PET_Challenge/processed/CT_Generative_Experiment2/high_res_wb/high_res_wb.nii.gz")
# original_image = original_image.get_fdata()
# original_image = original_image[0]
#
#
# original_recon = nib.load("/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb/ddpm_small_CT_spatial_conditioning_vq512_8_cond_dim_4/output/sample_0_wb_low_res_super_resolved_25steps_96_base_dim.nii.gz")
# original_recon = original_recon.get_fdata()
#
# residual = original_image - original_recon
# residual = np.absolute(residual)
#
# for repeat_std in range(20):
#     sample = nib.load("/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb/ddpm_small_CT_spatial_conditioning_vq512_8_cond_dim_4/output/sample_" + str(repeat_std) + "_wb_med_res_super_resolved_25steps_96_base_dim_stdcalc.nii.gz")
#     sample = sample.get_fdata()
#     if repeat_std == 0:
#         total_image = np.expand_dims(sample,0)
#     else:
#         total_image = np.concatenate((total_image,np.expand_dims(sample,0)),axis=0)
#
# std_over_recons = np.std(total_image, axis=0, keepdims=False)
#
# new_qform = [[1.6, 0, 0, 0],
#              [0, 1.6, 0, 0],
#              [0, 0, 2.5, 0],
#              [0, 0, 0, 1]]
#
# sample_resid_nii = nib.Nifti1Image(residual, affine=new_qform)
# nib.save(sample_resid_nii, "/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb/ddpm_small_CT_spatial_conditioning_vq512_8_cond_dim_4/output/high_res_super_resolution_residual.nii.gz")
#
# std_over_recons_nii = nib.Nifti1Image(std_over_recons, affine=new_qform)
# nib.save(std_over_recons_nii, "/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb/ddpm_small_CT_spatial_conditioning_vq512_8_cond_dim_4/output/high_res_super_resolution_std.nii.gz")




original_image = nib.load("/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb/ddpm_small_CT_spatial_conditioning_vq512_8_cond_dim_4/output/sample_original_sample_wb_lower_res_base_dim_192.nii.gz")
original_image = original_image.get_fdata()
original_image = original_image


original_recon = nib.load("/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb/ddpm_small_CT_spatial_conditioning_vq512_8_cond_dim_4/output/sample_0_wb_lower_res_super_resolved_25steps_96_result.nii.gz")
original_recon = original_recon.get_fdata()

residual = original_image - original_recon
residual = np.absolute(residual)

# for repeat_std in range(20):
#     sample = nib.load("/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb/ddpm_small_CT_spatial_conditioning_vq512_8_cond_dim_4/output/sample_" + str(repeat_std) + "_wb_lower_res_super_resolved_25steps_96_base_dim_stdcalc.nii.gz")
#
#     sample = sample.get_fdata()
#     if repeat_std == 0:
#         total_image = np.expand_dims(sample,0)
#     else:
#         total_image = np.concatenate((total_image,np.expand_dims(sample,0)),axis=0)
#
# std_over_recons = np.std(total_image, axis=0, keepdims=False)

new_qform = [[1.6, 0, 0, 0],
             [0, 1.6, 0, 0],
             [0, 0, 2.5, 0],
             [0, 0, 0, 1]]

sample_resid_nii = nib.Nifti1Image(residual, affine=new_qform)
nib.save(sample_resid_nii, "/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb/ddpm_small_CT_spatial_conditioning_vq512_8_cond_dim_4/output/med_res_super_resolution_residual.nii.gz")

# std_over_recons_nii = nib.Nifti1Image(std_over_recons, affine=new_qform)
# nib.save(std_over_recons_nii, "/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/DDPM_small_CT_wb/ddpm_small_CT_spatial_conditioning_vq512_8_cond_dim_4/output/med_res_super_resolution_std.nii.gz")
