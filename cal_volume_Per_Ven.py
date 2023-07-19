import SimpleITK as sitk
import os
import numpy as np


def cal_percent_Per_Ven(lung_lobe, per_ven_arr):
    spacing = lung_lobe.GetSpacing()
    Volume_voxel = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
    lobe_arr =  sitk.GetArrayFromImage(lung_lobe)
    per_ven_newarr = np.zeros_like(per_ven_arr)
    per_ven_newarr[np.logical_and(per_ven_arr > 0, lobe_arr == 1)] = 1
    per_ven_newarr[np.logical_and(per_ven_arr > 0, lobe_arr == 2)] = 2
    per_ven_newarr[np.logical_and(per_ven_arr > 0, lobe_arr == 3)] = 3
    per_ven_newarr[np.logical_and(per_ven_arr > 0, lobe_arr == 4)] = 4
    per_ven_newarr[np.logical_and(per_ven_arr > 0, lobe_arr == 5)] = 5

    lobe_one = (sum(per_ven_newarr[per_ven_newarr == 1])) * Volume_voxel
    lobe_two = (sum(per_ven_newarr[per_ven_newarr == 2]) / 2) * Volume_voxel
    lobe_three = (sum(per_ven_newarr[per_ven_newarr == 3]) / 3) * Volume_voxel
    lobe_four = (sum(per_ven_newarr[per_ven_newarr == 4]) / 4) * Volume_voxel
    lobe_five = (sum(per_ven_newarr[per_ven_newarr == 5]) / 5) * Volume_voxel

    lobe_all = lobe_one + lobe_two + lobe_three + lobe_four + lobe_five

    return lobe_one/lobe_all,lobe_two/lobe_all,lobe_three/lobe_all,lobe_four/lobe_all,lobe_five/lobe_all


# spect_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\lung018_CTDTPASPECTCT_LVBAC_FER.nii.gz'
# lobe_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\Lung_CT108\Lung_CT108\pre_iso_LVB\lung018_isotrop_LVBAC.nii.gz'

# spec = sitk.ReadImage(spect_path)
# lobe = sitk.ReadImage(lobe_path)
# spec_array = sitk.GetArrayFromImage(spec)

# a,b,c,d,e = cal_percent_Per_Ven(lobe, spec_array)
# print(a,b,c,d,e)


