def caculate_volume(ct_itk, pred_array):
    spacing = ct_itk.GetSpacing()
    volume = float(spacing[0]) * float(spacing[1]) * float(spacing[2])

    lobe1 = sum(pred_array[pred_array == 1])
    lobe2 = sum(pred_array[pred_array == 2])
    lobe3 = sum(pred_array[pred_array == 3])
    lobe4 = sum(pred_array[pred_array == 4])
    lobe5 = sum(pred_array[pred_array == 5])

    return round(lobe1 * volume), round(lobe2 * volume/2), round(lobe3 * volume/3), round(lobe4 * volume/4), round(lobe5 * volume/5)
import SimpleITK as sitk
import numpy as np
ct_itk = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\lung018_isotrop_LVBAC.nii.gz'
pre = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\Lung_CT108\Lung_CT108\pre_iso_LVB\lung018_isotrop_LVBAC.nii.gz'
ct = sitk.ReadImage(ct_itk)
pred = sitk.ReadImage(pre)
pred_arr = sitk.GetArrayFromImage(pred)

a,b,c,d,e = caculate_volume(ct,pred_arr)
print(a,b,c,d,e)
