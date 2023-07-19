import SimpleITK as sitk
import os
import numpy as np

ct_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\seg1anh\lung018_isotrop_LVBAC.nii.gz'
spect_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\lung018_CTDTPASPECTCT_LVBAC_FER.nii.gz'
lobe_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\Lung_CT108\Lung_CT108\pre_iso_LVB\lung018_isotrop_LVBAC.nii.gz'

def volume(ct_sitk):
    spac = ct_sitk.GetSpacing()
    vl_ct = spac[0]*spac[1]*spac[2]
    return vl_ct

ct = sitk.ReadImage(ct_path)
spec = sitk.ReadImage(spect_path)
lobe = sitk.ReadImage(lobe_path)

V = volume(ct)

ct_array = sitk.GetArrayFromImage(ct)
spec_array = sitk.GetArrayFromImage(spec)
lobe_array = sitk.GetArrayFromImage(lobe)

spec_binery = np.zeros_like(spec_array)
spec_binery[np.logical_and(spec_array > 0, lobe_array == 1)] = 1
spec_binery[np.logical_and(spec_array > 0, lobe_array == 2)] = 2
spec_binery[np.logical_and(spec_array > 0, lobe_array == 3)] = 3
spec_binery[np.logical_and(spec_array > 0, lobe_array == 4)] = 4
spec_binery[np.logical_and(spec_array > 0, lobe_array == 5)] = 5

V_lobe_one = (sum(spec_binery[spec_binery==1]))*V
V_lobe_two = (sum(spec_binery[spec_binery==2])/2)*V
V_lobe_three = (sum(spec_binery[spec_binery==3])/3)*V
V_lobe_four = (sum(spec_binery[spec_binery==4])/4)*V
V_lobe_five = (sum(spec_binery[spec_binery==5])/5)*V

print(V_lobe_one, V_lobe_two, V_lobe_three, V_lobe_four, V_lobe_five)



