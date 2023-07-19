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
spec_binery[spec_array > 0] = 6



combine_array = np.zeros_like(spec_array)
combine_array = spec_binery + lobe_array

V_lobe_one = (sum(combine_array[combine_array==7])/7)*V
V_lobe_two = (sum(combine_array[combine_array==8])/8)*V
V_lobe_three = (sum(combine_array[combine_array==9])/9)*V
V_lobe_four = (sum(combine_array[combine_array==10])/10)*V
V_lobe_five = (sum(combine_array[combine_array==11])/11)*V

print(V_lobe_one, V_lobe_two, V_lobe_three, V_lobe_four, V_lobe_five)



