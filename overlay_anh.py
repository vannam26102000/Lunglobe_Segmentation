import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def overlay(Iso_CT, Per):
    Per.SetOrigin(Iso_CT.GetOrigin())
    Per.SetSpacing(Iso_CT.GetSpacing())
    Per.SetDirection(Iso_CT.GetDirection())
    Iso_CT_arr = sitk.GetArrayFromImage(Iso_CT)
    Per_arr = sitk.GetArrayFromImage(Per)

    src = Iso_CT
    src_norm = sitk.Cast(sitk.IntensityWindowing(src, windowMinimum=int(np.min(Iso_CT_arr)),
                                                 windowMaximum=int(np.max(Iso_CT_arr)),
                                                 outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)

    SP_overlay = np.zeros((128, 256, 256))
    SP_overlay_arr = np.zeros((128, 256, 256))
    Per_norm_arr = np.zeros((128, 256, 256, 3))
    Per_norm = sitk.Image(256, 256, 128, sitk.sitkUInt8)
    color_list = []
    # anh xa cương do den dai pixel 0-255
    Per_norm = sitk.Cast(sitk.IntensityWindowing(Per, windowMinimum=int(np.min(Per_arr)),
                                                 windowMaximum=int(np.max(Per_arr)),
                                                 outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
    Per_norm_arr = sitk.GetArrayFromImage(Per_norm)

    bin = np.linspace(0, 255, 256)

    hist = np.histogram(Per_norm_arr, bin)
    bin = np.delete(bin, np.where(hist == 0))

    color_map = plt.cm.hot((1 / 255) * bin)  # bản màu hot
    # print(color_map)
    color_list = np.ndarray.flatten(color_map[:, :3] * 255).astype(int).tolist()
    # print(color_list)
    SP_overlay = sitk.LabelOverlay(image=src_norm, labelImage=Per_norm, opacity=0.3, backgroundValue=255,
                                   colormap=color_list)

    SP_overlay_arr = sitk.GetArrayFromImage(SP_overlay)
    return SP_overlay_arr


# per_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\downsize_2lan\lung018_CTDTPASPECTCT_LVBAC_FER.nii.gz'
# iso_ct_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\downsize_2lan\seg1anhlung018_isotrop_LVBAC.nii.gz'

# Iso_CT = sitk.ReadImage(iso_ct_path)
# Per = sitk.ReadImage(per_path)
# SP_overlay_arr = np.zeros((128, 256, 256))

# SP_overlay_arr = overlay(Iso_CT,Per)

# plt.imshow(SP_overlay_arr[130])
# plt.show()