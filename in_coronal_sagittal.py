import cv2
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage


def coronal(img):
    cor = np.zeros((img.shape[1], img.shape[0], img.shape[1]))
    for i in range(img.shape[2]):
        # img[i] = (cv2.flip(cv2.rotate(ct_array[i], calib_code), 1))
        cor[i] = img[:, i, :]
        # cor[i] = cv2.flip(cor[i], 0)

    return cor


def sagittal(img):
    sag = np.zeros((img.shape[1], img.shape[0], img.shape[1]))
    for i in range(img.shape[2]):
        # img[i] = (cv2.flip(cv2.rotate(ct_array[i], calib_code), 1))
        sag[i] = img[:, :, i]
        # sag[i] = cv2.flip(cv2.flip(sag[i],0),1)

    return sag


def coronal_overlay(img):
    cor = np.zeros((img.shape[1], img.shape[0], img.shape[1], 3))
    for i in range(img.shape[2]):
        # img[i] = (cv2.flip(cv2.rotate(ct_array[i], calib_code), 1))
        cor[i] = img[:, i, :]
        # cor[i] = cv2.flip(cor[i], 0)

    return cor


def sagittal_overlay(img):
    sag = np.zeros((img.shape[1], img.shape[0], img.shape[1], 3))
    for i in range(img.shape[2]):
        # img[i] = (cv2.flip(cv2.rotate(ct_array[i], calib_code), 1))
        sag[i] = img[:, :, i]
        # sag[i] = cv2.flip(cv2.flip(sag[i],0),1)

    return sag


def sagittal_rotate3D(img_numpy):
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    all_axes = [(1, 2)]
    angle = -90
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return ndimage.rotate(img_numpy, angle, axes=axes)


def coronal_rotate3D(img_numpy):
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    all_axes = [(1, 2)]
    angle = -90
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return ndimage.rotate(img_numpy, angle, axes=axes)


def sagittal_rotate3D_overlay(img_numpy):
    all_axes = [(1, 2)]
    angle = -90
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return ndimage.rotate(img_numpy, angle, axes=axes)


def coronal_rotate3D_overlay(img_numpy):
    all_axes = [(1, 2)]
    angle = -90
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return ndimage.rotate(img_numpy, angle, axes=axes)


def resize_data_volume_by_scale(data, scale):
    """
    Resize the data based on the provided scale
    """
    scale_list = [1, scale, scale]
    return ndimage.zoom(data, scale_list, order=0)


# def coronal_rotate3D(img_numpy):
#
#    assert img_numpy.ndim == 3, "provide a 3d numpy array"
#    all_axes = [(1, 0)]
#    angle = 90
#    axes_random_id = np.random.randint(low=0, high=len(all_axes))
#    axes = all_axes[axes_random_id]
#    return ndimage.rotate(img_numpy, angle, axes=axes)


ct_path = r'/home/avitech-pc-5500/Nam/2anh/i/lung018_isotrop_LVBAC.nii.gz'
per_path = r'/home/avitech-pc-5500/Nam/lung018_CTDTPASPECTCT_LVBAC_FER.nii.gz'
save = r'/home/avitech-pc-5500/Nam/2anh/zoom'
per = sitk.ReadImage(per_path)
spacing = per.GetSpacing()
origin = per.GetOrigin()
Direction = per.GetDirection()
print(spacing)
print(origin)
print(Direction)

ct = sitk.ReadImage(ct_path)
print(ct.GetSpacing())
print(ct.GetOrigin())
print(ct.GetDirection())
ct_array = sitk.GetArrayFromImage(ct)
#
# # b= coronal(ct_array)
c = resize_data_volume_by_scale(ct_array, 2)
#
anh = sitk.GetImageFromArray(c)

# anh.SetSpacing(spacing)
# anh.SetDirection(Direction)
# anh.SetOrigin(origin)
# print(anh.GetDirection())
# print(anh.GetOrigin())
# print(anh.GetSpacing())
#
# # sitk.WriteImage(anh,fileName = save+ '/xoayanh_coronal_90.nii.gz', useCompression=True)
sitk.WriteImage(anh, fileName=save + '/zoomanh2lan_isotro.nii.gz', useCompression=True)