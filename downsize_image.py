import cv2
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import os

def resize_data_volume_by_scale(data, scale):
   """
   Resize the data based on the provided scale
   """
   scale_list = [1,scale,scale]
   return ndimage.zoom(data, scale_list, order=0)

root_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\lung018_CTDTPASPECTCT_LVBAC_FER.nii.gz'
save_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\downsize_2lan'

# image_list = os.listdir(root_path)
#
# for image_file_name in image_list:
#
#     input_image = sitk.ReadImage(os.path.join(root_path, image_file_name))
#     input_arr = sitk.GetArrayFromImage(input_image)
#     input_downsize = resize_data_volume_by_scale(input_arr,0.5)
#
#     pred_itk = sitk.GetImageFromArray(input_downsize)
#     pred_itk.SetSpacing(input_image.GetSpacing())
#     pred_itk.SetOrigin(input_image.GetOrigin())
#     pred_itk.SetDirection(input_image.GetDirection())
#     sitk.WriteImage(pred_itk, os.path.join(save_path, image_file_name))

input_image = sitk.ReadImage(root_path)
input_arr = sitk.GetArrayFromImage(input_image)
input_downsize = resize_data_volume_by_scale(input_arr,2)

pred_itk = sitk.GetImageFromArray(input_downsize)
pred_itk.SetSpacing(input_image.GetSpacing())
pred_itk.SetOrigin(input_image.GetOrigin())
pred_itk.SetDirection(input_image.GetDirection())
sitk.WriteImage(pred_itk,fileName = save_path+ '/lung018_CTDTPASPECTCT_LVBAC_FER.nii.gz', useCompression=True)