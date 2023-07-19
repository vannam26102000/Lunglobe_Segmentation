import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
# A = np.array([[1, 2, 3], [3, 4, 5]])
#
# A[A != 3] = 1
# print(A.shape)

def cal_percent_per(lung_label, image_per):
    image_per[image_per != 0]=6
    for x in range (0, image_per.shape[0]):
        for y in range(0,image_per.shape[1]):
            for z in range(0, image_per.shape[2]):
                lung_label[x][y][z]= lung_label[x][y][z] + image_per[x][y][z]
    return lung_label
def resize_data_volume_by_scale(data, scale):
   """
   Resize the data based on the provided scale
   """
   scale_list = [1,scale,scale]
   return ndimage.zoom(data, scale_list, order=0)
lung_label = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\seg1anh\xemthu\lung018_isotrop_LVBAC.nii.gz'
image_per = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\lung018_CTDTPASPECTCT_LVBAC_FER.nii.gz'
save_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\seg1anh\p'
lung_label= sitk.ReadImage(lung_label)


image_per = sitk.ReadImage(image_per)
image_per.SetOrigin(lung_label.GetOrigin())
image_per.SetDirection(lung_label.GetDirection())
image_per.SetSpacing(lung_label.GetSpacing())

lung_label1 = sitk.GetArrayFromImage(lung_label)
# print(lung_label)
image_per1 = sitk.GetArrayFromImage(image_per)


a= cal_percent_per(lung_label1,image_per1)
b= sitk.GetImageFromArray(a)
b.SetSpacing(lung_label.GetSpacing())
b.SetOrigin(lung_label.GetOrigin())
b.SetDirection(lung_label.GetDirection())
# # sitk.WriteImage(b, os.path.join(save_path, image_file_name))
sitk.WriteImage(b,fileName = save_path+ 'xemthu1234567.nii.gz', useCompression=True)


