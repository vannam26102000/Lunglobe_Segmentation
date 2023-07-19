# Python program to demonstrate erosion and
# dilation of images.
import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
# Reading the input image
# src_lung = '/home/avitech-pc-5500/Nam/pre/lung018_isotrop_LVBAC.nii.gz'
# save = '/home/avitech-pc-5500/Nam/pre/showxoay'
# img = sitk.ReadImage(src_lung)
# img_arr = sitk.GetArrayFromImage(img)
# print(img_arr.shape)
 
# # Taking a matrix of size 5 as the kernel
# kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
# # The first parameter is the original image,
# # kernel is the matrix with which image is
# # convolved and third parameter is the number
# # of iterations, which will determine how much
# # you want to erode/dilate a given image.
# img_erosion = cv2.erode(img_arr, kernel, iterations=1)
# img_dilation = cv2.dilate(img_arr, kernel, iterations=1)
 
# # cv2.imshow('Input', img)
# cv2.imshow('Erosion', img_erosion)
# cv2.imshow('Dilation', img_dilation)
# a = np.zeros_like(img_arr)  
# a[img_arr == 1] = 1

# b = np.zeros_like(img_arr) 
# b[img_arr ==2] = 2

# c = np.zeros_like(img_arr) 
# c[img_arr ==3] = 3

# d = np.zeros_like(img_arr) 
# d[img_arr ==4] = 4

# e = np.zeros_like(img_arr) 
# e[img_arr ==5] = 5
def erosion_label(img_arr, kernel, iterations= 1):
    # img_arr = sitk.GetArrayFromImage(img)
    img_erosion = np.zeros_like(img_arr)
    img_narrow = np.zeros_like(img_arr)
    for i in range(img_arr.shape[0]):
        img_erosion[i] = cv2.erode(img_arr[i], kernel, iterations=1)
        img_narrow[i] = img_arr[i] - img_erosion[i]
    return img_narrow
# def erosion_five_label(img, kernel, iterations= 1):
#     img_arr = sitk.GetArrayFromImage(img)
#     img_erosion = np.zeros_like(img_arr)
#     img_narrow = np.zeros_like(img_arr)
#     for i in range(img_arr.shape[0]):
#         for j in range(1,6):
#             img_arr[i][img_arr[i] == j ] = j

#             img_erosion[i][j] = cv2.erode(img_arr[i][j], kernel, iterations=1)
#             img_narrow[i][j] = img_arr[i][j] - img_erosion[i][j]
#     return img_narrow
def erosion_five_label(img,kernel):
    img_arr = sitk.GetArrayFromImage(img)
    a = np.zeros_like(img_arr)  
    a[img_arr == 1] = 1

    b = np.zeros_like(img_arr) 
    b[img_arr ==2] = 2

    c = np.zeros_like(img_arr) 
    c[img_arr ==3] = 3

    d = np.zeros_like(img_arr) 
    d[img_arr ==4] = 4

    e = np.zeros_like(img_arr) 
    e[img_arr ==5] = 5
    a_narrow= erosion_label(a, kernel,iterations=1)
    b_narrow= erosion_label(b, kernel,iterations=1)
    c_narrow= erosion_label(c, kernel,iterations=1)
    d_narrow= erosion_label(d, kernel,iterations=1)
    e_narrow= erosion_label(e, kernel,iterations=1)

    img_narrow = a_narrow + b_narrow + c_narrow + d_narrow + e_narrow
    new_image = sitk.GetImageFromArray(img_narrow)
    new_image.SetSpacing(img.GetSpacing())
    new_image.SetDirection(img.GetDirection())
    new_image.SetOrigin(img.GetOrigin())

    return new_image



# b = sitk.LabelOverlay(image=src_Per, labelImage=new_image
#                                             opacity=0.1, backgroundValue = 0.7,
#                                             colormap = red + green + blue + yellow + glaucous)


# new_image_arr = erosion_five_label(img,kernel)
# new_image = sitk.GetImageFromArray(new_image_arr)
# sitk.WriteImage(new_image, fileName=save + '/xemthu12345.nii.gz', useCompression=True)
# plt.imshow(new_image_arr[100])
# plt.show()

