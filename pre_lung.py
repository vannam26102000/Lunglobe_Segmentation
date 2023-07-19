
import SimpleITK as sitk
from lungmask import mask
import os

root_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\lungmask\lung_4.nii.gz'
save_path = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\lungmask\seg_file_8_5'
# # #
# # # image_list = os.listdir(root_path)
# #
# # # for image_file_name in image_list:
# # #
# # #     input_image = sitk.ReadImage(os.path.join(root_path, image_file_name))
# # #     # input_arr = sitk.GetArrayFromImage(input_image) - 1000
# # #     # input_image = sitk.GetImageFromArray(input_arr)
# # #
# # #     model = mask.get_model('unet','LTRCLobes')
# # #
# # #     # state_dict = torch.load(r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\lungmask\checkpoint\net4_LUNA_DECA_GAN.pth', map_location=torch.device('cpu'))
# # #     #
# # #     # model.load_state_dict(state_dict)
# # #     # model.eval()
# # #
# # #     segmentation = mask.apply(input_image, model)
# # #
# # #     pred_itk = sitk.GetImageFromArray(segmentation)
# # #     pred_itk.SetSpacing(input_image.GetSpacing())
# # #     pred_itk.SetOrigin(input_image.GetOrigin())
# # #     sitk.WriteImage(pred_itk, os.path.join(save_path, image_file_name))
# #
input_image = sitk.ReadImage(root_path)
# input_arr = sitk.GetArrayFromImage(input_image)-1000
# input_image = sitk.GetImageFromArray(input_arr)

model = mask.get_model('unet','R231')
segmentation = mask.apply(input_image, model)

pred_itk = sitk.GetImageFromArray(segmentation)
pred_itk.SetSpacing(input_image.GetSpacing())
pred_itk.SetOrigin(input_image.GetOrigin())
sitk.WriteImage(pred_itk, fileName = save_path+ '/lung_4.nii.gz', useCompression=True)

#
# model = mask.get_model('unet','LTRCLobes')
# print(model)

