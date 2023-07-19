import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm:

data_dir = '/home/avitech-pc4/Nam/save/roi/'
data_dir_convert = '../data/roi_extensions/roi_extensions/'

    image_list = os.listdir(data_dir)

    for image_file_name in image_list:


    "    label_itk = sitk.ReadImage(os.path.join(data_dir, image_file_name))\n",
    "    \n",
    "    label_array = sitk.GetArrayFromImage(label_itk)\n",
    "    \n",
    "    image_shape = np.shape(label_array)\n",
    "    \n",
    "    save_array = np.zeros_like(label_array)\n",
    "    \n",
    "    for x_index in range(image_shape[0]):\n",
    "        for y_index in range(image_shape[1]):\n",
    "            for z_index in range(image_shape[2]):\n",
    "                if label_array[x_index, y_index, z_index] == 0:\n",
    "                    pass\n",
    "                else:\n",
    "                \n",
    "                    if label_array[x_index, y_index, z_index] == 8:\n",
    "                        save_array[x_index, y_index, z_index] = 2\n",
    "                    elif label_array[x_index, y_index, z_index] == 6:\n",
    "                        save_array[x_index, y_index, z_index] = 5\n",
    "                    elif label_array[x_index, y_index, z_index] == 5:\n",
    "                        save_array[x_index, y_index, z_index] = 4\n",
    "                    elif label_array[x_index, y_index, z_index] == 4:\n",
    "                        save_array[x_index, y_index, z_index] = 3\n",
    "                    elif label_array[x_index, y_index, z_index] == 7:\n",
    "                        save_array[x_index, y_index, z_index] = 1\n",
    "                    else:\n",
    "                        pass\n",
    "\n",
    "                \n",
    "    convert_label_itk = sitk.GetImageFromArray(save_array)\n",
    "    \n",
    "    convert_label_itk.SetSpacing(label_itk.GetSpacing())\n",
    "    convert_label_itk.SetOrigin(label_itk.GetOrigin())\n",
    "    \n",
    "    sitk.WriteImage(convert_label_itk, os.path.join(data_dir_convert, image_file_name))"