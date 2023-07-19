import os
import copy
import collections
from time import time
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology
# from cal_metric import Metirc
from test_metric import Metric


# dice_intersection = 0.0
# dice_union = 0.0

file_name = []
# time_pre_case = []

lung_score = collections.OrderedDict()
lung_score['dice'] = []
# lung_score['jaccard'] = []
# lung_score['voe'] = []
lung_score['fnr'] = []
lung_score['fpr'] = []
# lung_score['assd'] = []
# lung_score['rmsd'] = []
lung_score['msd'] = []

GT_dir = r'E:\ve_anh_phoi\ground_truth_chuanhoa'
seg_dir =r'E:\ve_anh_phoi\pre_imagesTr_gopnhan'

for file_index, file in tqdm(enumerate(os.listdir(GT_dir))):
    file_name.append(file)

    GT = sitk.ReadImage(os.path.join(GT_dir, file))
    GT_array = sitk.GetArrayFromImage(GT)
    # GT_array[GT_array < 2] = 0

    seg = sitk.ReadImage(os.path.join(seg_dir, file))
    seg_array = sitk.GetArrayFromImage(seg)
    # seg_array[seg_array < 2] = 0

    lung_metric = Metric(GT_array, seg_array, GT.GetSpacing(),num_classes=1)

    lung_score['dice'].append(lung_metric.get_dice_coefficient())
    # lung_score['jaccard'].append(lung_metric.get_jaccard_index())
    # lung_score['voe'].append(lung_metric.get_VOE())
    lung_score['fnr'].append(lung_metric.get_FNR())
    lung_score['fpr'].append(lung_metric.get_FPR())
    # lung_score['assd'].append(lung_metric.get_ASSD())
    # lung_score['rmsd'].append(lung_metric.get_RMSD())
    lung_score['msd'].append(lung_metric.get_MSD())

    # dice_intersection += tumor_metric.get_dice_coefficient()[1]
    # dice_union += tumor_metric.get_dice_coefficient()[2]


lung_data = pd.DataFrame(lung_score, index=file_name)

lung_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(lung_data.columns))
lung_statistics.loc['mean'] = lung_data.mean()
lung_statistics.loc['std'] = lung_data.std()
lung_statistics.loc['min'] = lung_data.min()
lung_statistics.loc['max'] = lung_data.max()

writer = pd.ExcelWriter(r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\lungmask\checkpoint\result_val\10_1_danhgiamean_isotrop_55CT_108.xlsx')
lung_data.to_excel(writer,'lung')
lung_statistics.to_excel(writer, 'lung_statistics')
writer.save()

# print('dice global:', dice_intersection / dice_union)