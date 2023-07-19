import math

import numpy as np
import scipy.spatial as spatial
import scipy.ndimage.morphology as morphology


import matplotlib.pyplot as plt

def surfd(input1, input2, spacing=1, connectivity=1):
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    input1_erosion = morphology.binary_erosion(input_1, conn).astype(int)
    # show_array(input1_erosion)
    input2_erosion = morphology.binary_erosion(input_2, conn).astype(int)
    # show_array(input2_erosion)


    S = input_1 - input1_erosion
    Sprime = input_2 - input2_erosion
    S_sum = np.sum(S)
    Sprime_sum = np.sum(Sprime)
    # show_array(S)
    # show_array(Sprime)



    S = S.astype(np.bool)
    Sprime = Sprime.astype(np.bool)

    dta = morphology.distance_transform_edt(~S, spacing)
    dtb = morphology.distance_transform_edt(~Sprime, spacing)
    # show_array(dta)
    # show_array(dtb)

    dta_multiply_Sprime = dta[Sprime != 0]  # 365703
    dtb_multiply_S = dtb[S != 0] # 428467
    # show_array(dta_multiply_Sprime)
    # show_array(dtb_multiply_S)
    count11 = np.count_nonzero(dta_multiply_Sprime)
    count22 = np.count_nonzero(dtb_multiply_S)

    ds1 = np.ravel(dta[Sprime != 0])
    ds2 = np.ravel(dtb[S != 0])

    count1 = np.count_nonzero(ds1)
    count2 = np.count_nonzero(ds2)

    sds = np.concatenate([ds1, ds2])

    surface_distance = ds1

    msd = surface_distance.mean()
    rms = np.sqrt((surface_distance ** 2).mean())
    hd = surface_distance.max()
    hd95 = np.percentile(surface_distance, 95)
    median = np.median(surface_distance)
    std = np.std(surface_distance)

    surface_distance = ds2

    msd = surface_distance.mean()
    rms = np.sqrt((surface_distance ** 2).mean())
    hd = surface_distance.max()
    hd95 = np.percentile(surface_distance, 95)
    median = np.median(surface_distance)
    std = np.std(surface_distance)

    return sds