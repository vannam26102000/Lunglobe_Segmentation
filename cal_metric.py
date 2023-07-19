import math

import numpy as np
import scipy.spatial as spatial
import scipy.ndimage.morphology as morphology


class Metirc():

    def __init__(self, real_mask, pred_mask, voxel_spacing, num_classes):

        self.num_classes = num_classes
        self.real_mask = real_mask
        self.pred_mask = pred_mask
        self.voxel_sapcing = voxel_spacing
        self.real_mask_surface_pts = self.get_surface(real_mask, voxel_spacing)
        self.pred_mask_surface_pts = self.get_surface(pred_mask, voxel_spacing)

        self.real2pred_nn = self.get_real2pred_nn()
        self.pred2real_nn = self.get_pred2real_nn()

    def get_surface(self, mask, voxel_spacing):



        kernel = morphology.generate_binary_structure(3, 2)
        surface = morphology.binary_erosion(mask, kernel) ^ mask

        surface_pts = surface.nonzero()

        surface_pts = np.array(list(zip(surface_pts[0], surface_pts[1], surface_pts[2])))


        return surface_pts * np.array(self.voxel_sapcing[::-1]).reshape(1, 3)

    def get_pred2real_nn(self):


        tree = spatial.cKDTree(self.real_mask_surface_pts)
        nn, _ = tree.query(self.pred_mask_surface_pts)

        return nn

    def get_real2pred_nn(self):

        tree = spatial.cKDTree(self.pred_mask_surface_pts)
        nn, _ = tree.query(self.real_mask_surface_pts)

        return nn

    def get_dice_coefficient(self):

        dice = []
        # outputs = outputs.squeeze(1)
        for num in range(1, self.num_classes + 1):
            # intersection = ((outputs == num) * (labels == num)).sum()
            # union = (outputs == num).sum() + (labels == num).sum() - intersection

            epsilon = 1e-15
            intersection = ((self.pred_mask == num) * (self.real_mask == num)).sum()
            union = (self.real_mask == num).sum() + (self.pred_mask == num).sum()
            if union == 0:
                dice.append(float('nan'))  # if there is no class in ground truth, do not include in evaluation
            else:
                dice.append(2 * intersection / (union + epsilon))
                # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

        # return np.nanmean(dice)
        return dice

    def get_jaccard_index(self):
        jaccard = []
        # outputs = outputs.squeeze(1)
        for num in range(1, self.num_classes + 1):
            # intersection = ((outputs == num) * (labels == num)).sum()
            # union = (outputs == num).sum() + (labels == num).sum() - intersection

            epsilon = 1e-15
            intersection = ((self.pred_mask == num) * (self.real_mask == num)).sum()
            union = (self.real_mask == num).sum() + (self.pred_mask == num).sum()
            if union == 0:
                jaccard.append(float('nan'))  # if there is no class in ground truth, do not include in evaluation
            else:
                jaccard.append(intersection / (union - intersection + epsilon))
                # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

        return np.nanmean(jaccard)
        # return jaccard

    def get_VOE(self):

        return 1 - self.get_jaccard_index()

    def get_RVD(self):


        return float(self.pred_mask.sum() - self.real_mask.sum()) / float(self.real_mask.sum())

    def get_FNR(self):
        fnr = []
        for num in range(1, self.num_classes + 1):
            fn = (self.real_mask == num).sum() - ((self.real_mask== num) * (self.pred_mask== num)).sum()
            union = ((self.real_mask== num) | (self.pred_mask== num)).sum()
            if union == 0:
                fnr.append(float('nan'))  # if there is no class in ground truth, do not include in evaluation
            else:
                fnr.append(fn/union)
        return np.nanmean(fnr)

    def get_FPR(self):
        fpr = []
        for num in range(1, self.num_classes + 1):
            fp = (self.pred_mask== num).sum() - ((self.real_mask == num)* (self.pred_mask == num)).sum()
            union = ((self.real_mask== num) | (self.pred_mask==num)).sum()
            if union == 0:
                fpr.append(float('nan'))  # if there is no class in ground truth, do not include in evaluation
            else:
                fpr.append(fp / union)
        return np.nanmean(fpr)
    def get_ASSD(self):
        assd = []
        for num in range(1, self.num_classes + 1):
            a = (self.pred2real_nn== num).sum() + (self.real2pred_nn== num).sum()
            b = (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0])
            assd.append(a/b)
        return np.nanmean(assd)

        # return (self.pred2real_nn.sum() + self.real2pred_nn.sum()) / \
        #        (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0])

    def get_RMSD(self):
        rmsd = []
        for num in range(1, self.num_classes + 1):
            a = np.power(self.pred2real_nn == num, 2).sum() + np.power(self.real2pred_nn == num, 2).sum()
            b = (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0])
            rmsd.append( math.sqrt(a/b))
        return np.nanmean(rmsd)

        # return math.sqrt((np.power(self.pred2real_nn, 2).sum() + np.power(self.real2pred_nn, 2).sum()) /
        #                  (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0]))

    def get_MSD(self):

        return max(self.pred2real_nn.max(), self.real2pred_nn.max())
