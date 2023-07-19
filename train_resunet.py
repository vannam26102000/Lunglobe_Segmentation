import torch
import torch.utils.data as dataset_torch
from abc import ABC
import os

import time
import numpy as np
from lungmask import utils
from lungmask import resunet
from torchvision import transforms

from lungmask import mask
import SimpleITK as sitk

import os

data_extensions = [
    '.nii.gz',
]
roi_extensions = [
    '.roi.nii.gz',
]


def is_image_file(filename, mode='data'):
    if mode == 'roi':
        return any(filename.endswith(extension) for extension in roi_extensions)
    elif mode == 'data':
        if not 'roi' in filename and \
                any(filename.endswith(extension) for extension in data_extensions):
            return True
        else:
            return False
    else:
        raise ValueError('Undefined mode %s while reading data' % mode)


def make_dataset(dir, max_dataset_size=float("inf"), mode='data'):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        # return of os.walk: root dir, folders, files
        for fname in fnames:
            if is_image_file(fname, mode):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class BaseDataset(dataset_torch.Dataset, ABC):
    def __init__(self, dir):
        """
        dir: File directory.
        """
        self.dir = dir
        self.img_list = sorted(make_dataset(dir, mode='data'))
        self.roi_list = sorted(make_dataset(dir, mode='roi'))

        self.A_size = len(self.img_list)  # get the size of dataset
        self.B_size = len(self.roi_list)  # get the size of roi-set

        assert (self.A_size == self.B_size)
        if self.A_size == 0:
            raise (RuntimeError("Found 0 datafiles in: " + dir))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.img_list[index]  # % self.A_size]  # make sure index is within then range
        B_path = self.roi_list[index]  # % self.B_size]  # make sure index is within then range

        # apply image transformation
        A = sitk.ReadImage(A_path)  # data
        B = sitk.ReadImage(B_path, sitk.sitkUInt8)  # roi

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


class LungLabelsDS_inf(dataset_torch.Dataset):
    def __init__(self, ds, lb):
        self.dataset = ds
        self.label = lb

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx, None, :, :], self.label[idx, None, :, :]


def preprocess(img_data, roi_data):
    img_raw = sitk.GetArrayFromImage(img_data)
    roi_raw = sitk.GetArrayFromImage(roi_data)
    directions = np.asarray(img_data.GetDirection())
    if len(directions) == 9:
        img_raw = np.flip(img_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])
        roi_raw = np.flip(roi_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])

    tvolslices, labelslices, xnew_box = utils.preprocess(img_raw, label=roi_raw, resolution=[256, 256])

    tvolslices = np.divide((tvolslices + 1024), 1624)

    torch_ds_val = LungLabelsDS_inf(tvolslices, xnew_box)

    return torch.utils.data.DataLoader(torch_ds_val, batch_size=16, shuffle=True, num_workers=1, pin_memory=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resunet.UNet(in_channels=1, n_classes=6, depth=5, wf=6, padding=False,batch_norm=False, up_mode='upconv', residual=True)
model.to(device)
model.train()

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
learning_rate_decay = [500, 750]

lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, learning_rate_decay)

dir = r'C:\Users\FR\PycharmProjects\Qtdesigner\Nam\data'
dataset = BaseDataset(dir)

total_iters = 0
torch.cuda.empty_cache()

for epoch in range(100):

    mean_loss = []
    lr_decay.step()

    for _, data in enumerate(dataset):

        dataloader_val = preprocess(data['A'], data['B'])
        epoch_loss = 0

        for step, (X, Y) in enumerate(dataloader_val):
            X = X.float().to(device)
            label = torch.tensor(Y[:, 0, :, :], dtype=torch.long).to(device)

            prediction = model(X)

            loss = criterion(prediction, label)

            mean_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #             if step % 5 == 0:
    #                 print('epoch:{}, step:{}, loss:{:.3f}'.format(epoch, step, loss.item()))

    mean_loss = sum(mean_loss) / len(mean_loss)

    print('epoch:{}, mean_loss:{:.3f}'.format(epoch, mean_loss))

    if epoch % 50 == 0 and epoch != 0:
        torch.save(model.state_dict(), './checkpoint/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))






