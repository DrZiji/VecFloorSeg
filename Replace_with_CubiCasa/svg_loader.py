# import lmdb
import pickle
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from numpy import genfromtxt
from floortrans.loaders.house import House
from PIL import Image, ImagePalette


class FloorplanSVG(Dataset):
    def __init__(self, data_folder, data_file, is_transform=True,
                 augmentations=None, img_norm=True, format='txt',
                 original_size=False, lmdb_folder='cubi_lmdb/'):
        self.img_norm = img_norm
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.get_data = None
        self.original_size = original_size
        self.image_file_name = 'svgImg_roughcast.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model_roughcast.svg'
        # self.palette = ImagePalette.random()
        imgPle = Image.open("E:\\Datasets\\cubicasa5k\\high_quality\\19\\icon.png")
        self.palette = imgPle.getpalette()
        self.palette[50] = 255  # 和outdoor类别(palette第3-5的值)做区分
        self.palette[35] = 160  # 和kitchen(palette第12-14的值)做区分

        if format == 'txt':
            self.get_data = self.get_txt
        if format == 'lmdb':
            self.lmdb = lmdb.open(data_folder + lmdb_folder, readonly=True,
                                  max_readers=8, lock=False,
                                  readahead=True, meminit=False)
            self.get_data = self.get_lmdb
            self.is_transform = False

        self.data_folder = data_folder
        # Load txt file to list
        excludeListTrain = [
            '\\high_quality_architectural\\2003\\',
            '\\high_quality_architectural\\2565\\',
            '\\high_quality_architectural\\6143\\',
            '\\high_quality_architectural\\10074\\',
            '\\high_quality_architectural\\10754\\',
            '\\high_quality_architectural\\10769\\',
            '\\high_quality_architectural\\14611\\',
            '\\high_quality\\7092\\',
            '\\high_quality\\1692\\',

            'high_quality_architectural\\10',  # img does not match label
        ]
        folders = genfromtxt(data_folder + data_file, dtype='str')
        self.folders = []
        for x in folders:
            x = x.replace('/', '\\')
            if x not in excludeListTrain:
                self.folders.append(x)

    def __len__(self):
        """__len__"""
        return len(self.folders)

    def __getitem__(self, index):
        sample = self.get_data(index)

        if self.augmentations is not None:
            sample = self.augmentations(sample)

        if self.is_transform:
            sample = self.transform(sample)

        return sample

    def get_txt(self, index):

        imgFplan = Image.open(self.data_folder + self.folders[index] + self.image_file_name)
        arrFplan = np.array(imgFplan)
        fplan = arrFplan[:, :, :3].copy()
        fplan[arrFplan[:, :, 3] == 0, :] = 255
        height, width, nchannel = fplan.shape

        # Getting labels for segmentation and heatmaps
        house = House(self.data_folder + self.folders[index] + self.svg_file_name, height, width)
        # Combining them to one numpy tensor
        labelArray = house.get_segmentation_tensor().astype(np.uint8)

        if np.max(labelArray[0]) >= 19 or np.max(labelArray[1]) >= 11:
            print(self.data_folder + self.folders[index])

        """将label拆分开，分别保存成wallLabel和iconLabel，用于DataRasterization.py"""
        wallLabel = Image.fromarray(labelArray[0], mode='P')
        iconLabel = Image.fromarray(labelArray[1], mode='P')
        wallLabel.putpalette(self.palette)
        iconLabel.putpalette(self.palette)

        wallLabel.save(self.data_folder + self.folders[index] + 'wall_svg.png')
        iconLabel.save(self.data_folder + self.folders[index] + 'icon_svg.png')


    def get_lmdb(self, index):
        key = self.folders[index].encode()
        with self.lmdb.begin(write=False) as f:
            data = f.get(key)

        sample = pickle.loads(data)
        return sample

    def transform(self, sample):
        fplan = sample['image']
        # Normalization values to range -1 and 1
        fplan = 2 * (fplan / 255.0) - 1

        sample['image'] = fplan

        return sample


if __name__ == "__main__":

    from tqdm import tqdm

    d = FloorplanSVG(
        data_folder="E:\\Datasets\\cubicasa5k",
        data_file='\\exclude_train_1.txt',
        is_transform=False
    )
    import multiprocessing
    from multiprocessing import Pool
    from tqdm import tqdm
    import pickle

    cpuCount = multiprocessing.cpu_count() - 4
    pool = Pool(cpuCount)

    idxes = list(range(0, len(d)))
    for res in tqdm(pool.imap_unordered(d.get_txt, idxes), total=len(idxes)):
        pass