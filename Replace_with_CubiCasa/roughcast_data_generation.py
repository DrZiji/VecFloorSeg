
import pickle
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from numpy import genfromtxt
from xml.dom import minidom
from PIL import Image, ImagePalette
from floortrans.loaders.house import House


class FloorplanSVG_Roughcast(Dataset):
    def __init__(self, data_folder, data_file, is_transform=True,
                 augmentations=None, img_norm=True, format='txt',
                 original_size=False, lmdb_folder='cubi_lmdb/'):
        self.img_norm = img_norm
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.get_data = None
        self.original_size = original_size
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'

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
        # Getting labels for segmentation and heatmaps
        _ = House_Roughcast(
            self.data_folder + self.folders[index] + self.svg_file_name
        )

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


class House_Roughcast:
    def __init__(self, path):
        self.retain_and_write_1(path)


    def retain_and_write_2(self, path):
        tree = minidom.parse(path)
        for elem in tree.getElementsByTagName('g'):
            elem_class = elem.getAttribute("class")
            elem_id = elem.getAttribute("id")
            if elem_class == 'Floor':
                if elem.hasAttribute("style"):
                    elem.setAttribute('style', 'display: inherit')
            elif 'Floor-' in elem_class:
                for child in elem.childNodes:
                    if child.localName == 'g':
                        class_name = child.getAttribute('class')
                        if 'Wall' in class_name or 'Railing' in class_name or 'Stairs' in class_name:
                            pass
                        else:
                            elem.removeChild(child)
                            print(class_name)
                    else:
                        print(child.localName)

        with open(path.replace('model', 'model_roughcast'), 'w', encoding='utf-8') as f:
            tree.writexml(f)



    def retain_and_write_1(self, path):
        tree = minidom.parse(path)
        for elem in tree.getElementsByTagName('g'):
            elem_class = elem.getAttribute("class")
            elem_id = elem.getAttribute("id")
            if elem_class == 'Floor':
                if elem.hasAttribute("style"):
                    elem.setAttribute('style', 'display: inherit')
            elif 'Floor' in elem_class or 'Door' in elem_class or 'Window' in elem_class:
                pass
            elif 'Wall' in elem_class or 'Railing' in elem_class:
                pass
            elif 'Stairs' in elem_class or 'Flight' in elem_class or 'WalkinLine' in elem_class:
                pass
            elif 'Winding' in elem_class or 'Panel' in elem_class or 'Steps' in elem_class:
                pass
            elif elem_id == 'Model':
                pass
            else:
                elem.parentNode.removeChild(elem)

        with open(path.replace('model', 'model_roughcast'), 'w', encoding='utf-8') as f:
            tree.writexml(f)



    def remove_and_write(self, path):
        tree = minidom.parse(path)
        for elem in tree.getElementsByTagName('g'):
            # 找到包含FixedFurniture的父节点，删掉
            # print(elem.getAttribute('class'))
            if elem.getAttribute("class") == 'Floor':
                if elem.hasAttribute("style"):
                    elem.setAttribute('style', 'display: inherit')
            for child in elem.childNodes:
                if child.localName == 'g':
                    class_name = child.getAttribute('class')
                    if 'FixedFurniture' in class_name:
                        elem.removeChild(child)
                    elif "Space" in class_name:
                        elem.removeChild(child)
                    elif "Sign" in class_name or "Bench" in class_name:
                        elem.removeChild(child)
                    else:
                        print(child.getAttribute('class'))
                else:
                    pass

        with open(path.replace('model', 'model_roughcast'), 'w', encoding='utf-8') as f:
            tree.writexml(f)


if __name__ == "__main__":
    """
    目标：将furnished floorplans变成roughcast floorplans
    方式，创建一个新的svg文件，将对应的attribute 写入
    """
    from tqdm import tqdm
    d = FloorplanSVG_Roughcast(
        data_folder="E:\\Datasets\\cubicasa5k",
        data_file='\\train.txt',
        is_transform=False
    )
    # multiple process
    import multiprocessing
    from multiprocessing import Pool
    from tqdm import tqdm
    import pickle

    cpuCount = multiprocessing.cpu_count() - 4
    pool = Pool(cpuCount)

    idxes = list(range(0, len(d)))
    for res in tqdm(pool.imap_unordered(d.get_txt, idxes), total=len(idxes)):
        pass

    # # single process
    # idxes = list(range(0, len(d)))
    # for idx in tqdm(idxes):
    #     d.get_txt(idx)
