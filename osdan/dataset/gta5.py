import numpy as np
import os
from PIL import Image
import os.path as osp
from osdan.dataset.base_dataset import BaseDataset


class GTA5DataSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)

        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.resize = crop_size
        # print(self.resize)
        self.crop_size = (1024, 512)

    def get_metadata(self, name):
        img_file = self.root / 'images' / name
        label_file = self.root / 'labels' / name
        return img_file, label_file

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        image = Image.fromarray(np.uint8(image))
        label = self.get_labels(label_file)
        label = Image.fromarray(np.uint8(label))

        # (left, upper, right, lower)
        left = self.resize[0] - self.crop_size[0]
        upper = self.resize[1] - self.crop_size[1]
        left = np.random.randint(0, high=left)
        upper = np.random.randint(0, high=upper)
        right = left + self.crop_size[0]
        lower = upper + self.crop_size[1]

        image = image.crop((left, upper, right, lower))
        image = np.asarray(image, np.float32)

        label = label.crop((left, upper, right, lower))
        label = np.asarray(label, np.float32)
        # print(label.shape)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        return image.copy(), label_copy.copy(), np.array(image.shape), name
