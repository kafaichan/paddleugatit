import paddle
from PIL import Image
import functools
import os
import os.path
from argparse import Namespace
import numpy as np
import random


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


class MyDatasetReader(object):
    def __init__(self, root, args):
        self.args = args
        self.root = root
        self.imgs = list()
        for root, _, fnames in sorted(os.walk(self.root)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    path = os.path.join(self.root, fname)
                    self.imgs.append(pil_loader(path))
        self.num_img = len(self.imgs)
        np.random.shuffle(self.imgs)
        self.idx = 0

    def get_batch(self, is_test=True):
        result = []
        for i in range(self.args.batch_size):
            result.append(self.img_transform(self.imgs[self.idx]))
            self.idx = (self.idx+1) % self.num_img
            if(self.idx == 0 and is_test): np.random.shuffle(self.imgs)
            else: break
        return np.array(result)

    def img_transform(self, img):
        if self.args.phase == 'train':
            return self.train_transform(img)
        elif self.args.phase == 'test':
            return self.test_transform(img)

    def train_transform(self, img):
        img = random_horizontal_flip(img)
        img = img.resize((self.args.img_size + 30, self.args.img_size+30))
        img = img_random_crop(img, self.args.img_size)
        img = np.array(img).astype('float32') / 255.0
        img -= np.array([0.5,0.5,0.5])
        img /= np.array([0.5, 0.5, 0.5])
        return img.transpose((2, 0, 1))

    def test_transform(self, img):
        img = img.resize((self.args.img_size, self.args.img_size))
        img = np.array(img).astype('float32') / 255.0
        img -= np.array([0.5,0.5,0.5])
        img /= np.array([0.5, 0.5, 0.5])
        return img.transpose((2, 0, 1))

def random_horizontal_flip(img):
    v = random.random()
    if v < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return img

def img_random_crop(img, target_size):
    w, h = img.size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
        "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    if w == tw and h == th:
        return img
    else:
        return img.crop((x1, y1, x1 + tw, y1 + th))

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')