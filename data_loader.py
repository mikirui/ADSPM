from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os
import random

class Dataset(data.Dataset):
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_pairs = []
        self.test_pairs = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_pairs)
        else:
            self.num_images = len(self.test_pairs)

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if i < 2000:
                self.test_pairs.append([filename, label])
            else:
                self.train_pairs.append([filename, label])

    def __getitem__(self, index):
        data_pairs = self.train_pairs if self.mode == 'train' else self.test_pairs
        filename, label = data_pairs[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, mode='train', num_workers=1, hq=False):
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    if hq == False:
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    
    dataset = Dataset(image_dir, attr_path, selected_attrs, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

