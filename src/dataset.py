import utils
import os
import torch.utils.data as data

class ImageDataset(data.Dataset):

    def __init__(self, df, data_root, transforms=None, output_labels=None):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transforms = transforms
        self.output_labels = output_labels

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index:int):
        if self.output_labels:
            target = self.df['label'][index]

        img = utils.get_img(os.path.join(self.data_root,f'{self.df["id"][index]}.png'))

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.output_labels:
            return img, target
        else:
            return img





