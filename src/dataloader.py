import dataset
import augmentations
from torch.utils.data import DataLoader
import config

def prepare_dataloader(df, trn_idx, val_idx, data_root='../input/siim-isic-melanoma-classification/jpeg/train/'):
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)

    train_ds = dataset.ImageDataset(train_, data_root, transforms = augmentations.get_train_aug(), output_labels=True)
    valid_ds = dataset.ImageDataset(valid_, data_root, transforms = augmentations.get_valid_aug(), output_labels=True)

    train_loader = DataLoader(
        train_ds,
        batch_size = config.CFG.train_bs,
        pin_memory = True,
        drop_last = False,
        shuffle = True,
        num_workers = config.CFG.num_workers,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size = config.CFG.valid_bs,
        pin_memory = False,
        shuffle = False,
        num_workers = config.CFG.num_workers,
    )

    return train_loader, valid_loader