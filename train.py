import src.config as config
import torch
from src.model import ImageClassifier
from src.utils import seed_everything
from src.dataloader import prepare_dataloader
from torch.cuda.amp import GradScaler
import torch.nn as nn
from src import engine
import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv('input/train_folds.csv')
    seed_everything(config.CFG.seed)

    for fold in range(config.CFG.num_folds):
        print(f"training with fold {fold} starts")
        print(
            f"train_size: {len( df[ df['kfold'] != fold] )} validation_size: {len( df[ df['kfold'] == fold] )} "
        )

        train_loader, valid_loader = prepare_dataloader(
            df, fold, data_root=config.CFG.data_root
        )
        device = torch.device(config.CFG.device)
        model = ImageClassifier(
            config.CFG.model_arch, df.label.nunique() - 1, pretrained=True
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.CFG.lr, weight_decay=config.CFG.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0 = config.CFG.T_0, T_mult=1, eta_min=config.CFG.min_lr, last_epoch=-1
        )

        loss_tr = nn.CrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        scaler = GradScaler()
        for epoch in range(config.CFG.epochs):
            engine.train_one_epoch(
                epoch=epoch,
                model=model,
                loss_fn=loss_tr,
                optimizer=optimizer,
                train_loader=train_loader,
                device=device,
                scheduler=scheduler,
                scaler=scaler,
            )
            with torch.no_grad():
                engine.valid_one_epoch(
                    epoch=epoch,
                    model=model,
                    loss_fn=loss_fn,
                    valid_loader=valid_loader,
                    device=device,
                    scheduler=scheduler,
                )
        torch.save(model, f"{config.CFG.model_arch}_fold_{fold}_{epoch}.pth")
        del model, optimizer, train_loader, valid_loader, scheduler
        torch.cuda.empty_cache()
