import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import config
import numpy as np
import torch
scaler = GradScaler()

def train_one_epoch(
    model, loss_fn, device, optimizer, scheduler, train_loader, epoch
):
    model.train()
    running_loss = None
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

    for step, (images, image_targets) in progress_bar:
        images = images.to(device).float()
        image_targets = image_targets.to(device).long()


        with autocast:
            image_predictions = model(images)
            loss = loss_fn(image_predictions, image_targets)
            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * 0.99 + loss.item * 0.01

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # scheduler.step()

        if ((step + 1) % config.CFG.verbosity == 1) or ((step + 1) == len(train_loader)):
            description = f"epoch: {epoch}, loss: {running_loss:.4f}"
            progress_bar.set_description(description)
    scheduler.step()


def valid_one_epoch(model, loss_fn, device, scheduler, epoch, valid_loader):
    model.eval()
    image_predictions_all = []
    image_targets_all = []
    sample_num = 0
    total_loss = 0

    progress_bar = tqdm(valid_loader, total=len(valid_loader))
    for step, (images, images_targets) in progress_bar:
        images = images.to(device).float()
        image_targets = image_targets.to(device).long()

        image_predictions = model(images)
        image_predictions_all += [
            torch.argmax(image_predictions, 1).detach().cpu().numpy()
        ]
        image_targets_all += [image_predictions.detach().cpu().numpy()]

        loss = loss_fn(image_predictions, image_targets)
        total_loss += loss.item() * image_targets.shape[0]
        sample_num += image_targets.shape[0]

        if ((step + 1) % config.CFG.verbosity == 0) or (
            (step + 1) % len(valid_loader) == 0
        ):
            description = f"epoch: {epoch}, loss: {total_loss/sample_num}"
            progress_bar.set_description(description)

    image_predictions_all = np.concatenate(image_predictions_all)
    image_targets_all = np.concatenate(image_targets_all)
    print(
        f"validation multi-class accuracy: {(image_predictions_all==image_targets_all).mean():.4f}"
    )
    scheduler.step()
