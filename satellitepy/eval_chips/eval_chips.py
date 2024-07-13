import numpy as np

import torch
import torchvision

from tqdm import tqdm

from satellitepy.data.utils import get_satellitepy_table
from satellitepy.eval_chips.data_chips import get_train_val_dataloaders
from satellitepy.eval_chips.models_chips import get_model, get_head
from satellitepy.models.utils import EarlyStopping


def train(model, head, device, train_loader, val_dataloader, optimizer, scheduler, loss_fn, n_epochs, save_path, patience=10, val_by_source=False):

    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=save_path
        )

    for epoch in range(n_epochs):
        train_losses = []
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{n_epochs}')
        model.train()

        for (x, y, s) in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = head(model(x))

            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        acc_sums = np.zeros([len(classes), 3])
        nums = np.zeros([len(classes), 3])
        val_pbar = tqdm(val_dataloader, desc=f'Validation {epoch + 1}')
        model.eval()
        with torch.no_grad():
            for (x, y, s) in val_pbar:
                x, y = x.to(device), y.to(device)
                y_hat = head(model(x))
                val_loss = loss_fn(y_hat, y)
                val_losses.append(val_loss.item())

                pred_int = torch.argmax(y_hat, dim=1)
                gt = torch.argmax(y, dim=1)

                acc_sum = torch.sum(pred_int == gt)  # /len(pred)
                acc_sums[gt, s] += acc_sum
                nums[gt, s] += 1

            val_accs = acc_sums / nums
            val_acc = np.sum(acc_sums, axis=(0, 1)) / np.sum(nums, axis=(0, 1))

            train_loss = np.average(train_losses)
            val_loss = np.average(val_losses)

            msg = (f'[{epoch+1}/{n_epochs}] ' +
                   f'train_loss: {train_loss:.5f} ' +
                   f'valid_loss: {val_loss:.5f} ' +
                   f'valid_acc: {val_acc:.5f} ')

            val_accs = np.where(nums != 0, val_accs, -1)

            if val_by_source:
                print(classes)
                print(nums)
                print(val_accs)
            print(msg)


            early_stopping(val_loss, model, optimizer, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # scheduler.step()

    print('Finished Training')


if __name__ == '__main__':
    in_image_path = 'S:\\UniBw\\chips\\images'
    in_labels_path = 'S:\\UniBw\\chips\\labels'
    classes = list(get_satellitepy_table()['fineair-class'].keys())

    transform = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomRotation(180), torchvision.transforms.RandomVerticalFlip()]

    train_dataloader, val_dataloader = get_train_val_dataloaders(in_image_path, in_labels_path, classes, train_batch_size=16, val_batch_size=1, transform=transform)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model('resnet34')
    head = get_head(model, len(classes))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    loss = torch.nn.CrossEntropyLoss()

    model.to(device)
    head.to(device)

    train(model, head, device, train_dataloader, val_dataloader, optimizer, scheduler, loss, 50, 'S:\\UniBw\\temp', patience=10, val_by_source=True)