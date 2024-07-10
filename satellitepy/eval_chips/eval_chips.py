import numpy as np

import torch

from tqdm import tqdm

from satellitepy.data.utils import get_satellitepy_table
from satellitepy.eval_chips.data_chips import get_train_val_dataloaders
from satellitepy.eval_chips.models_chips import get_model, get_head
from satellitepy.models.utils import EarlyStopping


def train(model, head, device, train_loader, val_dataloader, optimizer, scheduler, loss_fn, n_epochs, save_path, patience=10):

    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=save_path
        )

    for epoch in range(n_epochs):
        train_losses = []
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{n_epochs}')
        model.train()

        for (x, y) in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = head(model(x))

            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        acc_sums = 0
        val_pbar = tqdm(val_dataloader, desc=f'Validation {epoch + 1}')
        model.eval()
        with torch.no_grad():
            for (x, y) in val_pbar:
                x, y = x.to(device), y.to(device)
                y_hat = head(model(x))
                val_loss = loss_fn(y_hat, y)
                val_losses.append(val_loss.item())

                pred_int = torch.argmax(y_hat, dim=1)
                gt = torch.argmax(y, dim=1)

                acc_sum = torch.sum(pred_int == gt)  # /len(pred)
                acc_sums += acc_sum

            valid_acc = acc_sums / len(val_dataloader.dataset)
            train_loss = np.average(train_losses)
            val_loss = np.average(val_losses)

            msg = (f'[{epoch+1}/{n_epochs}] ' +
                   f'train_loss: {train_loss:.5f} ' +
                   f'valid_loss: {val_loss:.5f} ' +
                   f'valid_acc: {valid_acc:.2f}')
            print(msg)

            early_stopping(val_loss, model, optimizer, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        scheduler.step()

    print('Finished Training')


if __name__ == '__main__':
    in_image_path = 'S:\\UniBw\\chips\\images'
    in_labels_path = 'S:\\UniBw\\chips\\labels'
    classes = list(get_satellitepy_table()['fineair-class'].keys())

    train_dataloader, val_dataloader = get_train_val_dataloaders(in_image_path, in_labels_path, classes, 16)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model('efficientnet_b3')
    head = get_head(model, len(classes))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)       # try SGD
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    loss = torch.nn.CrossEntropyLoss()

    model.to(device)
    head.to(device)

    train(model, head, device, train_dataloader, val_dataloader, optimizer, scheduler, loss, 50, 'S:\\UniBw\\temp', patience=10)