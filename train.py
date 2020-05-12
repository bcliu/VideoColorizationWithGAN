import torch.nn as nn
import torch.optim
from model.unet import UNet
from dataset.video_dataset import VideoDataset
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from datetime import datetime
import os

def train(model, optimizer, criterion, train_dataloader, val_dataloader, args, checkpoint_dirname):
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')

        model.train()
        # Total loss so far in the epoch
        running_loss = 0.0

        dataloader_tqdm = tqdm(train_dataloader, smoothing=0, ncols=80)
        for batch_idx, batch in enumerate(dataloader_tqdm):
            L_channel = batch[:, 0:1, :, :]
            ab_channels = batch[:, [1, 2], :, :]

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(L_channel)
                loss = criterion(ab_channels, output)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            num_iter_so_far = (batch_idx + 1) * args.log_interval
            average_train_loss = running_loss / num_iter_so_far

            dataloader_tqdm.set_postfix(loss='{:.2f}'.format(loss.item()),
                                        average_loss='{:.2f}'.format(average_train_loss))

        torch.save({
            'epoch': epoch,
            'model_weights': model.state_dict()
        }, os.path.join(checkpoint_dirname, f'epoch{epoch}.pkl'))


def eval(model, criterion, dataloader):
    model.eval()

    print('Evaluating...')
    dataloader_tqdm = tqdm(dataloader, smoothing=0, ncols=80)
    for batch in dataloader_tqdm:
        L_channel = batch[:, 0:1, :, :]
        ab_channels = batch[:, [1, 2], :, :]

        with torch.set_grad_enabled(False):
            output = model(L_channel)
            loss = criterion(ab_channels, output)

        dataloader_tqdm.set_prefix(loss='{:.2f}'.format(loss.item()), refresh=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Path to training data', required=True)
    parser.add_argument('--val', help='Path to val data', required=True)
    parser.add_argument('--test', help='Path to test data', required=True)
    parser.add_argument('--epochs', help='Number of epochs', default=2, type=int)
    parser.add_argument('--lr', help='Learning rate', default=0.001, type=float)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--num-workers', help='Number of data loading workers', default=1, type=int)
    parser.add_argument('--log-interval', default=1, type=int)
    args = parser.parse_args()

    checkpoint_dirname = "checkpoint/lr{}_{}".format(
        args.lr,
        str(datetime.now())[:-7].replace(" ", "-").replace(":", "-"),
    )

    os.makedirs(checkpoint_dirname, exist_ok=True)

    train_dataset = VideoDataset(args.train)
    val_dataset = VideoDataset(args.val)
    test_dataset = VideoDataset(args.test)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model: UNet = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    train(model, optimizer, criterion, train_dataloader, val_dataloader, args, checkpoint_dirname)


if __name__ == '__main__':
    main()
