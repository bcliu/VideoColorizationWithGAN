import argparse
import os
from datetime import datetime

import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.video_dataset import VideoDataset
from model.unet import UNet


def train(model, optimizer, criterion, train_dataloader, val_dataloader, args, device, checkpoint_dirname, summary_writer):
    for epoch in range(args.epochs):
        tqdm.write(f'Epoch {epoch}')

        model.train()
        # Total loss so far in the epoch
        running_loss = 0.0

        dataloader_tqdm = tqdm(train_dataloader)
        for batch_idx, batch in enumerate(dataloader_tqdm):
            batch = batch.to(device)
            L_channel = batch[:, 0:1, :, :]
            ab_channels = batch[:, [1, 2], :, :]

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(L_channel)
                loss = criterion(ab_channels, output)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            average_loss = running_loss / (batch_idx + 1)

            dataloader_tqdm.set_postfix(loss='{:.2f}'.format(loss.item()),
                                        avg_train_loss='{:.2f}'.format(average_loss))

        tqdm.write('Evaluating on val...')
        eval(model, criterion, val_dataloader, device)

        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_weights': model.state_dict()
            }, os.path.join(checkpoint_dirname, f'epoch{epoch}.pkl'))


def eval(model, criterion, dataloader, device):
    model.eval()
    # Total loss so far in the epoch
    running_loss = 0.0

    dataloader_tqdm = tqdm(dataloader)
    for batch_idx, batch in enumerate(dataloader_tqdm):
        batch = batch.to(device)
        L_channel = batch[:, 0:1, :, :]
        ab_channels = batch[:, [1, 2], :, :]

        with torch.set_grad_enabled(False):
            output = model(L_channel)
            loss = criterion(ab_channels, output)

        running_loss += loss.item()
        average_loss = running_loss / (batch_idx + 1)
        dataloader_tqdm.set_postfix(loss='{:.2f}'.format(loss.item()),
                                    avg_val_loss='{:.2f}'.format(average_loss))


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
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--cuda-device-ids', default='0')
    parser.add_argument('--checkpoint-interval', help='Saving checkpoint per number of epochs', default=1, type=int)
    args = parser.parse_args()

    if args.cuda:
        assert torch.cuda.is_available()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    experiment_name = 'lr{}_{}'.format(
        args.lr,
        str(datetime.now())[:-7].replace(" ", "-").replace(":", "-")
    )

    summary_writer = SummaryWriter(os.path.join('tensorboard', experiment_name))
    checkpoint_dirname = os.path.join('checkpoint', experiment_name)
    os.makedirs(checkpoint_dirname, exist_ok=True)

    train_dataset = VideoDataset(args.train)
    val_dataset = VideoDataset(args.val)
    test_dataset = VideoDataset(args.test)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = UNet().to(device)
    if args.cuda:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    train(model, optimizer, criterion, train_dataloader, val_dataloader, args, device,
          checkpoint_dirname, summary_writer)

    tqdm.write('Evaluating on test...')
    eval(model, criterion, test_dataloader, device)

if __name__ == '__main__':
    main()
