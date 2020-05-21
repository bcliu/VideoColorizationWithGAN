import argparse
import json
import os
from datetime import datetime

import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.video_dataset import VideoDataset
from model.loss import FeatureAndStyleLoss
from model.resnet_unet import ResNetBasedUNet
from test import load_grayscale, load_grayscale_from_colored, predict


def train(model, optimizer, criterion, train_dataloader, val_dataloader,
          args, device, checkpoint_dirname, summary_writer):
    for epoch in range(args.epochs):
        tqdm.write(f'Epoch {epoch}')

        model.train()
        # Total loss so far in the epoch
        running_loss = 0
        batch_block_loss = 0

        dataloader_tqdm = tqdm(train_dataloader)
        for batch_idx, batch in enumerate(dataloader_tqdm):
            normalized_grayscale = batch[0].to(device)
            normalized_original = batch[1].to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(normalized_grayscale)
                loss = criterion(normalized_original, output)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            batch_block_loss += loss.item()
            average_loss = running_loss / (batch_idx + 1)

            dataloader_tqdm.set_postfix(loss='{:.3f}'.format(loss.item()),
                                        avg_train_loss='{:.3f}'.format(average_loss))

            iter_idx = epoch * len(train_dataloader) + batch_idx

            if (batch_idx + 1) % 100 == 0:
                summary_writer.add_scalar('Last 100 batches average train loss',
                                          batch_block_loss / 100,
                                          iter_idx)
                batch_block_loss = 0

            if (batch_idx + 1) % 500 == 0:
                log_weights(model, summary_writer, iter_idx)

            if (batch_idx + 1) % 1000 == 0:
                log_predictions(model, device, summary_writer, iter_idx)

            if args.checkpoint_iter_interval is not None and (batch_idx + 1) % args.checkpoint_iter_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'iter': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': average_loss,
                }, os.path.join(checkpoint_dirname, f'epoch{epoch}_iter{batch_idx}.pt'))

        tqdm.write('Evaluating on val...')
        val_loss = eval(model, criterion, val_dataloader, device)

        if (epoch + 1) % args.checkpoint_epoch_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': running_loss / len(dataloader_tqdm),
            }, os.path.join(checkpoint_dirname, f'epoch{epoch}_end.pt'))

        summary_writer.add_scalar('Epoch train loss', running_loss / len(train_dataloader), epoch)
        summary_writer.add_scalar('Epoch val loss', val_loss, epoch)
        summary_writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)


def eval(model, criterion, dataloader, device):
    model.eval()
    # Total loss so far in the epoch
    running_loss = 0.0

    dataloader_tqdm = tqdm(dataloader)
    for batch_idx, batch in enumerate(dataloader_tqdm):
        batch = batch.to(device)
        L_channel = batch[:, 0:1, :, :]
        ab_channels = batch[:, [1, 2], :, :]

        with torch.no_grad():
            output = model(L_channel)
            loss = criterion(ab_channels, output)

        running_loss += loss.item()
        average_loss = running_loss / (batch_idx + 1)
        dataloader_tqdm.set_postfix(loss='{:.3f}'.format(loss.item()),
                                    avg_val_loss='{:.3f}'.format(average_loss))

    return running_loss / len(dataloader_tqdm)


def log_predictions(model, device, summary_writer: SummaryWriter, iter_idx: int):
    color_images = {
        'train1': '../../datasets/train/qing-ep38-03894.png',
        'train2': '../../datasets/train/qing-ep49-04922.png',
        'test1': '../../datasets/test/qing-ep35-05470.png',
        'test2': '../../datasets/test/qing-ep15-05389.png',
    }
    bw_images = {
        'crowd': '../bw-frames/00003.png',
        'mayor': '../bw-frames/00008.png',
        'cityhall': '../bw-frames/00028.png',
        'cityhall_far': '../bw-frames/00056.png',
    }

    model.eval()

    for k, v in color_images.items():
        img = load_grayscale_from_colored(v).unsqueeze(0)
        rgb_output = predict(model, img, device)
        summary_writer.add_image(k, rgb_output, iter_idx)

    for k, v in bw_images.items():
        img = load_grayscale(v).unsqueeze(0)
        rgb_output = predict(model, img, device)
        summary_writer.add_image(k, rgb_output, iter_idx)

    model.train()


def log_weights(model, summary_writer, iter_idx: int):
    encoders = model.module._encoders
    decoders = model.module._decoders
    to_log = {
        'encoders[0][0].weight': encoders[0][0].weight,
        'encoders[0][0].weight.grad': encoders[0][0].weight.grad,
        'encoders[4][0].conv1.weight': encoders[4][0].conv1.weight,
        'encoders[4][0].conv1.weight.grad': encoders[4][0].conv1.weight.grad,
        'decoders[0].convs[0].weight': decoders[0].convs[0].weight,
        'decoders[0].convs[0].weight.grad': decoders[0].convs[0].weight.grad,
        'decoders[3].convs[0].weight': decoders[3].convs[0].weight,
        'decoders[3].convs[0].weight.grad': decoders[3].convs[0].weight.grad
    }

    for k, v in to_log.items():
        if v is not None:
            summary_writer.add_histogram(k, v, iter_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Path to training data', required=True)
    parser.add_argument('--val', help='Path to val data', required=True)
    parser.add_argument('--test', help='Path to test data', required=True)
    parser.add_argument('--epochs', help='Number of epochs', default=2, type=int)
    parser.add_argument('--lr', help='Learning rate', default=0.001, type=float)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--freeze-encoder', default=False, action='store_true')
    parser.add_argument('--num-workers', help='Number of data loading workers', default=1, type=int)
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--cuda-device-ids', default='0')
    parser.add_argument('--checkpoint-epoch-interval', help='Save checkpoint per number of epochs', default=1, type=int)
    parser.add_argument('--checkpoint-iter-interval',
                        help='Save checkpoint per number of iterations', default=None, type=int)
    parser.add_argument('--load-checkpoint', default=None, type=str)
    parser.add_argument('--experiment-prefix', default=None, type=str)
    args = parser.parse_args()

    if args.cuda:
        assert torch.cuda.is_available()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Saving checkpoints every {args.checkpoint_epoch_interval} epochs and {args.checkpoint_iter_interval} iterations')
    print(f'Running on {device.type}')

    experiment_name = 'lr{}_{}{}'.format(
        args.lr,
        str(datetime.now())[:-7].replace(" ", "-").replace(":", "-"),
        '' if args.experiment_prefix is None else f'_{args.experiment_prefix}'
    )

    summary_writer = SummaryWriter(os.path.join('tensorboard', experiment_name))
    checkpoint_dirname = os.path.join('checkpoint', experiment_name)
    os.makedirs(checkpoint_dirname, exist_ok=True)

    # Save parameters to file
    with open(os.path.join(checkpoint_dirname, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    train_dataset = VideoDataset(args.train)
    val_dataset = VideoDataset(args.val)
    test_dataset = VideoDataset(args.test)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = ResNetBasedUNet().to(device)

    if args.freeze_encoder:
        model.set_encoders_requires_grad(False)

    if args.cuda:
        model = torch.nn.DataParallel(model)

    if args.load_checkpoint is not None:
        saved_model = torch.load(args.load_checkpoint)
        model.load_state_dict(saved_model['model_state_dict'])

    print(f'Number of param tensors to be optimized: {len(list(filter(lambda p: p.requires_grad, model.parameters())))}')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = FeatureAndStyleLoss(device)

    summary_writer.add_graph(model.module, next(iter(train_dataloader))[0].to(device))

    train(model, optimizer, criterion, train_dataloader, val_dataloader, args, device,
          checkpoint_dirname, summary_writer)

    tqdm.write('Evaluating on test...')
    eval(model, criterion, test_dataloader, device)


if __name__ == '__main__':
    main()
