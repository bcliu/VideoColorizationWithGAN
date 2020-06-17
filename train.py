import argparse
import json
import os
from datetime import datetime

import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.user_guided_dataset import UserGuidedVideoDataset
from model.user_guided_unet import UserGuidedUNet
from test_utils import predict_user_guided


def train(model, optimizer, criterion, train_dataloader, val_dataloader,
          args, device, checkpoint_dirname, summary_writer, images_for_visualization):
    for epoch in range(args.epochs):
        tqdm.write(f'Epoch {epoch}')

        model.train()
        # Total loss so far in the epoch
        running_loss = 0
        batch_block_loss = 0

        dataloader_tqdm = tqdm(train_dataloader)
        for batch_idx, batch in enumerate(dataloader_tqdm):
            L_channel, ab_channels, ab_hint, ab_mask, _ = batch
            L_channel = L_channel.to(device)
            ab_channels = ab_channels.to(device)
            ab_hint = ab_hint.to(device)
            ab_mask = ab_mask.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(L_channel, ab_hint, ab_mask)
                loss = criterion(ab_channels, output)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            batch_block_loss += loss.item()
            average_loss = running_loss / (batch_idx + 1)

            dataloader_tqdm.set_postfix(loss='{:.3f}'.format(loss.item()),
                                        avg_train_loss='{:.3f}'.format(average_loss))

            iter_idx = epoch * len(train_dataloader) + batch_idx

            if (batch_idx + 1) % 50 == 0:
                summary_writer.add_scalar('Last 50 batches average train loss',
                                          batch_block_loss / 50,
                                          iter_idx)
                batch_block_loss = 0

            if (batch_idx + 1) % 200 == 0:
                log_weights(model, summary_writer, iter_idx)

            if (batch_idx + 1) % 1000 == 0:
                log_predictions(model, device, images_for_visualization, summary_writer, iter_idx)

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
        L_channel, ab_channels, ab_hint, ab_mask, _ = batch
        L_channel = L_channel.to(device)
        ab_channels = ab_channels.to(device)
        ab_hint = ab_hint.to(device)
        ab_mask = ab_mask.to(device)

        with torch.no_grad():
            output = model(L_channel, ab_hint, ab_mask)
            loss = criterion(ab_channels, output)

        running_loss += loss.item()
        average_loss = running_loss / (batch_idx + 1)
        dataloader_tqdm.set_postfix(loss='{:.3f}'.format(loss.item()),
                                    avg_val_loss='{:.3f}'.format(average_loss))

    return running_loss / len(dataloader_tqdm)


def log_predictions(model, device, images_for_visualization, summary_writer: SummaryWriter, iter_idx: int):
    model.eval()

    for name, images in images_for_visualization.items():
        for i in range(len(images)):
            L_channel, ab_channels, ab_hint, ab_mask, _ = images[i]
            rgb = predict_user_guided(model, device, L_channel.unsqueeze(0),
                                      ab_hint.unsqueeze(0), ab_mask.unsqueeze(0))
            summary_writer.add_image(f'{name}{i}', rgb, iter_idx, dataformats='HWC')

    model.train()


def log_weights(model, summary_writer, iter_idx: int):
    encoders = model.module._encoders
    decoders = model.module._decoders
    to_log = {
        'encoder1 conv1 weight': encoders['layer1'][0].weight,
        'encoder1 conv1 grad': encoders['layer1'][0].weight.grad,
        'encoder4 conv1 weight': encoders['layer4'][0].conv1.weight,
        'encoder4 conv1 grad': encoders['layer4'][0].conv1.weight.grad,
        'decoder1 conv1 weight': decoders['layer1'].convs[0].weight,
        'decoder1 conv1 grad': decoders['layer1'].convs[0].weight.grad,
        'decoder4 conv1 weight': decoders['layer4'].convs[0].weight,
        'decoder4 conv1 grad': decoders['layer4'].convs[0].weight.grad
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

    train_dataset = UserGuidedVideoDataset(args.train, augmentation=True)
    val_dataset = UserGuidedVideoDataset(args.val, augmentation=False)
    test_dataset = UserGuidedVideoDataset(args.test, augmentation=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    images_for_visualization = {
        'color': UserGuidedVideoDataset('datasets', augmentation=False, files=[
            'train/qing-ep38-03894.png',
            'train/qing-ep49-04922.png',
            'test/qing-ep35-05470.png',
            'test/qing-ep15-05389.png',
        ]),
        'grayscale': UserGuidedVideoDataset('datasets', augmentation=False, files=[
            'bw-frames/test/00003.png',
            'bw-frames/test/00008.png',
            'bw-frames/test/00028.png',
            'bw-frames/test/00056.png',
        ])
    }

    model = UserGuidedUNet().to(device)

    if args.freeze_encoder:
        model.set_encoders_requires_grad(False)

    if args.cuda:
        model = torch.nn.DataParallel(model)

    if args.load_checkpoint is not None:
        saved_model = torch.load(args.load_checkpoint)
        model.load_state_dict(saved_model['model_state_dict'])

    print(f'Number of param tensors to be optimized: {len(list(filter(lambda p: p.requires_grad, model.parameters())))}')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = lambda a, b: torch.nn.functional.smooth_l1_loss(a, b) * 110

    # summary_writer.add_graph(model.module, next(iter(train_dataloader))[0].to(device))

    train(model, optimizer, criterion, train_dataloader, val_dataloader, args, device,
          checkpoint_dirname, summary_writer, images_for_visualization)

    tqdm.write('Evaluating on test...')
    eval(model, criterion, test_dataloader, device)


if __name__ == '__main__':
    main()
