import os
import sys
import json
import argparse
from glob import glob
import itertools as it
from pathlib import Path
from copy import deepcopy
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import monai.data as data
import monai.transforms as transforms
from monai.metrics import DiceMetric, HausdorffDistanceMetric

import utils
import resnets


def get_args_parser():
    parser = argparse.ArgumentParser('Medical Segmentation Decathlon LTS', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='resnet50', type=str, choices=['resnet10', 'resnet18', 'resnet34', 'resnet50',
        'resnet101', 'resnet152', 'resnet200'])
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument("--checkpoint_key", default="teacher", type=str)
    parser.add_argument('--freeze_encoder', default=False, type=utils.bool_flag)

    # Training/Optimization parameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument('--batch_size_per_gpu', default=128, type=int)

    # Augmentation parameters
    parser.add_argument('--clip_range', default=(-1000, 1000), nargs=2, type=int)
    parser.add_argument('--shift_range', default=(0, 0, 0), nargs=3, type=int)

    # Misc
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--data_path', default='data/', type=str)
    parser.add_argument('--k_fold', default=5, type=int)
    parser.add_argument('--num_test', default=37, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--output_dir', default=".")

    return parser


def main(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============

    # image transforms
    transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"], image_only=True),
        transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=args.clip_range[0], a_max=args.clip_range[1],
                                        b_min=0, b_max=1, clip=True),
        transforms.RandAffine(translate_range=args.shift_range, padding_mode="zeros", prob=0.9),
        transforms.ToTensord(keys=["image", "label"]),
    ])
    transform.set_random_state(args.seed)

    # create dataset of all files
    image_paths = np.array(sorted(glob(os.path.join(args.data_path, 'images', '*.nii.gz'))))
    label_paths = np.array(sorted(glob(os.path.join(args.data_path, 'labels', '*.nii.gz'))))

    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)

    # define subset for testing
    test_indices, train_val_indices = np.split(indices, [args.num_test])
    test_paths = [{"image": image_paths[idx], "label": label_paths[idx]} for idx in test_indices]

    test_set = data.Dataset(
        data=test_paths,
        transform=transform)

    test_sampler = data.DistributedSampler(test_set)
    test_loader = data.DataLoader(
        test_set,
        sampler=test_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # check if number of samples is divisible by k-fold
    assert len(train_val_indices) % args.k_fold == 0, "training and validation samples must be divisible by k-fold"
    num_val = len(train_val_indices) // args.k_fold if args.k_fold != 1 else len(train_val_indices) // 5

    # chunk samples for k-fold cross-validation
    train_val_indices = list(utils.chunk_list(train_val_indices, num_val))

    for current_fold_idx in range(args.k_fold):
        print(f"Starting run {current_fold_idx + 1}/{args.k_fold} of {args.k_fold}-fold cross-validation")

        # create directory for k-fold
        output_dir = os.path.join(args.output_dir, str(current_fold_idx))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # define training and validation samples
        train_indices = deepcopy(train_val_indices)
        val_indices = train_indices.pop(current_fold_idx)
        train_indices = [ind for sublist in train_indices for ind in sublist]

        # create training set and loader
        train_paths = [{"image": image_paths[idx], "label": label_paths[idx]} for idx in train_indices]

        train_set = data.Dataset(
            data=train_paths,
            transform=transform)

        train_sampler = data.DistributedSampler(train_set, shuffle=True)
        train_loader = data.DataLoader(
            train_set,
            sampler=train_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # create validation set and loader
        val_paths = [{"image": image_paths[idx], "label": label_paths[idx]} for idx in val_indices]

        val_set = data.Dataset(
            data=val_paths,
            transform=transform)

        val_sampler = data.DistributedSampler(val_set)
        val_loader = data.DataLoader(
            val_set,
            sampler=val_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        print(f"Data loaded with {len(train_set)} train, {len(val_set)} validation, and {len(test_set)} test images.")

        # ============ building network ... ============

        if args.arch in resnets.__dict__.keys():
            model = resnets.__dict__[args.arch]()
            decoder = resnets.__dict__[args.arch + "_decoder"](
                num_classes=1)
        else:
            print(f"Unknown architecture: {args.arch}")
            sys.exit(1)

        # model to GPU
        model.cuda(), decoder.cuda()

        # load weights to evaluate
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key)
        print(f"Model {args.arch} and decoder built.")

        decoder = nn.parallel.DistributedDataParallel(decoder, device_ids=[args.gpu])
        if not args.freeze_encoder:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        # set optimizer
        params = decoder.parameters() if args.freeze_encoder else it.chain(decoder.parameters(), model.parameters())

        optimizer = torch.optim.SGD(
            params,
            args.lr,  # * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            momentum=0.9,
            weight_decay=0,  # we do not apply weight decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

        # Optionally resume from a checkpoint
        to_restore = {"epoch": 0, "best_dice": 0.}
        utils.restart_from_checkpoint(
            os.path.join(output_dir, "checkpoint.pth.tar"),
            run_variables=to_restore,
            encoder=model,
            decoder=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = to_restore["epoch"]
        best_dice = to_restore["best_dice"]

        for epoch in range(start_epoch, args.epochs):
            train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, decoder, optimizer, train_loader, epoch)
            scheduler.step()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}
            if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
                val_stats = validate_network(model, decoder, val_loader)
                print(
                    f"Dice at epoch {epoch} of the network on the {len(val_set)} validation images: {val_stats['dice']:.2f}")
                if val_stats['dice'] > best_dice:
                    best_dice = val_stats['dice']
                    if utils.is_main_process():
                        save_dict = {
                            "epoch": epoch + 1,
                            "encoder": model.state_dict(),
                            "decoder": decoder.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_dice": best_dice,
                        }
                        torch.save(save_dict, os.path.join(output_dir, "checkpoint_dice.pth.tar"))
                print(f'Max DSC so far: {best_dice:.2f}')
                log_stats = {**{k: v for k, v in log_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()}}
            if utils.is_main_process():
                with (Path(output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                save_dict = {
                    "epoch": epoch + 1,
                    "encoder": model.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_dice": best_dice,
                }
                torch.save(save_dict, os.path.join(output_dir, "checkpoint.pth.tar"))

        del optimizer, scheduler, train_loader, val_loader, train_sampler, val_sampler

        # load best model and evaluate on test set
        torch.distributed.barrier()
        best_dict = torch.load(os.path.join(output_dir, "checkpoint_dice.pth.tar"), map_location="cpu")
        model.load_state_dict(best_dict["encoder"])
        decoder.load_state_dict(best_dict["decoder"])
        test_stats = validate_network(model, decoder, test_loader)

        del model, decoder

        print(f"Training run {current_fold_idx + 1}/{args.k_fold} completed.\nTest Dice: {test_stats['dice']:.2f}")


def train(model, decoder, optimizer, loader, epoch):
    model.train() if not args.freeze_encoder else model.eval()
    decoder.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for (inp, target) in metric_logger.log_every(loader, 10, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = torch.where(target > 0, 1., 0.).cuda(non_blocking=True)

        # forward
        with torch.no_grad() if args.freeze_encoder else nullcontext():
            output = model.module.forward_features_unet(inp) if not args.freeze_encoder else model.forward_features_unet(inp)
        output = decoder(*output)

        # compute cross entropy loss
        loss = nn.BCEWithLogitsLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate_network(model, decoder, val_loader):
    model.eval(), decoder.eval()

    dice_metric = DiceMetric(include_background=True)
    hausdorff_metric = HausdorffDistanceMetric(include_background=True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for (inp, target) in metric_logger.log_every(val_loader, 10, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = torch.where(target > 0, 1., 0.).cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model.module.forward_features_unet(inp) if not args.freeze_encoder else model.forward_features_unet(inp)
            output = decoder(*output)

        loss = nn.BCEWithLogitsLoss()(output, target)

        dice_metric(y_pred=torch.where(output < 0., 0., 1.), y=target)
        hausdorff_metric(y_pred=torch.where(output < 0., 0., 1.), y=target)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    loss = metric_logger.meters['loss'].global_avg
    dice = dice_metric.aggregate().item()
    hausdorff = hausdorff_metric.aggregate().item()
    print(f'* Dice {dice:.3f} Hausdorff {hausdorff:.3f} loss {loss:.3f}')

    return {"loss": loss, "dice": dice, "hausdorff": hausdorff}


def collate_fn(batch):
    return torch.stack([b["image"] for b in batch]), torch.stack([b["label"] for b in batch])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Medical Segmentation Decathlon LTS', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
