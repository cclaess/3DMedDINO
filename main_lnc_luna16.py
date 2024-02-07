import os
import sys
import json
import random
import argparse
from glob import glob
import itertools as it
from pathlib import Path
from copy import deepcopy
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import monai.data as data
import monai.transforms as transforms
from monai.metrics import ROCAUCMetric, ConfusionMatrixMetric

import utils
import resnets
import vision_transformers as vits


def get_args_parser():
    parser = argparse.ArgumentParser('LUNA 16 LNC', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base',
        'vit_large', 'vit_huge', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200'])
    parser.add_argument('--patch_size', default=(12, 12, 6), type=int, nargs=3)
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
        transforms.EnsureChannelFirst(channel_dim="no_channel"),
        transforms.Orientation(axcodes="RAS"),  # {left-right, posterior-anterior, inferior-superior}
        transforms.ScaleIntensityRange(a_min=args.clip_range[0], a_max=args.clip_range[1], b_min=0, b_max=1, clip=True),
        transforms.RandAffine(translate_range=args.shift_range, padding_mode="zeros", prob=0.9),
        transforms.ToTensor(),
    ])
    transform.set_random_state(args.seed)

    # collect patients and shuffle
    subjects = np.array(sorted(os.path.basename(path) for path in glob(
        os.path.join(args.data_path, "subset*", "*", "*"))))
    subjects = np.unique(subjects)

    indices = np.arange(len(subjects))
    np.random.shuffle(indices)

    # define subset for testing
    test_indices, train_val_indices = np.split(indices, [args.num_test])
    test_subjects = subjects[test_indices]

    test_image_paths = [path for subject in sorted(test_subjects) for path in random.choices(sorted(glob(os.path.join(
        args.data_path, "subset*", "0", subject, "*.nii.gz"))), k=3)] + \
                       [path for subject in sorted(test_subjects) for path in sorted(glob(os.path.join(
        args.data_path, 'subset*', '1', subject, '*.nii.gz')))]

    test_labels = [float(path.split(os.sep)[-3]) for path in test_image_paths]

    test_set = data.ImageDataset(
        image_files=test_image_paths,
        transform=transform,
        labels=test_labels,
        image_only=True,
    )

    test_sampler = data.DistributedSampler(test_set)
    test_loader = data.DataLoader(
        test_set,
        collate_fn=torch.utils.data._utils.collate.default_collate,
        sampler=test_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
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
        train_subjects = subjects[train_indices]

        train_image_paths = [path for subject in sorted(train_subjects) for path in random.choices(
            sorted(glob(os.path.join(args.data_path, "subset*", "0", subject, "*.nii.gz"))), k=3)] + \
                            [path for subject in sorted(train_subjects) for path in sorted(glob(os.path.join(
            args.data_path, 'subset*', '1', subject, '*.nii.gz')))]

        train_labels = [float(path.split(os.sep)[-3]) for path in train_image_paths]

        train_set = data.ImageDataset(
            image_files=train_image_paths,
            transform=transform,
            labels=train_labels,
            image_only=True,
        )

        train_sampler = data.DistributedSampler(train_set)
        train_loader = data.DataLoader(
            train_set,
            collate_fn=torch.utils.data._utils.collate.default_collate,
            sampler=train_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # create validation set and loader
        val_subjects = subjects[val_indices]

        val_image_paths = [path for subject in sorted(val_subjects) for path in random.choices(
            sorted(glob(os.path.join(args.data_path, "subset*", "0", subject, "*.nii.gz"))), k=3)] + \
                          [path for subject in sorted(val_subjects) for path in sorted(glob(os.path.join(
            args.data_path, 'subset*', '1', subject, '*.nii.gz')))]

        val_labels = [float(path.split(os.sep)[-3]) for path in val_image_paths]

        val_set = data.ImageDataset(
            image_files=val_image_paths,
            transform=transform,
            labels=val_labels,
            image_only=True,
        )

        val_sampler = data.DistributedSampler(val_set)
        val_loader = data.DataLoader(
            val_set,
            collate_fn=torch.utils.data._utils.collate.default_collate,
            sampler=val_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"Data loaded with {len(train_set)} train, {len(val_set)} validation, and {len(test_set)} test images.")

        # ============ building network ... ============
        # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
        if args.arch in vits.__dict__.keys():
            model = vits.__dict__[args.arch](
                img_size=args.input_size,
                patch_size=args.patch_size)
            embed_dim = model.embed_dim
        # if the network is a ResNet (i.e. resnet10, resnet18, resnet34, resnet50)
        elif args.arch in resnets.__dict__.keys():
            model = resnets.__dict__[args.arch]()
            embed_dim = 512 if int(args.arch[6:]) < 50 else 2048
        else:
            print(f"Unknown architecture: {args.arch}")
            sys.exit(1)
        classifier = LinearClassifier(embed_dim, num_labels=1)

        # model to GPU
        model.cuda(), classifier.cuda()

        # load weights to evaluate
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key)
        print(f"Model {args.arch} and linear classifier built.")

        classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu])
        if not args.freeze_encoder:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        # set optimizer
        params = classifier.parameters() if args.freeze_encoder else it.chain(classifier.parameters(),
                                                                              model.parameters())
        optimizer = torch.optim.SGD(
            params,
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            momentum=0.9,
            weight_decay=0,  # we do not apply weight decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

        # Optionally resume from a checkpoint
        to_restore = {"epoch": 0, "best_auc": 0.}
        utils.restart_from_checkpoint(
            os.path.join(output_dir, "checkpoint.pth.tar"),
            run_variables=to_restore,
            encoder=model,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = to_restore["epoch"]
        best_auc = to_restore["best_auc"]

        for epoch in range(start_epoch, args.epochs):
            train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, classifier, optimizer, train_loader, epoch)
            scheduler.step()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}
            if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
                val_stats = validate_network(model, classifier, val_loader)
                print(
                    f"AUC at epoch {epoch} of the network on the {len(val_set)} validation images: {val_stats['auc']:.2f}")
                if val_stats['auc'] > best_auc:
                    best_auc = val_stats['auc']
                    if utils.is_main_process():
                        save_dict = {
                            "epoch": epoch + 1,
                            "encoder": model.state_dict(),
                            "classifier": classifier.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_auc": best_auc,
                        }
                        torch.save(save_dict, os.path.join(output_dir, "checkpoint_auc.pth.tar"))
                print(f'Max AUC so far: {best_auc:.2f}')
                log_stats = {**{k: v for k, v in log_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()}}
            if utils.is_main_process():
                with (Path(output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                save_dict = {
                    "epoch": epoch + 1,
                    "encoder": model.state_dict(),
                    "classifier": classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_auc": best_auc,
                }
                torch.save(save_dict, os.path.join(output_dir, "checkpoint.pth.tar"))

        del optimizer, scheduler, train_loader, val_loader, train_sampler, val_sampler

        # load best model and evaluate on test set
        torch.distributed.barrier()
        best_dict = torch.load(os.path.join(output_dir, "checkpoint_auc.pth.tar"), map_location="cpu")
        model.load_state_dict(best_dict["encoder"])
        classifier.load_state_dict(best_dict["classifier"])
        test_stats = validate_network(model, classifier, test_loader)

        del model, classifier

        print(f"Training run {current_fold_idx + 1}/{args.k_fold} completed.\nTest AUC: {test_stats['auc']:.2f}")


def train(model, classifier, optimizer, loader, epoch):
    model.train() if not args.freeze_encoder else model.eval()
    classifier.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for (inp, target) in metric_logger.log_every(loader, 10, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(-1)

        # forward
        with torch.no_grad() if args.freeze_encoder else nullcontext():
            output = model(inp)
        output = classifier(output)

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
def validate_network(model, classifier, val_loader):
    model.eval(), classifier.eval()

    auc_metric = ROCAUCMetric()
    accuracy_metric = ConfusionMatrixMetric(metric_name="accuracy", include_background=True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for (inp, target) in metric_logger.log_every(val_loader, 10, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(-1)

        # forward
        with torch.no_grad():
            output = model(inp)
            output = classifier(output)

        loss = nn.BCEWithLogitsLoss()(output, target)

        auc_metric(y_pred=F.sigmoid(output).cpu(), y=target.cpu())
        accuracy_metric(y_pred=torch.where(output.cpu() < 0., 0, 1), y=target.cpu())

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    loss = metric_logger.meters['loss'].global_avg
    auc = auc_metric.aggregate()
    accuracy = accuracy_metric.aggregate()[0].item()
    print(f'* Auc {auc:.3f} accuracy {accuracy:.3f} loss {loss:.3f}')

    return {"loss": loss, "auc": auc, "accuracy": accuracy}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of features"""

    def __init__(self, dim, num_labels=1):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LUNA 16 LNC', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
