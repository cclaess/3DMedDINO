import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import monai.data as data
import monai.transforms as transforms

import utils
import resnets
import vision_transformers as vits
from heads import DINOHead

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base',
        'vit_large', 'vit_huge', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200'])
    parser.add_argument('--patch_size', default=(12, 12, 6), type=int, nargs=3)
    parser.add_argument('--out_dim', default=65536, type=int)
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)
    parser.add_argument('--momentum_teacher', default=0.996, type=float)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float)
    parser.add_argument('--teacher_temp', default=0.04, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int)

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--weight_decay_end', type=float, default=0.4)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--batch_size_per_gpu', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--freeze_last_layer', default=1, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'])
    parser.add_argument('--drop_path_rate', type=float, default=0.1)

    # Multi-crop parameters
    parser.add_argument('--input_size', default=(96, 96, 48), nargs='+', type=int)
    parser.add_argument('--local_crops_number', type=int, default=8)
    parser.add_argument('--local_crops_scale', type=float, default=0.25)

    # Augmentation parameters
    parser.add_argument('--clip_range', default=(-1000, 1000), nargs=2, type=int)
    parser.add_argument('--use_rotation', type=utils.bool_flag, default=False)
    parser.add_argument('--use_scale', type=utils.bool_flag, default=False)
    parser.add_argument('--use_flip', type=utils.bool_flag, default=False)
    parser.add_argument('--use_elastic', type=utils.bool_flag, default=False)
    parser.add_argument('--use_gaussian', type=utils.bool_flag, default=False)
    parser.add_argument('--use_histogram', type=utils.bool_flag, default=False)

    # Misc
    parser.add_argument('--data_path', default='data/', type=str)
    parser.add_argument('--output_dir', default=".", type=str)
    parser.add_argument('--saveckp_freq', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============

    transform = transforms.Compose([
        utils.LoadRandSpatialCropHDF5(
            # ensures IoU > 0.25 for global views of random croppings
            roi_size=tuple(round(2 * x - ((0.4 ** (1/3)) * x)) for x in args.input_size),
            random_size=False, random_center=True, lps_to_ras=True,
            base_path=args.data_path,
        ),
        transforms.EnsureChannelFirst(channel_dim="no_channel"),
        transforms.ScaleIntensityRange(a_min=args.clip_range[0], a_max=args.clip_range[1], b_min=0, b_max=1, clip=True),
    ])
    transform = transform.set_random_state(args.seed)
    transform = transforms.Compose([
        transform,
        DataAugmentationDINO(args.local_crops_scale, args.local_crops_number, args.seed),
    ])

    train_images = [path for path in glob(os.path.join(args.data_path, '**', '*.hdf5'), recursive=True)]
    train_set = data.Dataset(
        data=train_images,
        transform=transform,
    )
    train_sampler = data.DistributedSampler(train_set, shuffle=True)
    train_loader = data.DataLoader(
        train_set,
        collate_fn=torch.utils.data._utils.collate.default_collate,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
    )

    print(f"Data loaded: there are {len(train_set)} images.")

    # ============ building student and teacher networks ... ============

    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base, vit_large, vit_huge)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            img_size=args.input_size,
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate)  # stochastic depth
        teacher = vits.__dict__[args.arch](
            img_size=args.input_size,
            patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a ResNet (i.e. resnet10, resnet18, resnet34, resnet50)
    elif args.arch in resnets.__dict__.keys():
        student = resnets.__dict__[args.arch]()
        teacher = resnets.__dict__[args.arch]()
        embed_dim = 512 if int(args.arch[6:]) < 50 else 2048
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student,
        DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head))

    # move networks to gpu
    student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============

    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(train_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============

    to_restore = {"epoch": 0, "best_val_loss": math.inf}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]
    best_val_los = to_restore["best_val_loss"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, train_loader, optimizer,
                                      lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args)

        log_stats = {f'train_{k}': v for k, v in train_stats.items()}

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            'best_val_loss': best_val_los,
        }

        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        log_stats.update({'epoch': epoch})

        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    student.train(), teacher.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, update_center=True):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        if update_center:
            self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, local_crops_scale, local_crops_number, seed=0):

        # transformation for the global crops
        self.global_transform = transforms.Compose([
            transforms.RandAffine(
                rotate_range=(30, 30, 30),
                padding_mode="zeros",
                prob=0.5
            ) if args.use_rotation else None,
            transforms.RandAffine(
                scale_range=(0.25, 0.25, 0.25),
                padding_mode="zeros",
                prob=0.5
            ) if args.use_scale else None,
            transforms.RandAxisFlip(
                prob=0.5)
            if args.use_flip else None,
            transforms.Rand3DElastic(
                sigma_range=(0, 7),
                magnitude_range=(0, 150),
                padding_mode="zeros",
                prob=0.5
            ) if args.use_elastic else None,
            transforms.RandGaussianSmooth(
                sigma_x=(0., 2.0),
                sigma_y=(0., 2.0),
                sigma_z=(0., 2.0),
                prob=0.5,
                approx='erf'
            ) if args.use_gaussian else None,
            transforms.RandHistogramShift(
                num_control_points=10,
                prob=0.5
            ) if args.use_histogram else None,
            transforms.RandSpatialCrop(
                roi_size=args.input_size,
                random_size=False,
                random_center=True
            ),
            transforms.SpatialPad(
                spatial_size=args.input_size
            ),
        ])
        self.global_transform = self.global_transform.set_random_state(seed)

        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandAffine(
                rotate_range=(30, 30, 30),
                padding_mode="zeros",
                prob=0.5
            ) if args.use_rotation else None,
            transforms.RandAffine(
                scale_range=(0.25, 0.25, 0.25),
                padding_mode="zeros",
                prob=0.5
            ) if args.use_scale else None,
            transforms.RandAxisFlip(
                prob=0.5)
            if args.use_flip else None,
            transforms.Rand3DElastic(
                sigma_range=(0, 7),
                magnitude_range=(0, 150),
                padding_mode="zeros",
                prob=0.5
            ) if args.use_elastic else None,
            transforms.RandGaussianSmooth(
                sigma_x=(0., 2.0),
                sigma_y=(0., 2.0),
                sigma_z=(0., 2.0),
                prob=0.5,
                approx='erf'
            ) if args.use_gaussian else None,
            transforms.RandHistogramShift(
                num_control_points=10,
                prob=0.5
            ) if args.use_histogram else None,
            transforms.RandSpatialCrop(
                roi_size=tuple(round(x * local_crops_scale) for x in args.input_size),
                random_size=False,
                random_center=True),
            transforms.SpatialPad(
                spatial_size=tuple(round(x * local_crops_scale) for x in args.input_size)),
        ])
        self.local_transform = self.local_transform.set_random_state(seed)

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform(image))
        crops.append(self.global_transform(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_dino(args)
