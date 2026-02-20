#! /usr/bin/env python3

"""inpc_unet_pretraining.py: Pretrains the U-Net for the INPC method based on a set of trained INPC models."""

from pathlib import Path
import random
from datetime import datetime

import torch
import torchvision.transforms as transforms
import concurrent.futures

import wandb
from torchvision.utils import make_grid

import utils
with utils.DiscoverSourcePath():
    from Logging import Logger
    from Methods.INPC.Loss import UNetPretrainingLoss
    from Methods.INPC.Modules.UNet import UNet
    from Thirdparty.Apex import FusedAdam


class INPCSceneDataset:
    def __init__(self, scene_path: Path, base_crop_size: int):
        self.samples = list((scene_path / 'pretraining_data').glob('*.pt'))
        if not self.samples:
            raise FileNotFoundError(f'No samples found in {scene_path / "pretraining_data"}')
        print(f'Found {len(self.samples)} samples in {scene_path / "pretraining_data"}')
        self.crop_sizes = [
            base_crop_size,
            int(base_crop_size * 1.25),
            int(base_crop_size * 1.5),
            int(base_crop_size * 1.75),
            base_crop_size * 2,
        ]
        self.base_crop_size = base_crop_size

    def sample_random(self):
        path = random.choice(self.samples)
        tensor = torch.load(path, map_location='cpu')
        crop_size = random.choice(self.crop_sizes)
        random_crop = transforms.RandomCrop(crop_size)
        cropped = random_crop(tensor)
        if crop_size != self.base_crop_size:
            cropped = torch.nn.functional.interpolate(
                cropped.unsqueeze(0),
                size=(self.base_crop_size, self.base_crop_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        return cropped[:4], cropped[4:]


def get_data_generator(base_path: Path, model_dirs: list[str], crop_size: int):
    scene_datasets = [
        INPCSceneDataset(base_path / scene, crop_size) for scene in model_dirs
    ]

    def sample_from_scene(scene_dataset):
        return scene_dataset.sample_random()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(model_dirs), 16)) as executor:
        while True:
            futures = [executor.submit(sample_from_scene, scene) for scene in scene_datasets]
            results = [future.result() for future in futures]
            inputs, targets = zip(*results)
            yield torch.stack(inputs), torch.stack(targets)


def main():
    BASE_PATH = Path('/path/to/pretrained/models')  # TODO: the directory containing the pretraining model directories
    PRETRAINING_MODEL_DIRS = [
        'ignatius',
        'barn',
        'church',
        'truck',
        'caterpillar',
        'meetingroom',
        'courthouse',
        'hydrant',
        'car',
        'teddybear',
        'plant',
    ]  # TODO: rename with the actual directory names for each scene

    # hyperparameters
    N_ITERATIONS = 10_000
    LR_INIT = 6e-4
    LR_FINAL = 1e-4
    CROP_SIZE = 512
    USE_FFC_BLOCK = True

    wandb.init(project='inpc-unet-pretraining', name='stage3_unet', dir='./output/', config={
        'iterations': N_ITERATIONS,
        'lr_init': LR_INIT,
        'lr_final': LR_FINAL,
        'crop_size': CROP_SIZE,
        'use_ffc_block': USE_FFC_BLOCK
    })
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    unet = UNet(USE_FFC_BLOCK).cuda().train()
    loss_fn = UNetPretrainingLoss()
    param_groups, schedulers = unet.get_optimizer_param_groups(N_ITERATIONS, LR_INIT, LR_FINAL)
    optimizer = FusedAdam(param_groups, lr=1.0, betas=(0.9, 0.99), eps=1e-15, adam_w_mode=True, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedulers)
    grad_scaler = torch.GradScaler(device='cuda', init_scale=128.0, growth_interval=N_ITERATIONS + 1)

    data_generator = get_data_generator(BASE_PATH, PRETRAINING_MODEL_DIRS, CROP_SIZE)
    progress_bar = Logger.log_progress(range(N_ITERATIONS), desc='INPC U-Net Pretraining')
    for iteration in progress_bar:
        input, target = next(data_generator)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16):
            output = unet(input).float()  # cast to float as in INPC renderer
            loss = loss_fn(output, target)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()
        loss_value = loss.item()
        progress_bar.set_postfix(loss=loss_value)
        wandb.log({'loss': loss_value}, step=iteration)
        if iteration % 500 == 0:
            with torch.no_grad():
                target_vis = target.cpu().clamp(0, 1)
                output_vis = output.cpu().clamp(0, 1)
                combined = torch.cat([target_vis, output_vis], dim=0)
                grid = make_grid(combined, nrow=target.size(0), padding=2, pad_value=1.0)
                wandb.log({'gt_vs_output_grid': wandb.Image(grid)}, step=iteration)
    torch.cuda.synchronize()
    checkpoint_path = Path(f'./output/inpc_pretrained_unet_weights_stage3_{timestamp}.pt')
    unet.save_weights(checkpoint_path)


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    main()
