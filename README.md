# INPC: Implicit Neural Point Clouds for Radiance Field Rendering
Florian Hahlbohm, Linus Franke, Moritz Kappel, Susana Castillo, Martin Eisemann, Marc Stamminger, Marcus Magnor<br>
| [Project page](https://fhahlbohm.github.io/inpc/) | [Paper](https://arxiv.org/abs/2403.16862) | [Video](https://youtu.be/XnMlwvNb2-Q) | [Evaluation Images (8 GB)](https://graphics.tu-bs.de/upload/publications/hahlbohm2025inpc/inpc_full_eval.zip) |<br>

## Overview
This repository contains the official implementation of "INPC: Implicit Neural Point Clouds for Radiance Field Rendering".
It is provided as an extension to the [NeRFICG](https://github.com/nerficg-project) framework.

## Getting Started

### Our Setup
Most of our tests as well as experiments for the paper were conducted using the following setup:
- Operating System: Ubuntu 22.04
- GPU: Nvidia GeForce RTX 4090
- CUDA Driver Version: 535.183.01
- CUDA Toolkit Version: 11.8
- Python Version: 3.11
- PyTorch Version: 2.5.1

For our larger models, we used an Nvidia A100 GPU with 80 GB of VRAM with the remaining setup being similar.

We have verified that everything works with CUDA Toolkit Version 12.4, but did not measure performance.

### Setup

As a preparatory step, the [NeRFICG framework](https://github.com/nerficg-project/nerficg) needs to be set up.

<details>
<summary><span style="font-weight: bold;">TL;DR NeRFICG Setup</span></summary>

- Clone the NeRFICG repository and its submodules:
	```shell
	git clone git@github.com:nerficg-project/nerficg.git --recursive && cd nerficg
	```
 
- Install the dependencies listed in `scripts/condaEnv.sh`, or automatically create a new conda environment by executing the script:
	```shell
	./scripts/condaEnv.sh && conda activate nerficg
	```
 
- [optional] For logging via [Weights & Biases](https://wandb.ai/site), run the following command and enter your account identifier:
	```shell
	wandb login
	```
 
</details>
<br>

Now, you can directly add this project as an additional method:

- Clone this repository into the `src/Methods/` directory:
	```shell
	git clone git@github.com:nerficg-project/INPC.git src/Methods/INPC
	```
- install all method-specific dependencies and CUDA extensions using:
	```shell
	./scripts/install.py -m INPC
	```

## Training and Inference

The INPC method is fully compatible with the NeRFICG scripts in the `scripts/` directory.
This includes config file generation via `defaultConfig.py`,
training via `train.py`,
inference and performance benchmarking via `inference.py`,
metric calculation via `generateTables.py`,
and live rendering via `gui.py` (Linux only).
We also used these scripts for the experiments in our paper.

For detailed instructions, please refer to the [NeRFICG framework repository](https://github.com/nerficg-project/nerficg).

### Example Configuration Files

We provide an exemplary configuration file for the bicycle scene from the [Mip-NeRF360](https://jonbarron.info/mipnerf360/) dataset and recommend copying it to the `configs/` directory.
For the eight *intermediate* scenes from the [Tanks and Temples](https://www.tanksandtemples.org/) dataset on which we evaluate our method in the paper, we used [our own calibration](https://cloud.tu-braunschweig.de/s/J5xYLLEdMnRwYPc) obtained using COLMAP.
For Tanks and Temples scenes we use the same dataloader as for Mip-NeRF360 scenes.

*Note:* There will be no documentation for the method-specific configuration parameters under `TRAINING.XXXX`/`MODEL.XXXX`/`RENDERER.XXXX`.
Please conduct the code and/or our paper for understanding what they do.

### Using Custom Data

While this method is compatible with most of the dataset loaders provided with the [NeRFICG framework](https://github.com/nerficg-project/nerficg),
we recommend using exclusively the Mip-NeRF360 loader (`src/Datasets/MipNeRF360.py`) for custom data.
It is compatible with the COLMAP format for single-camera captures:
```
custom_scene
└───images
│   │   00000.jpg
│   │   ...
│   
└───sparse/0
│   │   cameras.bin
│   │   images.bin
│   │   points3D.bin
│
└───images_2  (optional)
│   │   00000.jpg
│   │   ...
```

To use it, just modify `DATASET.PATH` near the bottom of one of the exemplary configuration files. Furthermore, you may want to modify the following dataset configuration parameters:
- `DATASET.IMAGE_SCALE_FACTOR`: Set this to `null` for using the original resolution or a between zero and one to train on downscaled images.
 If `DATASET.USE_PRECOMPUTED_DOWNSCALING` is set to `true` specifying `0.5`/`0.25`/`0.125` will load images from directories `images_2`/`images_4`/`images_8` respectively.
 We recommend using this feature and downscaling manually via, e.g., `mogrify -resize 50% *.jpg` for the best results.
- `DATASET.TO_DEVICE`: Set this to `false` for large datasets or if you have less than 24 GB of VRAM.
- `DATASET.BACKGROUND_COLOR`: Will be ignored as INPC uses a separate background model.
- `DATASET.NEAR_PLANE`: We used `0.01` for all scenes in our experiments.
- `DATASET.FAR_PLANE`: We used `100.0` for all scenes in our experiments.
- `DATASET.TEST_STEP`: Set to `8` for the established evaluation protocol. Set to `0` to use all images for training.
- `DATASET.APPLY_PCA`: Tries to align the world space so that the up-axis is parallel to the direction of gravity using principal component analysis.
 Although it does not always work, we recommend setting this to `true` if you want to view the final model inside a GUI.
 For our experiments, we also use `DATASET.APPLY_PCA_RESCALE: true` to scale the scene so that all camera poses are inside the \[-1, 1\] cube.

If using your custom data fails, you have two options:
1. (Easy) Re-calibrate using, e.g., `./scripts/colmap.py -i <path/to/your/scene> --camera_mode single` and add `-u` at the end if your images are distorted.
2. (Advanced) Check the NeRFICG instructions for using custom data [here](https://github.com/nerficg-project/nerficg?tab=readme-ov-file#training-on-custom-image-sequences) and optionally dive into the NeRFICG code to extend one of the dataloaders to handle your data.

### torch.compile Usage

For accelerated  rendering during inference, we use [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) to compile the U-Net module.
This reduces rendering times by about 8-10 ms per frame, which is a considerable speedup especially for the rendering mode that uses a pre-extracted global point cloud.
If this causes problems on your machine, you can try disabling U-Net compilation by setting `RENDERER.USE_COMPILED_UNET_INFERENCE: false` in the configuration file.

We also experimented with using [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for a few other functions.
You can find these places in the code by searching for `TORCH_COMPILE_NOTE`. Enabling them speeds up training by up to 20% but might result in slightly lower quality.

## License and Citation
This project is licensed under the MIT license (see [LICENSE](LICENSE)).

If you use this code for your research projects, please consider a citation:
```bibtex
@inproceedings{hahlbohm2025inpc,
  title     = {{INPC}: Implicit Neural Point Clouds for Radiance Field Rendering},
  author    = {Hahlbohm, Florian and Franke, Linus and Kappel, Moritz and Castillo, Susana and Eisemann, Martin and Stamminger, Marc and Magnor, Marcus},
  booktitle = {International Conference on 3D Vision},
  year      = {2025},
  url       = {https://fhahlbohm.github.io/inpc/}
}
```
