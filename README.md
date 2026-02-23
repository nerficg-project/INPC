# INPC: Implicit Neural Point Clouds

This repository contains the reference implementation of our work on Implicit Neural Point Cloud (INPC) as an extension
to the [NeRFICG](https://github.com/nerficg-project) framework.  
The key idea of INPC is to represent a point cloud _implicitly_ using a 3D probability distribution and a neural
appearance field. This avoids disadvantages commonly associated with explicit point-based representations for novel
view synthesis, most notably direct optimization of point positions and point cloud densification. As a result, INPC
achieves better quality than explicit point-based methods such as 3D Gaussian Splatting. Compared to Zip-NeRF, it
renders faster and achieves similar or better visual fidelity.  
You can find more details about the method and our experiments in our papers:

> __INPC: Implicit Neural Point Clouds for Radiance Field Rendering__  
> [Florian Hahlbohm](https://fhahlbohm.github.io), [Linus Franke](https://lfranke.github.io/), [Moritz Kappel](https://moritzkappel.github.io/), [Susana Castillo](https://graphics.tu-bs.de/people/castillo), [Martin Eisemann](https://graphics.tu-bs.de/people/eisemann), [Marc Stamminger](https://www.lgdv.tf.fau.de/person/marc-stamminger/), [Marcus Magnor](https://graphics.tu-bs.de/people/magnor)  
 _International Conference on 3D Vision (3DV), March 2025_  
> __[Project page](https://fhahlbohm.github.io/inpc/)&nbsp;| [Paper](https://arxiv.org/abs/2403.16862)&nbsp;| [Video](https://youtu.be/XnMlwvNb2-Q)&nbsp;| [User Study (Reimpl.)](https://fhahlbohm.github.io/inpc_vs_zipnerf/)&nbsp;| [Evaluation Images (8 GB)](https://graphics.tu-bs.de/upload/publications/hahlbohm2025inpc/inpc_full_eval.zip)__

> __A Bag of Tricks for Efficient Implicit Neural Point Clouds__  
> [Florian Hahlbohm](https://fhahlbohm.github.io), [Linus Franke](https://lfranke.github.io/), [Leon Overkämping](https://orcid.org/0009-0001-7756-6702), [Paula Wespe](https://orcid.org/0009-0007-2477-2725), [Susana Castillo](https://graphics.tu-bs.de/people/castillo), [Martin Eisemann](https://graphics.tu-bs.de/people/eisemann), [Marcus Magnor](https://graphics.tu-bs.de/people/magnor)  
 _Vision, Modeling and Visualization (VMV), September 2025_  
> __[Project page](https://fhahlbohm.github.io/inpc_v2/)&nbsp;| [Paper](http://arxiv.org/abs/2508.19140)__

This repository contains the combined implementation of both papers. For the reference implementation of the original
INPC paper, please refer to the previous release ([v1.0-3dv](https://github.com/nerficg-project/INPC/tree/v1.0-3dv)).

## Getting Started

### Requirements

- An NVIDIA GPU
- Linux (preferred) or Windows
- A recent CUDA SDK ([CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) recommended) and a compatible C++ compiler
- [Anaconda / Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) installed

### Setup

As a preparatory step, the [NeRFICG framework](https://github.com/nerficg-project/nerficg) needs to be set up.
Please follow the instructions in its README to set up a compatible Conda environment.

Now add INPC as an additional method by cloning this repository into `src/Methods/INPC`:
```shell
# HTTPS
git clone https://github.com/nerficg-project/INPC.git src/Methods/INPC
```
or
```shell
# SSH
git clone git@github.com:nerficg-project/INPC.git src/Methods/INPC
```

Finally, install all method-specific dependencies and CUDA extensions using:
```shell
python ./scripts/install.py -m INPC
```

_Note: The framework determines on-the-fly what extra modules need to be installed. Sometimes this causes unnecessary errors/warnings that can interrupt the installation process. In this case, first try to rerun the command before investigating the error in detail._

## Training and Inference

The INPC method is fully compatible with the NeRFICG scripts in the `scripts/` directory.
This includes config file generation via `create_config.py`,
training via `train.py`,
inference and performance benchmarking via `inference.py`,
metric calculation via `generate_tables.py`,
and live rendering via `gui.py`.
We also used these scripts for the experiments in our paper.

For detailed instructions, please refer to the [NeRFICG framework repository](https://github.com/nerficg-project/nerficg).

### Example Configuration Files

We provide an exemplary configuration file for the bicycle scene from the [Mip-NeRF360](https://jonbarron.info/mipnerf360/) dataset and recommend copying it to the `configs/` directory.
For the eight _intermediate_ scenes from the [Tanks and Temples](https://www.tanksandtemples.org/) dataset on which we evaluate our method in the paper, we used [our own calibration](https://cloud.tu-braunschweig.de/s/J5xYLLEdMnRwYPc) obtained using COLMAP.
For Tanks and Temples scenes we use the same dataloader as for Mip-NeRF360 scenes.

_Note:_ There will be no documentation for the method-specific configuration parameters under `TRAINING.XXXX`/`MODEL.XXXX`/`RENDERER.XXXX`.
Please conduct the code and/or our paper for understanding what they do.

### Using Custom Data

While this method is compatible with most of the dataloaders provided with the [NeRFICG framework](https://github.com/nerficg-project/nerficg),
we recommend using the Mip-NeRF360 loader (`src/Datasets/MipNeRF360.py`) for custom data.
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

To use it, simply modify `DATASET.PATH` near the bottom of one of the exemplary configuration files. Furthermore, you may want to modify the following configuration parameters:
- `TRAINING.DATA.PRELOADING_LEVEL`: Set this depending on the available RAM/VRAM as well as the size of your dataset (`2`: store training images in VRAM, `1`: in RAM, `0`: no preloading).
- `DATASET.IMAGE_SCALE_FACTOR`: Set this to `null` for using the original resolution or alternatively to a value between zero and one to train on downscaled images.
 The NeRFICG MipNeRF360 dataloader has special support for specific downscaling factors (`0.5`/`0.25`/`0.125`), where it will always load images from the respective directory (`images`/`images_2`/`images_4`/`images_8`).
 We recommend using this feature and downscaling custom data manually via, e.g., `mogrify -resize 50% *.jpg` for the best results.
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

### `torch.compile` Usage

For accelerated rendering during inference, we use [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) to JIT-compile the U-Net module.
This reduces rendering times by about 8-10 ms per frame, which is a considerable speedup especially for the rendering mode that uses a pre-extracted global point cloud.
Due to Windows-specific issues, U-Net compilation is disabled by default. You can try enabling it by setting `RENDERER.USE_COMPILED_UNET_INFERENCE: true` in the configuration file.

## License and Citation
This project is licensed under the MIT license (see [LICENSE](LICENSE)).

If you use this code for your research projects, please consider a citation:
```bibtex
@inproceedings{hahlbohm2025inpc,
  title     = {{INPC}: Implicit Neural Point Clouds for Radiance Field Rendering},
  author    = {Hahlbohm, Florian and Franke, Linus and Kappel, Moritz and Castillo, Susana and Eisemann, Martin and Stamminger, Marc and Magnor, Marcus},
  booktitle = {International Conference on 3D Vision (3DV)},
  year      = {2025},
  page      = {168--178},
  doi       = {10.1109/3DV66043.2025.00021},
  url       = {https://fhahlbohm.github.io/inpc/}
}
```
```bibtex
@inproceedings{hahlbohm2025inpcv2,
  title     = {A Bag of Tricks for Efficient Implicit Neural Point Clouds},
  author    = {Hahlbohm, Florian and Franke, Linus and Overkämping, Leon and Wespe, Paula and Castillo, Susana and Eisemann, Martin and Magnor, Marcus},
  booktitle = {Vision, Modeling and Visualization (VMV)},
  year      = {2025},
  doi       = {10.2312/vmv.20251229},
  url       = {https://fhahlbohm.github.io/inpc_v2/}
}
```
