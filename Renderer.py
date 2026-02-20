"""INPC/Renderer.py: Renderer for the INPC method."""

import torch
import Framework

from Datasets.utils import View
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.INPC.Model import INPCModel
from Methods.INPC.utils import adjust_for_unet, PreextractedPointCloud, MultisamplingRingBuffer
from Methods.INPC.INPCCudaBackend import INPCRasterizer
from Visual.utils import apply_color_map


@Framework.Configurable.configure(
    SPLATTING_MODE=0,  # 0: Bilinear, 1: Gaussian
    N_SAMPLES=8_388_352,  # 2 ** 23 - 256
    N_MULTISAMPLES=4,
    ENABLE_PREEXTRACTION=False,
    N_POINTS_PREEXTRACTION=33_554_432,  # 2 ** 25
    USE_EXPECTED_SAMPLES=True,
    USE_COMPILED_UNET_INFERENCE=False,  # disabled due to Windows-specific torch.compile issues, works on Linux
    USE_DISTILLED_BG_TEXTURE_INFERENCE=False,  # set to True for performance benchmarking
    USE_RINGBUFFER_INFERENCE=False,
    AVERAGE_WITH_PREMULTIPLIED_ALHPA=True,
    SIGMA_WORLD_SCALE=1.0,
    SIGMA_CUTOFF=3.0,
)
class INPCRenderer(BaseRenderer):
    """Renderer for the INPC method."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [INPCModel])
        if self.SPLATTING_MODE not in [0, 1]:
            raise Framework.RendererError('Invalid splatting mode')
        self.rasterizer = INPCRasterizer()
        self.point_cloud = PreextractedPointCloud()
        self.compiled_unet = torch.compile(self.model.unet)
        self.ring_buffer = MultisamplingRingBuffer()

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        # TODO: rendering could be optimized for benchmarking frame times
        # clear pre-extracted point cloud if not needed
        if self.point_cloud.is_set and (self.model.training or not self.ENABLE_PREEXTRACTION):
            self.point_cloud.clear()
            torch.cuda.empty_cache()

        # apply padding to avoid padding in the UNet
        padding_info = adjust_for_unet(view.camera)

        # call appropriate rendering function
        if self.model.training:
            outputs = self.render_image_train(view)
        elif self.ENABLE_PREEXTRACTION:
            if not self.point_cloud.is_set:
                self.preextract_point_cloud()
            outputs = self.render_image_preextracted(view, to_chw)
        elif self.USE_RINGBUFFER_INFERENCE:
            outputs = self.render_image_inference_ringbuffer(view, to_chw)
        else:
            outputs = self.render_image_inference(view, to_chw)

        # remove padding
        padding_info.unapply(view.camera)

        # return outputs without padding
        roi_slices = (slice(None),) + padding_info.roi_slices if to_chw or self.model.training else padding_info.roi_slices + (slice(None),)
        for key in ['rgb', 'depth', 'alpha', 'alpha_sum', 'feature_image']:
            if key in outputs:
                outputs[key] = outputs[key][roi_slices]
        return outputs

    def render_image_train(self, view: View) -> dict[str, torch.Tensor]:
        """Renders an image for optimization."""
        positions, indices = self.model.probability_field.generate_training_samples(self.N_SAMPLES, view)
        features, opacities = self.model.appearance_field(positions, batch_size=8_388_352)
        feature_image, alpha_image, blending_weights = self.rasterizer(view, self.SPLATTING_MODE, positions, features, opacities)
        feature_image = feature_image + (1.0 - alpha_image) * self.model.background_model(view)
        hdr_image = self.model.unet(feature_image)
        if self.model.tone_mapper is None:
            rgb_image = hdr_image.float()
        else:
            rgb_image = self.model.tone_mapper(hdr_image, view).float()
        return {
            'rgb': rgb_image,
            'indices': indices,
            'blending_weights': blending_weights
        }

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given view. Multisampling is done sequentially to reduce VRAM usage."""
        n_multisamples = max(1, self.N_MULTISAMPLES)
        feature_image_bg = self.model.background_model(view, self.USE_DISTILLED_BG_TEXTURE_INFERENCE)
        feature_image = depth_image = alpha_image = None
        for i in range(n_multisamples):
            if self.USE_EXPECTED_SAMPLES:
                positions = self.model.probability_field.generate_expected_samples(self.N_SAMPLES, 1, view)
            else:
                positions = self.model.probability_field.generate_multisamples(self.N_SAMPLES, 1, view)
            raw_features = self.model.appearance_field(positions, batch_size=16_777_216, return_raw=True)
            feature_image_, depth_image_, alpha_image_ = self.rasterizer.render(view, self.SPLATTING_MODE, positions, raw_features, sigma_world_scale=self.SIGMA_WORLD_SCALE, sigma_cutoff=self.SIGMA_CUTOFF)
            if not self.AVERAGE_WITH_PREMULTIPLIED_ALHPA and n_multisamples > 1:
                feature_image_ = torch.where(alpha_image_ > 0.0, feature_image_ / alpha_image_, 0.0)
            if i == 0:
                feature_image = feature_image_
                depth_image = depth_image_
                alpha_image = alpha_image_
            else:
                feature_image += feature_image_
                depth_image += depth_image_
                alpha_image += alpha_image_
        multisampling_factor = 1.0 / n_multisamples
        feature_image *= multisampling_factor
        depth_image *= multisampling_factor
        alpha_image_sum = alpha_image.clamp(0.0, 1.0)
        alpha_image *= multisampling_factor
        if not self.AVERAGE_WITH_PREMULTIPLIED_ALHPA and n_multisamples > 1:
            feature_image *= alpha_image
        feature_image += (1.0 - alpha_image) * feature_image_bg
        if self.USE_COMPILED_UNET_INFERENCE:
            hdr_image = self.compiled_unet(feature_image)
        else:
            hdr_image = self.model.unet(feature_image)
        if self.model.tone_mapper is None:
            rgb_image = hdr_image.float()
        else:
            rgb_image = self.model.tone_mapper(hdr_image, view).float()
        return {
            'rgb': rgb_image if to_chw else rgb_image.permute(1, 2, 0),
            'depth': depth_image if to_chw else depth_image.permute(1, 2, 0),
            'alpha': alpha_image if to_chw else alpha_image.permute(1, 2, 0),
            'alpha_sum': alpha_image_sum if to_chw else alpha_image_sum.permute(1, 2, 0),
            'feature_image': feature_image if to_chw else feature_image.permute(1, 2, 0),  # for extracting pre-training data
        }

    @torch.no_grad()
    def render_image_inference_ringbuffer(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given view. Uses a ring buffer approach to reuse samples from previous frames."""
        n_multisamples = max(1, self.N_MULTISAMPLES)
        feature_image_bg = self.model.background_model(view, self.USE_DISTILLED_BG_TEXTURE_INFERENCE)
        feature_image = depth_image = alpha_image = None
        if self.USE_EXPECTED_SAMPLES:
            positions = self.model.probability_field.generate_expected_samples(self.N_SAMPLES, 1, view)
        else:
            positions = self.model.probability_field.generate_multisamples(self.N_SAMPLES, 1, view)
        raw_features = self.model.appearance_field(positions, batch_size=16_777_216, return_raw=True)
        self.ring_buffer.append(positions, raw_features, n_multisamples)

        for i in range(n_multisamples):
            positions, raw_features = self.ring_buffer.get_next()
            feature_image_, depth_image_, alpha_image_ = self.rasterizer.render(view, self.SPLATTING_MODE, positions, raw_features, sigma_world_scale=self.SIGMA_WORLD_SCALE, sigma_cutoff=self.SIGMA_CUTOFF)
            if not self.AVERAGE_WITH_PREMULTIPLIED_ALHPA and n_multisamples > 1:
                feature_image_ = torch.where(alpha_image_ > 0.0, feature_image_ / alpha_image_, 0.0)
            if i == 0:
                feature_image = feature_image_
                depth_image = depth_image_
                alpha_image = alpha_image_
            else:
                feature_image += feature_image_
                depth_image += depth_image_
                alpha_image += alpha_image_
        multisampling_factor = 1.0 / n_multisamples
        feature_image *= multisampling_factor
        depth_image *= multisampling_factor
        alpha_image_sum = alpha_image.clamp(0.0, 1.0)
        alpha_image *= multisampling_factor
        if not self.AVERAGE_WITH_PREMULTIPLIED_ALHPA and n_multisamples > 1:
            feature_image *= alpha_image
        feature_image += (1.0 - alpha_image) * feature_image_bg
        if self.USE_COMPILED_UNET_INFERENCE:
            hdr_image = self.compiled_unet(feature_image)
        else:
            hdr_image = self.model.unet(feature_image)
        if self.model.tone_mapper is None:
            rgb_image = hdr_image.float()
        else:
            rgb_image = self.model.tone_mapper(hdr_image, view).float()
        return {
            'rgb': rgb_image if to_chw else rgb_image.permute(1, 2, 0),
            'depth': depth_image if to_chw else depth_image.permute(1, 2, 0),
            'alpha': alpha_image if to_chw else alpha_image.permute(1, 2, 0),
            'alpha_sum': alpha_image_sum if to_chw else alpha_image_sum.permute(1, 2, 0),
        }

    @torch.no_grad()
    def render_image_preextracted(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given view using a pre-extracted point cloud."""
        feature_image_bg = self.model.background_model(view, self.USE_DISTILLED_BG_TEXTURE_INFERENCE)
        feature_image, depth_image, alpha_image = self.rasterizer.render_preextracted(view, self.SPLATTING_MODE, *self.point_cloud.get(), sigma_world_scale=self.SIGMA_WORLD_SCALE, sigma_cutoff=self.SIGMA_CUTOFF)
        feature_image += (1.0 - alpha_image) * feature_image_bg
        if self.USE_COMPILED_UNET_INFERENCE:
            hdr_image = self.compiled_unet(feature_image)
        else:
            hdr_image = self.model.unet(feature_image)
        if self.model.tone_mapper is None:
            rgb_image = hdr_image.float()
        else:
            rgb_image = self.model.tone_mapper(hdr_image, view).float()
        return {
            'rgb': rgb_image if to_chw else rgb_image.permute(1, 2, 0),
            'depth': depth_image if to_chw else depth_image.permute(1, 2, 0),
            'alpha': alpha_image if to_chw else alpha_image.permute(1, 2, 0),
        }

    @torch.no_grad()
    def preextract_point_cloud(self) -> None:
        """Pre-extracts a global point cloud for faster rendering."""
        positions = self.model.probability_field.extract_global(self.N_POINTS_PREEXTRACTION, True)
        features, opacities = self.model.appearance_field(positions, batch_size=16_777_216)
        self.point_cloud.set(positions, features, opacities)
        torch.cuda.empty_cache()

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {
            'rgb': outputs['rgb'],
            'depth': apply_color_map(
                color_map='SPECTRAL',
                image=outputs['depth'],
                min_max=None,
                mask=outputs['depth'] > 0.0,
                interpolate=True
            ),
            'alpha': outputs['alpha'].expand(outputs['rgb'].shape),
            'alpha_sum': outputs['alpha_sum'].expand(outputs['rgb'].shape),
        }
