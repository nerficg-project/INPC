"""INPC/Renderer.py: """

import torch
import Framework

from Datasets.utils import View
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.INPC.Model import INPCModel
from Methods.INPC.utils import adjust_for_unet, PreextractedPointCloud
from Methods.INPC.INPCCudaBackend import INPCRasterizer


@Framework.Configurable.configure(
    N_SAMPLES=8_388_352,  # 2 ** 23 - 256
    N_MULTISAMPLES=4,
    ENABLE_PREEXTRACTION=False,
    N_POINTS_PREEXTRACTION=67_108_864,  # 2 ** 26
    USE_EXPECTED_SAMPLES=True,
    LOW_VRAM_INFERENCE=True,
    USE_COMPILED_UNET_INFERENCE=False,  # disabled due to Windows-specific torch.compile issues, works on Linux
)
class INPCRenderer(BaseRenderer):
    """Renderer for the INPC method."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [INPCModel])
        self.rasterizer = INPCRasterizer()
        self.point_cloud = PreextractedPointCloud()
        self.compiled_unet = torch.compile(self.model.unet)

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        to_chw = True if benchmark else to_chw
        if self.point_cloud.is_set and (self.model.training or not self.ENABLE_PREEXTRACTION):
            self.point_cloud.clear()
            torch.cuda.empty_cache()
        padding_info = adjust_for_unet(view.camera)
        if self.model.training:
            outputs = self.render_image_train(view)
        elif self.ENABLE_PREEXTRACTION:
            if not self.point_cloud.is_set:
                self.preextract_point_cloud()
            outputs = self.render_image_preextracted(view, to_chw)
        elif self.LOW_VRAM_INFERENCE:
            outputs = self.render_image_inference_low_vram(view, to_chw)
        else:
            outputs = self.render_image_inference(view, to_chw)
        padding_info.unapply(view.camera)
        roi_slices = (slice(None),) + padding_info.roi_slices if to_chw or self.model.training else padding_info.roi_slices + (slice(None),)
        outputs['rgb'] = outputs['rgb'][roi_slices]
        return outputs

    def render_image_train(self, view: View) -> dict[str, torch.Tensor]:
        """Renders an image for optimization."""
        positions, indices = self.model.probability_field.generate_samples(self.N_SAMPLES, view=view, ensure_visibility=True)
        opacities, features = self.model.appearance_field(positions, batch_size=8_388_352)
        feature_image, alpha_image, blending_weights = self.rasterizer(view, positions, features, opacities)
        feature_image = feature_image + (1.0 - alpha_image) * self.model.background_model(view).permute(2, 0, 1)
        feature_image = self.model.unet(feature_image)
        rgb_image = feature_image if self.model.tone_mapper is None else self.model.tone_mapper(feature_image, view)
        return {'rgb': rgb_image.float(), 'indices': indices, 'blending_weights': blending_weights}

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if self.USE_EXPECTED_SAMPLES:
            positions = self.model.probability_field.generate_expected_samples(self.N_SAMPLES, self.N_MULTISAMPLES, view)
        else:
            positions = self.model.probability_field.generate_multisamples(self.N_SAMPLES, self.N_MULTISAMPLES, view)
        raw_features = self.model.appearance_field(positions, batch_size=16_777_216, return_raw=True)
        feature_image_bg = self.model.background_model(view)
        feature_image = self.rasterizer.render(view, positions, raw_features, feature_image_bg, self.N_MULTISAMPLES)
        if self.USE_COMPILED_UNET_INFERENCE:
            feature_image = self.compiled_unet(feature_image)
        else:
            feature_image = self.model.unet(feature_image)
        rgb_image = feature_image if self.model.tone_mapper is None else self.model.tone_mapper(feature_image, view)
        rgb_image = rgb_image if to_chw else rgb_image.permute(1, 2, 0)
        return {'rgb': rgb_image.float()}

    @torch.no_grad()
    def render_image_inference_low_vram(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given view. Multisampling is done sequentially to reduce VRAM usage."""
        feature_image_bg = self.model.background_model(view)
        feature_image = None
        for i in range(self.N_MULTISAMPLES):
            if self.USE_EXPECTED_SAMPLES:
                positions = self.model.probability_field.generate_expected_samples(self.N_SAMPLES, 1, view)
            else:
                positions = self.model.probability_field.generate_multisamples(self.N_SAMPLES, 1, view)
            raw_features = self.model.appearance_field(positions, batch_size=16_777_216, return_raw=True)
            if feature_image is None:
                feature_image = self.rasterizer.render(view, positions, raw_features, feature_image_bg, 1)
            else:
                feature_image += self.rasterizer.render(view, positions, raw_features, feature_image_bg, 1)
        feature_image = feature_image * (1.0 / self.N_MULTISAMPLES)
        if self.USE_COMPILED_UNET_INFERENCE:
            feature_image = self.compiled_unet(feature_image)
        else:
            feature_image = self.model.unet(feature_image)
        rgb_image = feature_image if self.model.tone_mapper is None else self.model.tone_mapper(feature_image, view)
        rgb_image = rgb_image if to_chw else rgb_image.permute(1, 2, 0)
        return {'rgb': rgb_image.float()}

    @torch.no_grad()
    def render_image_preextracted(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given view using a pre-extracted point cloud."""
        feature_image_bg = self.model.background_model(view)
        feature_image = self.rasterizer.render_preextracted(view, *self.point_cloud.get(), feature_image_bg)
        if self.USE_COMPILED_UNET_INFERENCE:
            feature_image = self.compiled_unet(feature_image)
        else:
            feature_image = self.model.unet(feature_image)
        rgb_image = feature_image if self.model.tone_mapper is None else self.model.tone_mapper(feature_image, view)
        rgb_image = rgb_image if to_chw else rgb_image.permute(1, 2, 0)
        return {'rgb': rgb_image.float()}

    @torch.no_grad()
    def preextract_point_cloud(self) -> None:
        """Pre-extracts a global point cloud for faster rendering."""
        positions = self.model.probability_field.extract_global(self.N_POINTS_PREEXTRACTION, True)
        opacities, features = self.model.appearance_field(positions, batch_size=16_777_216)
        self.point_cloud.set(positions, features, opacities)
        torch.cuda.empty_cache()

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {'rgb': outputs['rgb'].clamp_(0.0, 1.0)}
