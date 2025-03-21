# -- coding: utf-8 --

"""INPC/Model.py: INPC model implementation."""

import Framework

from Methods.Base.Model import BaseModel
from Methods.INPC.Modules import *
from Methods.INPC.utils import LRDecayPolicy


@Framework.Configurable.configure(
    USE_TONE_MAPPER=True,
    USE_FFC_BOCK=True,
)
class INPCModel(BaseModel):
    """Defines the INPC model."""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self.probability_field = None
        self.appearance_field = None
        self.background_model = None
        self.unet = None
        self.tone_mapper = None

    def build(self) -> 'INPCModel':
        """Builds the model."""
        self.probability_field = ProbabilityField()
        self.appearance_field = AppearanceField()
        self.background_model = NeuralEnvironmentMap()
        self.unet = UNet(self.USE_FFC_BOCK)
        if self.USE_TONE_MAPPER:
            self.tone_mapper = ToneMapper()
        return self

    def get_optimizer_param_groups(self, n_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the optimizer parameter groups and learning rate schedulers."""
        learnable_components = [self.appearance_field, self.background_model, self.unet]
        if self.tone_mapper is not None:
            learnable_components.append(self.tone_mapper)
        param_groups, schedulers = zip(*(component.get_optimizer_param_groups(n_iterations) for component in learnable_components))
        return sum(param_groups, []), sum(schedulers, [])
