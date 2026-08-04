from toycc.passes.fusion import FusionPass, pass_pipeline
from toycc.passes.layout import LayoutPass
from toycc.passes.constfold import ConstantFoldPass
from toycc.passes.dce import DCEPass
from toycc.passes.pipeline import run_passes

__all__ = ["FusionPass", "LayoutPass", "ConstantFoldPass", "DCEPass",
           "pass_pipeline", "run_passes"]
