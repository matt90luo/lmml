from lmml.core.llm.ppo.module import CausalLMPPOLightningModule
from lmml.core.llm.rl_common.recipe import CausalLMRLRecipe


class CausalLMPPORecipe(CausalLMRLRecipe):
    module_class = CausalLMPPOLightningModule


__all__ = ["CausalLMPPORecipe"]
