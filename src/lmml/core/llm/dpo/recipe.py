from lmml.core.llm.dpo.module import CausalLMDPOLightningModule
from lmml.core.llm.rl_common.recipe import CausalLMRLRecipe


class CausalLMDPORecipe(CausalLMRLRecipe):
    module_class = CausalLMDPOLightningModule


__all__ = ["CausalLMDPORecipe"]
