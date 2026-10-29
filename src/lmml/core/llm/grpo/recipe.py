from lmml.core.llm.grpo.module import CausalLMGRPOLightningModule
from lmml.core.llm.rl_common.recipe import CausalLMRLRecipe


class CausalLMGRPORecipe(CausalLMRLRecipe):
    module_class = CausalLMGRPOLightningModule


__all__ = ["CausalLMGRPORecipe"]
