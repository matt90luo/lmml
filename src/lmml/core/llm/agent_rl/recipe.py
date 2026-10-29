from lmml.core.llm.agent_rl.module import CausalLMAgentRLLightningModule
from lmml.core.llm.rl_common.recipe import CausalLMRLRecipe


class CausalLMAgentRLRecipe(CausalLMRLRecipe):
    module_class = CausalLMAgentRLLightningModule


__all__ = ["CausalLMAgentRLRecipe"]
