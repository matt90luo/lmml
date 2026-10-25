"""lmml: CLI toolkit for RL / ML training (Hydra + Lightning + TensorBoard)."""

from lmml.__about__ import __version__

__all__ = ["__version__"]


def _register_envs() -> None:
    import gymnasium as gym

    if "lmml/GridWorld-v0" not in gym.registry:
        gym.register(id="lmml/GridWorld-v0", entry_point="lmml.envs.gridworld:GridWorldEnv")


_register_envs()
