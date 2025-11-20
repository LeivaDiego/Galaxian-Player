from typing import Protocol

# ---------- Interfaz de Política ----------
class Policy(Protocol):
    """
    Interfaz para políticas de acción.
    """
    def __call__(self, observation, info:dict, action_space) -> int: ...


# ---------- Políticas ----------
class RandomPolicy:
    """
    Politica de ejemplo que elige acciones al azar.
    """
    def __call__(self, observation, info, action_space) -> int:
        return action_space.sample()

# TODO: politicas de los modelos a implementar