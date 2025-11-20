from typing import Protocol
import torch
import numpy as np

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


class DQNPolicy:
    """
    Policy para usar un modelo DQN / DuelingDQN entrenado.
    Asume que observation ya viene preprocesada por AtariPreprocessing+FrameStackObservation,
    es decir, algo tipo (stack_size, 84, 84) o similar.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, epsilon: float = 0.0):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.model.eval()

    def __call__(self, observation, info, action_space) -> int:
        # Exploración epsilon-greedy
        if np.random.rand() < self.epsilon:
            return int(action_space.sample())

        # Convertir observation a np.ndarray sin copiar si no es necesario
        obs = np.array(observation, copy=False) 

        # Asegurar formato (C,H,W)
        if obs.ndim == 3:
            # Gymnasium Atari con procesamiento suele devolver (84, 84, stack) o (stack, 84, 84)
            # Si el canal está al final, se mueve al principio:
            if obs.shape[-1] in (1, 4) and obs.shape[0] == 84:
                # (84,84,C) -> (C,84,84)
                obs = np.transpose(obs, (2, 0, 1))
        elif obs.ndim == 2:
            # solo un frame (84,84) -> (1,84,84)
            obs = obs[None, ...]

        # batch: (1, C, H, W)
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(obs_t)        # (1, num_actions)
            action = int(q_values.argmax(dim=1).item())

        return action