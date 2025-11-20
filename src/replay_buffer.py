from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    Buffer de replay para almacenar transiciones (s,a,r,s',done).
    Permite muestrear batches aleatorios para entrenamiento.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
        dtype: np.dtype = np.uint8,
    ):
        """
        Inicializa el replay buffer.
        Args:
            capacity: número máximo de transiciones a almacenar.
            obs_shape: (4, 84, 84) para Atari con 4 frames apilados.
            device: dispositivo de PyTorch para tensores muestreados.
            dtype: tipo para almacenar estados (uint8 para pixeles crudos).
        """
        self.capacity = capacity
        self.device = device

        # Prealocar memoria para las transiciones
        self.states = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

        # Puntero circular y bandera de buffer lleno
        self.pos = 0
        self.full = False


    def __len__(self) -> int:
        """
        Retorna el número actual de transiciones almacenadas.
        """
        return self.capacity if self.full else self.pos


    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Agrega una transición al buffer.
        Args:
            state: estado actual.
            action: acción tomada.
            reward: recompensa recibida.
            next_state: estado siguiente.
            done: si el episodio terminó.
        """
        idx = self.pos

        # Almacenar la transición en la posición actual
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        # Actualizar la posición circularmente
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True


    def sample(self, batch_size: int):
        """
        Muestra un batch aleatorio de transiciones.
        Args:
            batch_size: número de transiciones a muestrear.
        Returns:
            Tupla de tensores: (states, actions, rewards, next_states, dones)
        """
        # Verificar que hay suficientes muestras
        size = len(self)
        if size < batch_size:
            raise ValueError(
                f"No hay suficientes muestras en el buffer: {size} < {batch_size}"
            )

        # Muestrear índices aleatorios
        indices = np.random.randint(0, size, size=batch_size)

        # Convertir a tensores de PyTorch en el dispositivo correcto
        states = torch.as_tensor(self.states[indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        dones = torch.as_tensor(self.dones[indices].astype(np.float32), device=self.device)

        return states, actions, rewards, next_states, dones
