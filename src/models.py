from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out_size(h: int, w: int) -> Tuple[int, int]:
    """
    Calcula el tamaño (h,w) de la salida del bloque convolucional
    dado un tamaño de entrada (h,w) para la arquitectura DQN/Nature2015.
    3 capas convolucionales con:
    - Conv1: kernel=8, stride=4
    - Conv2: kernel=4, stride=2
    - Conv3: kernel=3, stride=1
    Args:
        h (int): altura de la entrada.
        w (int): ancho de la entrada.
    Returns:
        Tuple[int, int]: altura y ancho de la salida.
    """
    h = (h - 8) // 4 + 1  # conv1
    w = (w - 8) // 4 + 1
    h = (h - 4) // 2 + 1  # conv2
    w = (w - 4) // 2 + 1
    h = (h - 3) // 1 + 1  # conv3
    w = (w - 3) // 1 + 1
    return h, w


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) básico según la arquitectura de Nature 2015.
    - 3 capas convolucionales.
    - 2 capas fully connected.
    - Entrada: (C, 84, 84)
    - Salida: Q(s,a) para cada acción a.
    """

    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        h, w = conv_out_size(84, 84)  # -> 7x7
        conv_out_dim = 64 * h * w      # 64*7*7 = 3136

        self.fc1 = nn.Linear(conv_out_dim, 512)
        self.fc_out = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor (B, C, 84, 84), típicamente uint8 o float en [0,255].
        """
        # Normalizar a [0,1] como en muchos impl. de Atari
        x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        q_values = self.fc_out(x)
        return q_values


class DuelingDQN(nn.Module):
    """
    Deep Dueling Q-Network según la arquitectura de Wang et al. 2016.
    - 3 capas convolucionales.
    - Dos ramas fully connected para valor y ventaja.
    - Entrada: (C, 84, 84)
    - Salida: Q(s,a) para cada acción a.
    """

    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        h, w = conv_out_size(84, 84)
        conv_out_dim = 64 * h * w  # 3136

        # Stream de Valor V(s)
        self.fc_val = nn.Linear(conv_out_dim, 512)
        self.val_out = nn.Linear(512, 1)

        # Stream de Ventaja A(s,a)
        self.fc_adv = nn.Linear(conv_out_dim, 512)
        self.adv_out = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalizar a [0,1]
        x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        val = F.relu(self.fc_val(x))
        val = self.val_out(val)           # (B, 1)

        adv = F.relu(self.fc_adv(x))
        adv = self.adv_out(adv)           # (B, num_actions)

        # Combinar en Q(s,a)
        adv_mean = adv.mean(dim=1, keepdim=True)
        q_values = val + (adv - adv_mean)   # broadcasting
        return q_values
