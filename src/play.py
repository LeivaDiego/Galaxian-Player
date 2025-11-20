# play_trained.py
import os
import torch
import numpy as np

from gym_envs import make_galaxian_env
from models import DQN, DuelingDQN
from policies import DQNPolicy
from recording import record_episode
from utils import get_seed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ===== 1) Crear un env SOLO para conocer in_channels y num_actions =====
    # Usamos los mismos wrappers que en entrenamiento
    tmp_env = make_galaxian_env(
        render_mode=None,
        deepmind_wrappers=True,
        frame_stack=4,
    )
    seed = get_seed()
    obs, _ = tmp_env.reset(seed=seed)

    # Convertir obs a (C,H,W) para saber in_channels
    arr = np.array(obs, copy=False)
    if arr.ndim == 3:
        # (H,W,C) -> (C,H,W) si hace falta
        if arr.shape[0] == 84 and arr.shape[1] == 84:
            arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 2:
        arr = arr[None, ...]
    else:
        raise ValueError(f"Obs shape inesperado: {arr.shape}")

    in_channels = arr.shape[0]
    num_actions = tmp_env.action_space.n
    tmp_env.close()

    print(f"in_channels={in_channels}, num_actions={num_actions}")

    # ===== 2) Reconstruir modelo y cargar pesos =====
    # Cambia DQN por DuelingDQN si guardaste un dueling
    model = DQN(in_channels=in_channels, num_actions=num_actions).to(device)

    checkpoint_path = "models/dqn_galaxian_medium.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Si guardaste solo state_dict:
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.eval()
    print(f"Modelo cargado desde {checkpoint_path}")

    # ===== 3) Crear policy con epsilon pequeño (casi greedy) =====
    policy = DQNPolicy(model=model, device=device, epsilon=0.01)

    # ===== 4) Grabar episodio =====
    # IMPORTANTE: aquí lo ideal es que record_episode cree un env con deepmind_wrappers=True,
    # para que la observation que ve la policy coincida con el preprocesamiento del entrenamiento.
    video_path = record_episode(policy, video_dir="videos", deepmind_wrappers=True, frame_stack=4)  # ver nota abajo
    print("Video guardado en:", video_path)


if __name__ == "__main__":
    main()
