from pathlib import Path
from typing import Optional, List

import numpy as np
import imageio.v2 as imageio
import cv2

from utils import get_email, get_timestamp, get_seed, get_video_dir, get_video_fps
from gym_envs import make_galaxian_env
from policies import Policy


def capture_frame(env) -> np.ndarray | None:
    """
    Capturar el frame actual del entorno si está disponible.
    Args:
        env (gym.Env): Entorno de Gymnasium.
    Returns:
        np.ndarray | None: Frame capturado como un array numpy, o None si no está disponible
    """
    try:
        frame = env.render()
    except Exception:
        return None
    return frame


# ---------- Ejecución del Episodio ----------
def run_episode(env, policy:Policy) -> int:
    """
    Ejecutar un episodio en el entorno dado utilizando la política proporcionada.
    Args:
        env (gym.Env): Entorno de Gymnasium.
        policy (Policy): Política para seleccionar acciones.
    Returns:
        tuple: (frames, total_reward)
            frames (list of np.ndarray): Lista de frames capturados durante el episodio.
            total_reward (int): Recompensa total obtenida durante el episodio.
    """
    # Inicializar el entorno y variables del episodio
    print("[INFO] Reiniciando entorno Galaxian...")
    observation, info = env.reset(seed=get_seed())
    frames: List[np.ndarray] = []
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    # Frame inicial (de renderizado no de observación)
    frame = capture_frame(env)
    if isinstance(frame, np.ndarray):
        frames.append(frame.astype(np.uint8))
    
    # Ejecutar el episodio
    print("[INFO] Iniciando episodio...")
    while not (terminated or truncated):
        # Tomar acción de la política
        action = int(policy(observation, info, env.action_space))

        # Validar acción
        if action < 0 or action >= env.action_space.n:
            raise ValueError(f"Acción inválida: {action}. Debe estar en [0, {env.action_space.n - 1}]")
    
        # Avanzar un paso en el entorno
        observation, reward, terminated, truncated, info = env.step(action)
        # Acumular recompensa
        total_reward += reward
        steps += 1

        if steps % 500 == 0:  # cada 500 pasos muestra avance
            print(f"  - Progreso: {steps} pasos, recompensa parcial = {total_reward:.0f}")
        
        # Almacenar frame de renderizado
        frame = capture_frame(env)
        if isinstance(frame, np.ndarray):
            frames.append(frame.astype(np.uint8))

    print(f"[INFO] Episodio finalizado. Pasos: {steps}, Recompensa total: {total_reward:.0f}")
    return frames, int(round(total_reward))


# ---------- Grabación del Episodio ----------
def save_video(frames: List[np.ndarray], output_path: Path, fps: int = 30, scale: float = 1.0) -> None:
    """
    Guardar una lista de frames como un archivo de video MP4.
    Args:
        frames (List[np.ndarray]): Lista de frames a guardar.
        output_path (Path): Ruta del archivo de salida.
        fps (int): Fotogramas por segundo del video.
        scale (float): Factor de escala para redimensionar los frames.
    """
    print(f"[INFO] Guardando video ({len(frames)} frames, escala x{scale})...")
    # Asegurarse de que el directorio de salida exista
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar video usando imageio
    with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8,  macro_block_size=1) as video_writer:
        for frame in frames:
            if scale != 1.0:
                height, width = frame.shape[:2]
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
            video_writer.append_data(frame)
    print(f"[INFO] Video guardado en: {output_path}")


# ---------- Función Principal ----------
def record_episode(policy:Policy, deepmind_wrappers: bool = False, video_dir:Optional[str]=None, video_name: Optional[str]=None) -> None:
    """
    Grabar un episodio completo utilizando la política proporcionada.
    Args:
        policy (Policy): Política para seleccionar acciones.
        deepmind_wrappers (bool): Indica si se deben usar los wrappers de DeepMind para el entorno.
        video_dir (Optional[str]): Directorio donde guardar el video. 
        Si es None, se usa la variable de entorno VIDEO_DIR o "videos".
    Returns:
        Path: Ruta del video guardado.
    """
    # Configuraciones desde variables de entorno
    print("[INFO] Iniciando grabación de episodio...")
    email = get_email()
    timestamp = get_timestamp()
    env_video_dir = get_video_dir()
    video_fps = get_video_fps()
    video_dir = video_dir if video_dir is not None else env_video_dir
    
    # Preparar ruta de salida
    output_path = Path(video_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Crear entorno
    print("[INFO] Creando entorno ALE/Galaxian-v5...")
    galaxian = make_galaxian_env(
        render_mode='rgb_array',
        deepmind_wrappers=deepmind_wrappers
    )
    
    # Ejecutar episodio
    try:
        frames, score = run_episode(galaxian, policy)
    finally:
        galaxian.close()

    # Guardar video
    if video_name is not None:
        video_filename = video_name
    else:
        video_filename = f"{email}_{timestamp}_{score}.mp4"
    output_path = Path (video_dir) / video_filename
    save_video(frames, output_path, fps=video_fps)
    
    return output_path