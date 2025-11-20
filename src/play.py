import os
import datetime
from dotenv import load_dotenv
from typing import Protocol, Optional
from pathlib import Path
import gymnasium as gym
import numpy as np
import imageio.v2 as imageio
import ale_py
import cv2

# Cargar variables de entorno desde .env
try:
    load_dotenv()
except Exception as e:
    print(f"Error loading .env file: {e}")

gym.register_envs(ale_py)

# ---------- Funciones de ayuda ----------
def get_email() -> str:
    """
    Obtener el nombre de usuario del correo electrónico de la 
        variable de entorno UVG_EMAIL.
    Returns:
        str: Nombre de usuario del correo electrónico (parte antes de @uvg.edu.gt)
    """
    email = os.getenv("UVG_EMAIL")
    if email:
        return email.split("@")[0]
    else:
        raise ValueError("UVG_EMAIL no está definida en las variables de entorno.")
    
def get_timestamp() -> str:
    """
    Obtener la marca de tiempo actual en formato YYYYMMDDHHMM.
    Returns:
        str: Marca de tiempo actual como cadena.
    """
    return datetime.datetime.now().strftime("%Y%m%d%H%M")

def get_seed() -> Optional[int]:
    """
    Obtener la semilla para el entorno Galaxian desde la variable de entorno GALAXIAN_SEED.
    Returns:
        Optional[int]: Semilla como entero si está definida y es válida, de lo contrario None.
    """
    seed = os.getenv("GALAXIAN_SEED").strip()
    return int(seed) if seed.isdigit() else None
    
# ---------- Interfaz de Política ----------

class Policy(Protocol):
    """
    Interfaz para políticas de acción.
    """
    def __call__(self, observation, info:dict, action_space) -> int: ...

class RandomPolicy:
    """
    Politica de ejemplo que elige acciones al azar.
    """
    def __call__(self, observation, info, action_space) -> int:
        return action_space.sample()

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
    frames: list[np.ndarray] = []
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    # Frame inicial
    if isinstance(observation, np.ndarray):
        frames.append(observation.astype(np.uint8))
    
    # Ejecutar el episodio
    print("[INFO] Iniciando episodio...")
    while not (terminated or truncated):
        # Tomar acción de la política
        action = int(policy(observation, info, env.action_space))
        # Validar acción
        if action < 0 or action >= env.action_space.n:
            raise ValueError(f"Acción inválida: {action}. Debe estar en [0, {env.action_space.n - 1}]")
    
        # Ejecutar acción
        observation, reward, terminated, truncated, info = env.step(action)
        # Acumular recompensa
        total_reward += reward
        steps += 1

        if steps % 500 == 0:  # cada 500 pasos muestra avance
            print(f"  - Progreso: {steps} pasos, recompensa parcial = {total_reward:.0f}")
        
        # Almacenar frame
        if isinstance(observation, np.ndarray):
            frames.append(observation.astype(np.uint8))

    print(f"[INFO] Episodio finalizado. Pasos: {steps}, Recompensa total: {total_reward:.0f}")
    return frames, int(round(total_reward))

# ---------- Grabación del Episodio ----------
def save_video(frames: list[np.ndarray], output_path: Path, fps: int = 30, scale: float = 2.0) -> None:
    """
    Guardar una lista de frames como un archivo de video MP4.
    Args:
        frames (list of np.ndarray): Lista de frames a guardar.
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
def record_episode(policy:Policy, video_dir:Optional[str]=None) -> None:
    """
    Grabar un episodio completo utilizando la política proporcionada.
    Args:
        policy (Policy): Política para seleccionar acciones.
        video_dir (Optional[str]): Directorio donde guardar el video. 
        Si es None, se usa la variable de entorno VIDEO_DIR o "videos".
    Returns:
        Path: Ruta del video guardado.
    """
    # Configuraciones desde variables de entorno
    print("[INFO] Iniciando grabación de episodio...")
    email = get_email()
    timestamp = get_timestamp()
    env_video_dir = os.getenv("VIDEO_DIR", "videos")
    video_dir = video_dir if video_dir is not None else env_video_dir
    video_fps = int(os.getenv("VIDEO_FPS", "30"))
    
    # Preparar ruta de salida
    output_path = Path(video_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Crear entorno
    print("[INFO] Creando entorno ALE/Galaxian-v5...")
    galaxian = gym.make('ALE/Galaxian-v5', render_mode='rgb_array')
    
    # Ejecutar episodio
    try:
        frames, score = run_episode(galaxian, policy)
    finally:
        galaxian.close()

    # Guardar video
    video_filename = f"{email}_{timestamp}_{score}.mp4"
    output_path = Path (video_dir) / video_filename
    save_video(frames, output_path, fps=video_fps)
    
    return output_path

# ---------- Ejemplo de Uso ----------
if __name__ == "__main__":
    # Política de ejemplo: acciones aleatorias
    policy = RandomPolicy()
    
    # Grabar episodio
    video_path = record_episode(policy)