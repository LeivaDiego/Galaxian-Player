import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

# Registrar los entornos de ALE en Gymnasium
gym.register_envs(ale_py)


def make_galaxian_env(
    render_mode: str | None = None,
    deepmind_wrappers: bool = False,
    frame_stack: int = 4,
) -> gym.Env:
    """
    Crear un entorno de Gymnasium para el juego Galaxian con opciones de preprocesamiento.
    Args:
        render_mode (str | None): Modo de renderizado del entorno. Puede ser 'rgb_array', 'human', o None.
        deepmind_wrappers (bool): Si es True, aplicar preprocesamiento estilo DeepMind (escala de grises, recorte, apilamiento de frames).
        frame_stack (int): Número de frames a apilar si se usan los wrappers de DeepMind.
    Returns:
        gym.Env: Entorno de Gymnasium configurado para Galaxian.
    """
    if deepmind_wrappers:
        # Crear entorno base
        env = gym.make(
            "ALE/Galaxian-v5", 
            render_mode=render_mode, 
            frameskip=1
        )
        # Aplicar wrappers de DeepMind
        env = AtariPreprocessing(
            env,
            grayscale_obs=True,
            frame_skip=4,
            screen_size=84,
        )
        # Apilar frames
        env = FrameStackObservation(
            env=env, 
            stack_size=frame_stack
        )
    else:
        # Crear entorno sin preprocesamiento adicional
        env = gym.make(
            "ALE/Galaxian-v5", 
            render_mode=render_mode
        )

    return env
