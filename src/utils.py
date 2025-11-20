import os
import datetime
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno desde .env al importar este módulo
try:
    load_dotenv()
except Exception as e:
    print(f"Error loading .env file: {e}")

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

def get_video_dir(default: str = "videos") -> str:
    """
    Obtener el directorio para guardar videos desde la variable de entorno VIDEO_DIR.
    Args:
        default (str): Valor por defecto si VIDEO_DIR no está definida.
    Returns:
        str: Directorio para guardar videos.
    """
    return os.getenv("VIDEO_DIR", default)

def get_video_fps(default: str = "30") -> int:
    """
    Obtener los fotogramas por segundo para los videos desde la variable de entorno VIDEO_FPS.
    Args:
        default (str): Valor por defecto si VIDEO_FPS no está definida.
    Returns:
        int: Fotogramas por segundo para los videos.
    """
    return int(os.getenv("VIDEO_FPS", default))