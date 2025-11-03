import os
import datetime
from dotenv import load_dotenv


try:
    load_dotenv()
except Exception as e:
    print(f"Error loading .env file: {e}")

# ---------- Helper Functions ----------
def get_email() -> str:
    """
    Get the email username from environment variable UVG_EMAIL.
    Returns:
        str: Email username (part before @uvg.edu.gt)
    """
    email = os.getenv("UVG_EMAIL")
    if email:
        return email.split("@")[0]
    else:
        raise ValueError("ERROR: UVG_EMAIL not set in environment variables")
    
def get_timestamp() -> str:
    """
    Get the current timestamp in YYYYMMDD_HHMMSS format.
    Returns:
        str: Current timestamp as a string.
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

