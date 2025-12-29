from pathlib import Path
from dotenv import load_dotenv
import os

current_dir = Path.cwd() / "ui" # Simulate app.py location relative to CWD
# But app.py uses __file__.
# Let's approximate.
project_root = Path.cwd()
env_path = project_root.parent / ".env"

print(f"Loading .env from: {env_path}")
try:
    load_dotenv(env_path, verbose=True)
    print("Success!")
    print(f"OPENROUTER_API_KEY: {os.getenv('OPENROUTER_API_KEY')}")
except Exception as e:
    print(f"Error: {e}")
