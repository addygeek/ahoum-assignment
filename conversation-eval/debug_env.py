from pathlib import Path
import sys

def check_file(path_str):
    p = Path(path_str).resolve()
    print(f"\nChecking: {p}")
    if not p.exists():
        print("  [DOES NOT EXIST]")
        return
    
    try:
        content = p.read_bytes()
        print(f"  Size: {len(content)} bytes")
        print(f"  First 16 bytes (hex): {content[:16].hex(' ')}")
        print(f"  First 16 bytes (raw): {content[:16]}")
        
        try:
            text = content.decode('utf-8')
            print("  [UTF-8 OK]")
        except UnicodeDecodeError as e:
            print(f"  [UTF-8 FAIL]: {e}")
            
        try:
            text = content.decode('utf-16')
            print("  [UTF-16 OK]")
        except UnicodeDecodeError as e:
            print(f"  [UTF-16 FAIL]: {e}")
            
    except Exception as e:
        print(f"  Error reading: {e}")

# Simulate logic in app.py
current_dir = Path('ui').resolve() # assuming we run from conversation-eval root
project_root = current_dir
# Wait, app.py is in ui/. Logic:
# current_dir = Path(__file__).parent  -> ui/
# project_root = current_dir.parent    -> conversation-eval/
# load_dotenv(project_root.parent / ".env") -> asign/.env

# Let's fix the anchor. If we run from conversation-eval:
# ui/app.py exists.
app_path = Path('ui/app.py').resolve()
print(f"App path: {app_path}")

current_dir = app_path.parent
project_root = current_dir.parent
target_env = project_root.parent / ".env"

check_file(target_env)
check_file(".env")
check_file("../.env")
