import pathlib
import os

def fix_file(path_str):
    p = pathlib.Path(path_str).resolve()
    print(f"Processing: {p}")
    if not p.exists():
        print("  Not found.")
        return

    try:
        raw = p.read_bytes()
        print(f"  Original size: {len(raw)}")
        print(f"  Original header: {raw[:4]}")
        
        content = None
        # Try UTF-16 LE BOM
        if raw.startswith(b'\xff\xfe'):
            print("  Detected UTF-16 LE BOM.")
            content = raw.decode('utf-16')
        # Try UTF-16 BE BOM
        elif raw.startswith(b'\xfe\xff'):
            print("  Detected UTF-16 BE BOM.")
            content = raw.decode('utf-16')
        # Try UTF-8 BOM
        elif raw.startswith(b'\xef\xbb\xbf'):
            print("  Detected UTF-8 BOM.")
            content = raw.decode('utf-8-sig')
        else:
            # Try plain UTF-8
            try:
                content = raw.decode('utf-8')
                print("  Decoded as UTF-8.")
            except UnicodeDecodeError:
                # Try latin-1 as fallback
                print("  UTF-8 failed, trying latin-1.")
                content = raw.decode('latin-1')
        
        if content is not None:
            p.write_text(content, encoding='utf-8')
            print("  Saved as UTF-8 (no BOM).")
            
    except Exception as e:
        print(f"  Error: {e}")

# Fix both potential locations
fix_file(".env")
fix_file("../.env")
