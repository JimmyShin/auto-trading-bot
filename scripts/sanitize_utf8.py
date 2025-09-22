import sys
from pathlib import Path

def sanitize(path: Path):
    b = path.read_bytes()
    s = b.decode('utf-8', errors='ignore')
    # Remove problematic control chars in U+0080..U+009F range
    s = ''.join(ch for ch in s if not (0x80 <= ord(ch) <= 0x9F))
    path.write_text(s, encoding='utf-8')

if __name__ == '__main__':
    p = Path(sys.argv[1] if len(sys.argv) > 1 else 'main.py')
    sanitize(p)
    print('SANITIZED:', p)
