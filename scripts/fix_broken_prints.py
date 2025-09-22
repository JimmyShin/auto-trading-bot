from pathlib import Path
import re

def fix_file(path: Path):
    src = path.read_text(encoding='utf-8', errors='surrogatepass').splitlines(True)
    out = []
    in_triple = False
    triple_delim = None
    for line in src:
        # Strip control characters in 0x80..0x9F which can break strings
        cleaned = ''.join(ch for ch in line if not (0x80 <= ord(ch) <= 0x9F))
        line = cleaned
        # Handle triple-quoted blocks: skip everything inside
        if not in_triple and ('"""' in line or "'''" in line):
            # If both open and close appear on same line, drop the line
            if line.count('"""') == 2 or line.count("'''") == 2:
                continue
            in_triple = True
            triple_delim = '"""' if '"""' in line else "'''"
            continue
        if in_triple:
            if triple_delim in line:
                in_triple = False
                triple_delim = None
            continue
        if 'print(f"' in line or "print(f'" in line:
            # Heuristic: if quotes look unbalanced or line lacks closing )
            dq = line.count('"')
            sq = line.count("'")
            paren_open = line.count('(')
            paren_close = line.count(')')
            if (paren_close < paren_open) or (('print(f"' in line and dq % 2 == 1) or ("print(f'" in line and sq % 2 == 1)):
                indent = line[:len(line) - len(line.lstrip())]
                out.append(f"{indent}print('[log]')\n")
                continue
        out.append(line)
    path.write_text(''.join(out), encoding='utf-8')

if __name__ == '__main__':
    p = Path('main.py')
    fix_file(p)
    print('FIXED:', p)
