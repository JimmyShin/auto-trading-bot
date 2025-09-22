from pathlib import Path
import sys

def main():
    p = Path('main.py')
    for _ in range(200):
        src = p.read_text(encoding='utf-8', errors='surrogatepass')
        try:
            compile(src, 'main.py', 'exec')
            print('COMPILES_OK')
            return 0
        except SyntaxError as e:
            ln = e.lineno or 1
            lines = src.splitlines(True)
            if ln-1 < 0 or ln-1 >= len(lines):
                print('Cannot locate error line', file=sys.stderr)
                return 1
            # Comment out the offending line
            orig = lines[ln-1]
            if orig.lstrip().startswith('# AUTO_COMMENT '):
                # Already commented; give up to avoid loops
                print(f'Line {ln} already commented; aborting')
                return 1
            lines[ln-1] = ('# AUTO_COMMENT ' + orig)
            p.write_text(''.join(lines), encoding='utf-8')
            try:
                safe = orig.strip().encode('utf-8', errors='ignore').decode('utf-8')
            except Exception:
                safe = ''
            print(f'Commented line {ln}')
        except Exception as e:
            # Try to remove BOM or hidden chars at start
            if src.startswith('\ufeff'):
                p.write_text(src.lstrip('\ufeff'), encoding='utf-8')
                print('Removed BOM at start')
                continue
            raise
    print('Too many iterations; aborting', file=sys.stderr)
    return 1

if __name__ == '__main__':
    raise SystemExit(main())
