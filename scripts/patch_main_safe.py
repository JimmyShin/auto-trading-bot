from pathlib import Path
import re
p = Path('main.py')
src = p.read_text(encoding='utf-8', errors='surrogatepass')
# Fix trend_icon line
src = re.sub(r"trend_icon\s*=.*", "trend_icon = 'UP' if sig['fast_ma'] > sig['slow_ma'] else 'DOWN'", src)
# Fix regime_icon line
src = re.sub(r"regime_icon\s*=.*", "regime_icon = {'UP':'UP','DOWN':'DOWN','RANGE':'RANGE','UNKNOWN':'?'}.get(regime, '?')", src)
# Ensure signal_handler calls emergency_cleanup
src = src.replace('policy_emergency_cleanup()', 'emergency_cleanup()')
# Ensure atexit uses emergency_cleanup alias (we will alias later in code)
src = src.replace('atexit.register(emergency_cleanup)', 'atexit.register(emergency_cleanup)')
# Write back
p.write_text(src, encoding='utf-8')
print('PATCHED main.py')
