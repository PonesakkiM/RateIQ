import re
with open('frontend/app.py', encoding='utf-8') as f:
    lines = f.readlines()

# 1. Literal 8-digit hex  e.g. #EF444422
eight_hex  = re.compile(r'#[0-9A-Fa-f]{8}\b')
# 2. f-string variable + 2 hex digits  e.g. {DANGER}22  {SUCCESS}18  {PRIMARY}40
fvar_alpha = re.compile(r'\{(DANGER|WARNING|SUCCESS|PRIMARY|SECONDARY|MUTED|BORDER)\}([0-9A-Fa-f]{2})\b')

found = []
for i, l in enumerate(lines, 1):
    m1 = eight_hex.findall(l)
    m2 = fvar_alpha.findall(l)
    if m1 or m2:
        found.append((i, l.rstrip(), m1, m2))

if found:
    print('INVALID COLOR PATTERNS FOUND (%d lines):' % len(found))
    for ln, text, h8, fv in found:
        print('L%4d: %s' % (ln, text[:110]))
        if h8: print('       8-digit hex:', h8)
        if fv: print('       var+alpha:   ', fv)
else:
    print('No invalid 8-digit hex colors found.')
    print('Gauge step colors at L803-806 are already valid rgba() format.')
    print('No fix required for gauge color issue.')

# Also confirm gauge lines are correct
print()
print('Gauge step lines:')
for i, l in enumerate(lines[799:810], 800):
    if 'range' in l or 'color' in l or 'step' in l.lower():
        print('L%4d: %s' % (i, l.rstrip()))
