#!/usr/bin/env python3
"""Check actual madvise syscalls from synthetic HostMemory tests, not game traces."""
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
for mode in ('unset', 'pageout', 'cold', 'off', 'invalid'):
    trace = (root / (mode + '.strace')).read_text()
    calls = re.findall(r'madvise\([^,]+,\s*196608,\s*(MADV_[A-Z0-9_]+)\)\s*=\s*([^\n]+)', trace)
    if mode == 'off':
        assert not calls, (mode, calls)
        assert not re.search(r'MADV_(?:COLD|PAGEOUT)', trace), mode
    else:
        advice = 'MADV_COLD' if mode == 'cold' else 'MADV_PAGEOUT'
        assert len(calls) == 4, (mode, calls)
        assert all(flag == advice and result.strip() == '0' for flag, result in calls), (mode, calls)
    print(mode + ': actual syscalls and return values PASS')
