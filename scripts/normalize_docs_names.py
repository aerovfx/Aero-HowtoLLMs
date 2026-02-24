#!/usr/bin/env python3
import re
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'

def normalized(name: str) -> str:
    # replace ' - ' with '-' and collapse multiple spaces
    s = re.sub(r"\s*-\s*", "-", name)
    s = re.sub(r"\s+", "-", s)
    s = s.replace(' ', '-')
    s = s.strip('-')
    return s


def run(cmd):
    print('RUN:', ' '.join(cmd))
    return subprocess.run(cmd, cwd=ROOT)


def main():
    for p in DOCS.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        newname = normalized(name)
        if newname == name:
            continue
        target = DOCS / newname
        if target.exists():
            print(f"Target exists, skipping rename: {name} -> {newname}")
            continue
        try:
            # try git mv first
            res = run(['git', 'mv', str(p), str(target)])
            if res.returncode == 0:
                print(f"Renamed with git: {name} -> {newname}")
            else:
                raise Exception('git mv failed')
        except Exception as e:
            print('git mv failed, falling back to os.rename:', e)
            os.rename(p, target)
            print(f"Renamed filesystem: {name} -> {newname}")

    # update links in markdown files
    md_files = list(DOCS.rglob('*.md'))
    for md in md_files:
        text = md.read_text(encoding='utf-8')
        orig = text
        for p in DOCS.iterdir():
            # original names might still be present in text; replace variants
            # replace 'Old Name/' with 'NewName/' for patterns with spaces and hyphens
            pass
        if text != orig:
            md.write_text(text, encoding='utf-8')
            print(f"Updated links in: {md}")

if __name__ == '__main__':
    main()
