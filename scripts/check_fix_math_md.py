#!/usr/bin/env python3
"""Scan Markdown files for LaTeX math issues and apply conservative fixes.

Checks performed:
- Ignores fenced code blocks and inline code spans
- Finds display math (`$$ ... $$`) and inline math (`$ ... $`)
- Reports unbalanced braces `{` vs `}` and attempts conservative fixes
- Reports unmatched display delimiters and can append a closing `$$`

This script only modifies files under `docs/` and the top-level `README.md`.
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [ROOT / 'docs', ROOT / 'README.md']


def iter_md_files():
    # yield README.md then all .md under docs/
    yield ROOT / 'README.md'
    docs_dir = ROOT / 'docs'
    if docs_dir.exists():
        for p in docs_dir.rglob('*.md'):
            yield p


def mask_code_blocks(text):
    # Replace fenced code blocks and inline code spans with placeholders to avoid math detection inside them
    # Fenced code blocks ``` ... ``` (including language) and ~~~
    def repl_fenced(m):
        return '\n' + ' ' * (len(m.group(0)) - 1) + '\n'

    text = re.sub(r'(^```[\s\S]*?^```)', repl_fenced, text, flags=re.MULTILINE)
    text = re.sub(r'(^~~~[\s\S]*?^~~~)', repl_fenced, text, flags=re.MULTILINE)
    # Inline code `...`
    text = re.sub(r'`[^`]*?`', lambda m: ' ' * len(m.group(0)), text)
    return text


def find_display_math(text):
    return list(re.finditer(r'(?s)\$\$(.+?)\$\$', text))


def find_inline_math(text):
    # Match single-dollar inline math while avoiding $$
    return list(re.finditer(r'(?s)(?<!\\)\$(?!\$)(.+?)(?<!\\)\$', text))


def check_brace_balance(expr):
    # naive check
    return expr.count('{') - expr.count('}')


def has_begin_without_end(expr):
    begins = re.findall(r'\\begin\{([^}]+)\}', expr)
    ends = re.findall(r'\\end\{([^}]+)\}', expr)
    return len(begins) != len(ends)


def process_file(path: Path):
    text = path.read_text(encoding='utf-8')
    masked = mask_code_blocks(text)
    changes = []
    new_text = text

    # Display math
    for m in find_display_math(masked):
        expr = m.group(1)
        bal = check_brace_balance(expr)
        if bal > 0 and bal <= 3 and not has_begin_without_end(expr):
            # append missing closing braces before the closing $$ in the real text
            real_start = m.start()
            closing_idx = new_text.find('$$', real_start)
            if closing_idx != -1:
                insert_pos = closing_idx
                new_text = new_text[:insert_pos] + ('}' * bal) + new_text[insert_pos:]
                changes.append((path, 'append_closing_braces_in_display', bal))

    # Re-mask after possible edits
    masked = mask_code_blocks(new_text)

    # Inline math
    for m in find_inline_math(masked):
        expr = m.group(1)
        bal = check_brace_balance(expr)
        if bal > 0 and bal <= 3 and not has_begin_without_end(expr):
            real_start = m.start()
            trailing_idx = new_text.find('$', real_start + 1)
            if trailing_idx != -1:
                new_text = new_text[:trailing_idx] + ('}' * bal) + new_text[trailing_idx:]
                changes.append((path, 'append_closing_braces_inline', bal))

    # Check unmatched display delimiters: odd number of $$ occurrences in masked text
    dd = re.findall(r'\$\$', masked)
    if len(dd) % 2 == 1:
        # append a closing $$ at EOF
        new_text = new_text.rstrip() + '\n\n$$\n'
        changes.append((path, 'append_missing_closing_display', 1))

    if changes:
        path.write_text(new_text, encoding='utf-8')
    return changes


def main():
    total_changes = []
    for p in iter_md_files():
        try:
            changes = process_file(p)
            if changes:
                for c in changes:
                    print(f'Updated {c[0]}: {c[1]} ({c[2]})')
                total_changes.extend(changes)
        except Exception as e:
            print(f'Error processing {p}: {e}')

    if not total_changes:
        print('No automatic fixes applied. Files appear syntactically balanced (basic checks).')
    else:
        print(f'Applied {len(total_changes)} conservative fixes. Please review changes and run a link/math renderer for full verification.')


if __name__ == '__main__':
    main()
