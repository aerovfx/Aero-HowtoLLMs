#!/usr/bin/env python3
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'

MAPPING = {
    'LLM_Course': '01-LLM_Course',
    '3 - Words to tokens to numbers': '02-Words-to-tokens-to-numbers',
    '30 - Python intro Indexing and slicing': '03-Python-Indexing-and-slicing',
    '1_buildGPT': '04-buildGPT',
    '4 - Embeddings spaces': '05-Embeddings-spaces',
    '2_pretraining': '06-pretraining',
    '3_Fine-tune pretrained models': '07-Fine-tune-pretrained-models',
    '4_Instruction tuning': '08-Instruction-tuning',
    '11 - Quantitative evaluations': '09-Quantitative-evaluations',
    '21 - Identifying circuits and components': '10-Identifying-circuits',
    '17 - Investigating token embeddings': '11-Investigating-token-embeddings',
    '18 - Investigating neurons and dimensions': '12-Investigating-neurons-dimensions',
    '19 - Investigating layers': '13-Investigating-layers',
    '23 - How to modify activations': '14-Modify-activations',
    '24 - Editing hidden states': '15-Editing-hidden-states',
    '25 - Interfering with attention': '16-Interfering-with-attention',
    '26 - Modifying MLP': '17-Modifying-MLP',
    'rag': '18-RAG',
    '14 - AI safety': '19-AI-safety',
    '28 - Python intro Colab and notebooks': '20-Python-Colab-notebooks',
    '29 - Python intro Data types': '21-Python-Data-types',
    '31 - Python intro Functions': '22-Python-Functions',
    '32 - Python intro Flow control': '23-Python-Flow-control',
    '33 - Python intro Data visualization': '24-Python-Data-visualization',
    '34 - Python intro Strings and texts': '25-Python-Strings-texts',
    '35 - Python intro Pytorch': '26-Python-PyTorch',
    '37 - Math of deep learning': '27-Math-deep-learning',
    '38 - How models learn gradient descent': '28-Gradient-descent',
    '39 - Essence of deep learning modeling': '29-Essence-deep-learning',
}

# Extensions to scan
EXTS = {'.md', '.mdx', '.html', '.htm', '.tsx', '.ts', '.jsx', '.js', '.json', '.txt', '.yml', '.yaml'}

def percent_encode(s: str) -> str:
    return s.replace(' ', '%20')


def build_patterns():
    patterns = []
    for old, new in MAPPING.items():
        # exact folder form
        patterns.append((re.escape(old), new))
        # percent-encoded form
        patterns.append((re.escape(percent_encode(old)), new))
        # old with slashes (e.g., docs/old/)
        patterns.append((re.escape('docs/' + old), 'docs/' + new))
        patterns.append((re.escape('docs/' + percent_encode(old)), 'docs/' + new))
    return patterns


def scan_and_replace():
    patterns = build_patterns()
    changed_files = []
    for path in ROOT.rglob('*'):
        if path.is_file() and path.suffix.lower() in EXTS:
            try:
                text = path.read_text(encoding='utf-8')
            except Exception:
                continue
            orig = text
            for pat, repl in patterns:
                text = re.sub(pat, repl, text)
            if text != orig:
                path.write_text(text, encoding='utf-8')
                changed_files.append(path)
                print(f'Updated links in: {path}')
    return changed_files

if __name__ == '__main__':
    changed = scan_and_replace()
    print(f'Done. Files changed: {len(changed)}')
