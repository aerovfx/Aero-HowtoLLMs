#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'

# mapping: old folder name -> new folder name
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


def run(cmd, check=True):
    print('RUN:', ' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=check)


def rename_dirs():
    for old, new in MAPPING.items():
        old_path = DOCS / old
        new_path = DOCS / new
        if old_path.exists() and old_path.is_dir():
            if new_path.exists():
                print(f"Target already exists, skipping: {new_path}")
                continue
            # use git mv if available
            try:
                run(['git', 'mv', str(old_path), str(new_path)])
                print(f"Renamed: {old} -> {new}")
            except Exception as e:
                print('git mv failed, falling back to os.rename:', e)
                os.rename(old_path, new_path)
        else:
            print(f"Source not found or not a dir, skipping: {old_path}")


def update_links():
    # Walk all markdown files under docs and replace occurrences of old/ with new/
    md_files = list(DOCS.rglob('*.md'))
    for md in md_files:
        text = md.read_text(encoding='utf-8')
        orig = text
        for old, new in MAPPING.items():
            # replace occurrences of "old/" and "old)" and "old]"
            text = text.replace(f"{old}/", f"{new}/")
            text = text.replace(f"({old}/", f"({new}/")
            text = text.replace(f"[{old}/", f"[{new}/")
            # also replace links to folder names without trailing slash
            text = text.replace(f"{old})", f"{new})")
            text = text.replace(f"{old}]", f"{new}]")
        if text != orig:
            md.write_text(text, encoding='utf-8')
            print(f"Updated links in: {md}")


if __name__ == '__main__':
    rename_dirs()
    update_links()
    print('Done. Please review changes and commit.')
