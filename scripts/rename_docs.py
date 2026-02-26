#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'

# mapping: old folder name -> new folder name
MAPPING = {
    'LLM_Course': '01_llm_course',
    '3 - Words to tokens to numbers': '02_words_to_tokens_to_numbers',
    '30 - Python intro Indexing and slicing': '03_python_indexing_and_slicing',
    '1_buildGPT': '04_buildgpt',
    '4 - Embeddings spaces': '05_embeddings_spaces',
    '2_pretraining': '06_pretraining',
    '3_Fine-tune pretrained models': '07_fine_tune_pretrained_models',
    '4_Instruction tuning': '08_instruction_tuning',
    '11 - Quantitative evaluations': '09_quantitative_evaluations',
    '21 - Identifying circuits and components': '10_identifying_circuits',
    '17 - Investigating token embeddings': '11_investigating_token_embeddings',
    '18 - Investigating neurons and dimensions': '12_investigating_neurons_dimensions',
    '19 - Investigating layers': '13_investigating_layers',
    '23 - How to modify activations': '14_modify_activations',
    '24 - Editing hidden states': '15_editing_hidden_states',
    '25 - Interfering with attention': '16_interfering_with_attention',
    '26 - Modifying MLP': '17_modifying_mlp',
    'rag': '18_rag',
    '14 - AI safety': '19_ai_safety',
    '28 - Python intro Colab and notebooks': '20_python_colab_notebooks',
    '29 - Python intro Data types': '21_python_data_types',
    '31 - Python intro Functions': '22_python_functions',
    '32 - Python intro Flow control': '23_python_flow_control',
    '33 - Python intro Data visualization': '24_python_data_visualization',
    '34 - Python intro Strings and texts': '25_python_strings_texts',
    '35 - Python intro Pytorch': '26_python_pytorch',
    '37 - Math of deep learning': '27_math_deep_learning',
    '38 - How models learn gradient descent': '28_gradient_descent',
    '39 - Essence of deep learning modeling': '29_essence_deep_learning',
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
