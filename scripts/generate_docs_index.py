import os
import re

# Define categories based on the docs structure
CATEGORIES = [
    {"name": "Fundamentals", "path": "docs/", "folders": ["01-LLM_Course", "02-Words-to-tokens-to-numbers", "03-Python-Indexing-and-slicing", "05-Embeddings-spaces", "27-Math-deep-learning"]},
    {"name": "Build & Train", "path": "docs/", "folders": ["04-buildGPT", "06-pretraining", "28-Gradient-descent", "29-Essence-deep-learning"]},
    {"name": "Fine-tuning & RAG", "path": "docs/", "folders": ["07-Fine-tune-pretrained-models", "08-Instruction-tuning", "18-RAG"]},
    {"name": "Interpretability & Safety", "path": "docs/", "folders": ["09-Quantitative-evaluations", "10-Identifying-circuits", "19-AI-safety"]},
    {"name": "Python & Tools", "path": "docs/", "folders": ["20-Python-Colab-notebooks", "21-Python-Data-types"]}
]

def get_title_from_md(filepath):
    """Extracts the first H1 title from a markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r'^#\s+(.*)', line)
                if match:
                    return match.group(1).strip()
    except Exception:
        pass
    return os.path.basename(filepath)

def generate_breadcrumb(rel_path):
    if rel_path == ".":
        return "**Home**"
    
    parts = rel_path.split(os.sep)
    breadcrumb = ["[Home](../README.md)"]
    
    current_path = ""
    for i, part in enumerate(parts):
        # Calculate number of levels back to reach the current part from the current index file
        # The index file is at depth `len(parts)`
        # The part is at depth `i+1`
        # Levels back: len(parts) - (i + 1)
        depth_back = len(parts) - (i + 1)
        link = "../" * depth_back + "index.md" if depth_back > 0 else "index.md"
        
        if i == len(parts) - 1:
            breadcrumb.append(f"**{part}**")
        else:
            breadcrumb.append(f"[{part}]({link})")
            
    return " > ".join(breadcrumb)

def generate_sidebar(depth):
    prefix = "../" * depth
    sidebar = ["### 🧭 Quick Navigation\n"]
    sidebar.append(f"- [🏠 Cổng tài liệu]({prefix}README.md)")
    sidebar.append(f"- [📚 Module 01: LLM Course]({prefix}01-LLM_Course/index.md)")
    sidebar.append(f"- [🔢 Module 02: Tokenization]({prefix}02-Words-to-tokens-to-numbers/index.md)")
    sidebar.append(f"- [🏗️ Module 04: Build GPT]({prefix}04-buildGPT/index.md)")
    sidebar.append(f"- [🎯 Module 07: Fine-tuning]({prefix}07-Fine-tune-pretrained-models/index.md)")
    sidebar.append(f"- [🔍 Module 19: AI Safety]({prefix}19-AI-safety/index.md)")
    return "\n".join(sidebar)

def generate_indexes(base_dir):
    docs_dir = os.path.join(base_dir, 'docs')
    if not os.path.exists(docs_dir):
        print(f"Directory {docs_dir} not found.")
        return

    for root, dirs, files in os.walk(docs_dir):
        rel_path = os.path.relpath(root, docs_dir)
        depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
        
        # Filter markdown files
        md_files = [f for f in files if f.endswith('.md') and f.lower() not in ['index.md', 'readme.md']]
        md_files.sort()
        
        # Subdirectories
        subfolders = sorted([d for d in dirs if not d.startswith('.') and not d.startswith('_')])
        
        if not md_files and not subfolders and rel_path != ".":
            continue

        index_content = []
        folder_name = os.path.basename(root)
        
        # Header
        if folder_name == 'docs':
            index_content.append(f"# 📂 Master Index — Aero-HowtoLLMs\n")
        else:
            index_content.append(f"# 📂 Index — {folder_name}\n")

        # Breadcrumbs
        index_content.append(generate_breadcrumb(rel_path))
        index_content.append("\n---\n")

        # Left Column (Sidebar-like)
        index_content.append(generate_sidebar(depth))
        index_content.append("\n---\n")

        # Main Content
        if subfolders:
            index_content.append(f"## 📁 Thư mục con\n")
            for sub in subfolders:
                index_content.append(f"- [**{sub}**]({sub}/index.md)")
            index_content.append("")

        if md_files:
            index_content.append(f"## 📄 Tài liệu trong mục này\n")
            for md in md_files:
                title = get_title_from_md(os.path.join(root, md))
                index_content.append(f"- [{title}]({md})")
        
        index_content.append(f"\n---\n*Tự động cập nhật bởi Aero-Indexer*")

        index_path = os.path.join(root, 'index.md')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(index_content))
            
        print(f"Generated index for {root} (depth {depth})")

if __name__ == "__main__":
    base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    generate_indexes(base_directory)
