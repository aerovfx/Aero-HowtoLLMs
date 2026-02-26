import os
import re

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

def generate_indexes(base_dir):
    docs_dir = os.path.join(base_dir, 'docs')
    if not os.path.exists(docs_dir):
        print(f"Directory {docs_dir} not found.")
        return

    for root, dirs, files in os.walk(docs_dir):
        # Skip the root docs dir itself if we want a manual README.md there
        # but we can still generate an index for it if preferred.
        
        # Get relative path for breadcrumbs
        rel_path = os.path.relpath(root, docs_dir)
        
        # Filter for markdown files, excluding index.md and README.md
        md_files = [f for f in files if f.endswith('.md') and f.lower() not in ['index.md', 'readme.md']]
        # Sort files (assuming they have numbers like aero_LLM_01...)
        md_files.sort()
        
        # Subdirectories
        subfolders = sorted([d for d in dirs if not d.startswith('.')])
        
        if not md_files and not subfolders:
            continue

        index_content = []
        folder_name = os.path.basename(root)
        if folder_name == 'docs':
            index_content.append(f"# 📂 Master Index — Aero-HowtoLLMs\n")
        else:
            index_content.append(f"# 📂 Index — {folder_name}\n")

        # Navigation
        if folder_name != 'docs':
            index_content.append(f"[← Quay lại danh mục chính](../README.md)\n")
        
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
            
        print(f"Generated index for {root}")

if __name__ == "__main__":
    # Base directory is the parent of the scripts folder
    base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    generate_indexes(base_directory)
