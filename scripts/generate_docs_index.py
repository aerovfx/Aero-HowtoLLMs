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

def generate_breadcrumb(rel_path):
    if rel_path == ".":
        return "**Home**"
    
    parts = rel_path.split(os.sep)
    breadcrumb = ["[Home](../README.md)"]
    
    for i, part in enumerate(parts):
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
        # Include README.md in the listing if we are not overwriting it
        md_files = [f for f in files if f.endswith('.md') and f.lower() not in ['index.md']]
        md_files.sort()
        
        # Subdirectories
        subfolders = sorted([d for d in dirs if not d.startswith('.') and not d.startswith('_')])
        
        if not md_files and not subfolders and rel_path != ".":
            continue

        index_content = []
        folder_name = os.path.basename(root)
        
        # --- HEADER ---
        if folder_name == 'docs':
            index_content.append(f"# 🚀 Master Index: Aero-HowtoLLMs\n")
            index_content.append(f"> **Danh mục tổng hợp toàn bộ lộ trình và tài liệu nghiên cứu LLM.**\n")
        else:
            index_content.append(f"# 📂 Module: {folder_name}\n")
            index_content.append(f"> **Tài liệu chuyên sâu và bài tập thuộc phần {folder_name}.**\n")

        # --- BADGES ---
        index_content.append(f"[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]() ")
        index_content.append(f"[![Content: 100% Vietnamese](https://img.shields.io/badge/Content-Vietnamese-red.svg)]()\n")

        # --- NAVIGATION ---
        index_content.append(f"\n{generate_breadcrumb(rel_path)}")
        index_content.append("\n---\n")

        # --- SIDEBAR & CONTENT ---
        index_content.append(generate_sidebar(depth))
        index_content.append("\n---\n")

        if subfolders:
            index_content.append(f"## 📁 Thư mục con\n")
            for sub in subfolders:
                index_content.append(f"[{sub}]({sub}/index.md)")
            index_content.append("\n")

        if md_files:
            index_content.append(f"## 📄 Tài liệu chi tiết\n")
            index_content.append("| Bài học | Liên kết |")
            index_content.append("| :--- | :--- |")
            for md in md_files:
                title = get_title_from_md(os.path.join(root, md))
                index_content.append(f"| {title} | [Xem bài viết →]({md}) |")
            index_content.append("\n")
        
        # --- FOOTER ---
        index_content.append(f"---\n")
        index_content.append(f"## 🤝 Liên hệ & Đóng góp\n")
        index_content.append(f"Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.\n\n")
        index_content.append(f"> *\"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!\"* 🚀\n")
        index_content.append(f"\n*Cập nhật tự động bởi Aero-Indexer - 2026*")

        full_content = '\n'.join(index_content)
        
        # Write to index.md
        index_path = os.path.join(root, 'index.md')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        # Smart update for README.md: Inject styles but keep unique content
        readme_path = os.path.join(root, 'README.md')
        if os.path.exists(readme_path) and rel_path != ".":
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    orig_lines = f.readlines()
                
                # Check if it was already updated (look for the indexer footer)
                if any("Aero-Indexer" in line for line in orig_lines):
                    # It's already our style or was overwritten
                    with open(readme_path, 'w', encoding='utf-8') as f:
                        f.write(full_content)
                else:
                    # It's an original README. Prepend header and append footer
                    header = []
                    header.append(f"# 📂 Module: {folder_name}\n")
                    header.append(f"[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]() ")
                    header.append(f"[![Content: 100% Vietnamese](https://img.shields.io/badge/Content-Vietnamese-red.svg)]()\n")
                    header.append(f"\n{generate_breadcrumb(rel_path)}\n")
                    header.append("\n---\n")
                    header.append(generate_sidebar(depth))
                    header.append("\n---\n")
                    
                    footer = []
                    footer.append(f"\n---\n")
                    footer.append(f"## 🤝 Liên hệ & Đóng góp\n")
                    footer.append(f"Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.\n\n")
                    footer.append(f"> *\"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!\"* 🚀\n")
                    footer.append(f"\n*Cập nhật tự động bởi Aero-Indexer - 2026*")
                    
                    # Construct new content: Header + Original (skipping original H1 if redundant) + Footer
                    new_readme = header
                    h1_found = False
                    for line in orig_lines:
                        if line.startswith('# ') and not h1_found:
                            h1_found = True
                            continue # Skip the first H1 as we added our own
                        new_readme.append(line)
                    new_readme.extend(footer)
                    
                    with open(readme_path, 'w', encoding='utf-8') as f:
                        f.writelines(new_readme)
            except Exception as e:
                print(f"Error updating {readme_path}: {e}")
            
        print(f"Index created/Readme styled for {root}")

if __name__ == "__main__":
    base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    generate_indexes(base_directory)
