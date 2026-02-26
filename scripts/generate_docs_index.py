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

def generate_breadcrumb(rel_path, is_file=False):
    if rel_path == ".":
        return "**Home**"
    
    parts = rel_path.split(os.sep)
    # Root Home link pointing to the main index.md
    depth = len(parts)
    # File is one level deeper than its base directory
    steps_to_root = depth + (1 if is_file else 0)
    home_link = "../" * steps_to_root + "index.md"
    breadcrumb = [f"[🏠 Home]({home_link})"]

    for i, part in enumerate(parts):
        # Calculate how many levels to go back to reach this part's index.md
        # Part i depth relative to root: i + 1
        steps_back = depth - (i + 1)
        if is_file:
            steps_back += 1
        
        link = "../" * steps_back + "index.md"
        
        # Clean up the part name for display (e.g. remove numbering)
        display_name = part.replace("_", " ").replace("-", " ")
        match = re.match(r'^\d+-(.*)', display_name)
        if match:
            display_name = match.group(1).strip()

        if i == len(parts) - 1 and not is_file:
            breadcrumb.append(f"**{display_name}**")
        else:
            breadcrumb.append(f"[{display_name}]({link})")
            
    return " > ".join(breadcrumb)

def generate_sidebar(depth):
    prefix = "../" * depth
    sidebar = ["### 🧭 Điều hướng nhanh\n"]
    sidebar.append(f"- [🏠 Cổng tài liệu]({prefix}index.md)")
    sidebar.append(f"- [📚 Module 01: LLM Course]({prefix}01-LLM_Course/index.md)")
    sidebar.append(f"- [🔢 Module 02: Tokenization]({prefix}02-Words-to-tokens-to-numbers/index.md)")
    sidebar.append(f"- [🏗️ Module 04: Build GPT]({prefix}04-buildGPT/index.md)")
    sidebar.append(f"- [🎯 Module 07: Fine-tuning]({prefix}07-Fine-tune-pretrained-models/index.md)")
    sidebar.append(f"- [🔍 Module 19: AI Safety]({prefix}19-AI-safety/index.md)")
    sidebar.append(f"- [🐍 Module 20: Python for AI]({prefix}20-Python-Colab-notebooks/index.md)")
    return "\n".join(sidebar)

def process_all_markdowns(base_dir):
    docs_dir = os.path.join(base_dir, 'docs')
    if not os.path.exists(docs_dir):
        return

    for root, dirs, files in os.walk(docs_dir):
        rel_path = os.path.relpath(root, docs_dir)
        depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
        
        subfolders = sorted([d for d in dirs if not d.startswith('.') and not d.startswith('_')])
        md_files = [f for f in files if f.endswith('.md')] # Process ALL md files

        # 1. Generate/Update index.md for this folder
        generate_folder_index(root, rel_path, depth, subfolders, md_files)

        # 2. Process every markdown file in this folder
        for md in md_files:
            if md.lower() == 'index.md': continue
            # We treat README.md specially in generate_folder_index, but let's ensure consistency
            file_path = os.path.join(root, md)
            update_article_navigation(file_path, rel_path, depth, md.lower() == 'readme.md')

def update_article_navigation(file_path, rel_dir_path, dir_depth, is_readme):
    # For a file, the effective depth for links is dir_depth + 1
    depth = dir_depth if is_readme else dir_depth + 1
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if already processed
        if "<!-- Aero-Navigation-Start -->" in content:
            # Strip existing navigation to refresh it
            content = re.sub(r"<!-- Aero-Navigation-Start -->.*?<!-- Aero-Navigation-End -->", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"<!-- Aero-Footer-Start -->.*?<!-- Aero-Footer-End -->", "", content, flags=re.DOTALL).strip()

        # Generate Navigation Header
        header = ["\n<!-- Aero-Navigation-Start -->\n"]
        header.append(f"{generate_breadcrumb(rel_dir_path, not is_readme)}\n")
        header.append("\n---\n")
        header.append(generate_sidebar(depth))
        header.append("\n---\n<!-- Aero-Navigation-End -->\n")
        
        # Generate Footer
        footer = ["\n<!-- Aero-Footer-Start -->\n---\n"]
        footer.append(f"## 🤝 Liên hệ & Đóng góp\n")
        footer.append(f"Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.\n\n")
        footer.append(f"> *\"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!\"* 🚀\n")
        footer.append(f"\n*Cập nhật tự động bởi Aero-Indexer - 2026*\n<!-- Aero-Footer-End -->\n")

        # Combine: Header + Content + Footer
        # We don't want to double titles if it's already there
        new_content = "".join(header) + content + "".join(footer)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def generate_folder_index(root, rel_path, depth, subfolders, md_files):
    # Filter files for listing (exclude the index itself)
    list_files = [f for f in md_files if f.lower() != 'index.md']
    list_files.sort()

    folder_name = os.path.basename(root)
    index_content = ["<!-- Aero-Navigation-Start -->\n"]
    
    if folder_name == 'docs':
        index_content.append(f"# 🚀 Master Index: Aero-HowtoLLMs\n")
        index_content.append(f"> **Danh mục tổng hợp toàn bộ lộ trình và tài liệu nghiên cứu LLM.**\n")
    else:
        index_content.append(f"# 📂 Module: {folder_name}\n")
        index_content.append(f"> **Tài liệu chuyên sâu và bài tập thuộc phần {folder_name}.**\n")

    index_content.append(f"[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]() ")
    index_content.append(f"[![Content: 100% Vietnamese](https://img.shields.io/badge/Content-Vietnamese-red.svg)]()\n")
    index_content.append(f"\n{generate_breadcrumb(rel_path, False)}\n")
    index_content.append("\n---\n")
    index_content.append(generate_sidebar(depth))
    index_content.append("\n---\n<!-- Aero-Navigation-End -->\n")

    if subfolders:
        index_content.append(f"## 📁 Thư mục con\n")
        index_content.append("| Thư mục | Liên kết |\n")
        index_content.append("| :--- | :--- |\n")
        for sub in subfolders:
            display_name = sub.replace("_", " ").replace("-", " ")
            match = re.match(r'^\d+-(.*)', display_name)
            if match:
                display_name = match.group(1).strip()
            index_content.append(f"| **{display_name}** | [Mở thư mục →]({sub}/index.md) |\n")
        index_content.append("\n")

    if list_files:
        index_content.append(f"## 📄 Tài liệu chi tiết\n")
        index_content.append("| Bài học | Liên kết |\n")
        index_content.append("| :--- | :--- |\n")
        for md in list_files:
            title = get_title_from_md(os.path.join(root, md))
            # Make the title itself a link as well for better UX
            index_content.append(f"| [{title}]({md}) | [Xem bài viết →]({md}) |\n")
        index_content.append("\n")

    # Footer
    index_content.append(f"<!-- Aero-Footer-Start -->\n---\n")
    index_content.append(f"## 🤝 Liên hệ & Đóng góp\n")
    index_content.append(f"Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.\n\n")
    index_content.append(f"> *\"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!\"* 🚀\n")
    index_content.append(f"\n*Cập nhật tự động bởi Aero-Indexer - 2026*\n<!-- Aero-Footer-End -->")

    full_content = "".join(index_content)
    index_path = os.path.join(root, 'index.md')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    print(f"Generated index for {root}")

if __name__ == "__main__":
    base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    process_all_markdowns(base_directory)
