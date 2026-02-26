import os
import re
import urllib.parse

def get_title_from_md(filepath):
    """Extracts the first H1 title from a markdown file and cleans it."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r'^#\s+(.*)', line)
                if match:
                    title = match.group(1).strip()
                    # Remove markdown links: [text](url) -> text
                    title = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', title)
                    # Remove emphasis
                    title = title.replace('**', '').replace('__', '')
                    return title
    except Exception:
        pass
    # Fallback to filename
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]
    # Remove numbering if present
    name = re.sub(r'^\d+-', '', name)
    return name.replace('_', ' ').replace('-', ' ').strip()

def safe_link(path):
    """Encodes spaces and special characters in links."""
    return urllib.parse.quote(path)

def generate_breadcrumb(rel_path, is_file=False):
    if rel_path == ".":
        return "**Home**"
    
    parts = [p for p in rel_path.split(os.sep) if p]
    depth = len(parts)
    steps_to_root = depth
    home_link = ("../" * steps_to_root) + "index.md"
    breadcrumb = [f"[🏠 Home]({safe_link(home_link)})"]

    for i, part in enumerate(parts):
        steps_back = depth - (i + 1)
        link = ("../" * steps_back) + "index.md"
        display_name = part.replace("_", " ").replace("-", " ")
        match = re.match(r'^\d+-(.*)', display_name)
        if match:
            display_name = match.group(1).strip()

        if i == len(parts) - 1 and not is_file:
            breadcrumb.append(f"**{display_name}**")
        else:
            breadcrumb.append(f"[{display_name}]({safe_link(link)})")
            
    return " > ".join(breadcrumb)

def generate_sidebar(depth):
    prefix = "../" * depth
    sidebar = ["### 🧭 Điều hướng nhanh\n"]
    sidebar.append(f"- [🏠 Cổng tài liệu]({safe_link(prefix + 'index.md')})")
    sidebar.append(f"- [📚 Module 01: LLM Course]({safe_link(prefix + '01_llm_course/index.md')})")
    sidebar.append(f"- [🔢 Module 02: Tokenization]({safe_link(prefix + '02_words_to_tokens_to_numbers/index.md')})")
    sidebar.append(f"- [🏗️ Module 04: Build GPT]({safe_link(prefix + '04_buildgpt/index.md')})")
    sidebar.append(f"- [🎯 Module 07: Fine-tuning]({safe_link(prefix + '07_fine_tune_pretrained_models/index.md')})")
    sidebar.append(f"- [🔍 Module 19: AI Safety]({safe_link(prefix + '19_ai_safety/index.md')})")
    sidebar.append(f"- [🐍 Module 20: Python for AI]({safe_link(prefix + '20_python_colab_notebooks/index.md')})")
    return "\n".join(sidebar)

def process_all_markdowns(base_dir):
    docs_dir = os.path.join(base_dir, 'docs')
    if not os.path.exists(docs_dir):
        return

    for root, dirs, files in os.walk(docs_dir):
        rel_path = os.path.relpath(root, docs_dir)
        depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
        
        subfolders = sorted([d for d in dirs if not d.startswith('.') and not d.startswith('_')])
        md_files = [f for f in files if f.endswith('.md')]
        list_files = sorted([f for f in md_files if f.lower() != 'index.md'])

        generate_folder_index(root, rel_path, depth, subfolders, md_files)

        for md in md_files:
            if md.lower() == 'index.md': continue
            file_path = os.path.join(root, md)
            update_article_navigation(file_path, rel_path, depth, md.lower() == 'readme.md', list_files=list_files, current_dir=root)

def update_article_navigation(file_path, rel_dir_path, dir_depth, is_readme, list_files=None, current_dir=None):
    depth = dir_depth
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if "<!-- Aero-Navigation-Start -->" in content:
            content = re.sub(r"<!-- Aero-Navigation-Start -->.*?<!-- Aero-Navigation-End -->", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"<!-- Aero-Footer-Start -->.*?<!-- Aero-Footer-End -->", "", content, flags=re.DOTALL).strip()

        header = ["\n<!-- Aero-Navigation-Start -->\n"]
        header.append(f"{generate_breadcrumb(rel_dir_path, not is_readme)}\n")
        header.append("\n---\n")
        header.append(generate_sidebar(depth))
        header.append("\n---\n<!-- Aero-Navigation-End -->\n")

        sibling_table = ""
        if list_files and len(list_files) > 1:
            sibling_table = ["\n## 📄 Tài liệu cùng chuyên mục\n"]
            sibling_table.append("| Bài học | Liên kết |\n")
            sibling_table.append("| :--- | :--- |\n")
            for sibling in list_files:
                is_current = sibling == os.path.basename(file_path)
                prefix = "📌 **" if is_current else ""
                suffix = "**" if is_current else ""
                
                title = get_title_from_md(os.path.join(current_dir, sibling))
                sibling_table.append(f"| {prefix}[{title}]({safe_link(sibling)}){suffix} | [Xem bài viết →]({safe_link(sibling)}) |\n")
            sibling_table.append("\n")
            sibling_table = "".join(sibling_table)

        footer = ["\n<!-- Aero-Footer-Start -->\n"]
        if sibling_table:
            footer.append(sibling_table)
        footer.append("---\n## 🤝 Liên hệ & Đóng góp\n")
        footer.append(f"Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.\n\n")
        footer.append(f"> *\"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!\"* 🚀\n")
        footer.append(f"\n*Cập nhật tự động bởi Aero-Indexer - 2026*\n<!-- Aero-Footer-End -->\n")

        new_content = "".join(header) + content + "".join(footer)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def generate_folder_index(root, rel_path, depth, subfolders, md_files):
    list_files = sorted([f for f in md_files if f.lower() != 'index.md'])
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
            index_content.append(f"| **{display_name}** | [Mở thư mục →]({safe_link(sub + '/index.md')}) |\n")
        index_content.append("\n")

    if list_files:
        index_content.append(f"## 📄 Tài liệu chi tiết\n")
        index_content.append("| Bài học | Liên kết |\n")
        index_content.append("| :--- | :--- |\n")
        for md in list_files:
            title = get_title_from_md(os.path.join(root, md))
            index_content.append(f"| [{title}]({safe_link(md)}) | [Xem bài viết →]({safe_link(md)}) |\n")
        index_content.append("\n")

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
