import os
import re

def fix_cases_syntax(content):
    # Tìm các khối cases: \begin{cases} ... \end{cases}
    def internal_fix(match):
        inner = match.group(1)
        # Nếu thấy " \ " (khoảng trắng + backslash + khoảng trắng) mà không phải " \\ "
        # Hoặc các pattern phổ biến của lỗi merge
        
        # 1. Phát hiện các điểm ngắt dòng tiềm năng trong cases (thường có dấu \ hoặc \ dán liền)
        # Sửa: text \ -\infty -> text \\ -\infty
        # Sửa: text \ 0 -> text \\ 0
        new_inner = re.sub(r'(\s)\\(\s|-|\d|[a-zA-Z])', r'\1\\\\\2', inner)
        
        # Đảm bảo không bị nhân đôi quá đà nếu đã có \\
        new_inner = new_inner.replace('\\\\\\\\', '\\\\')
        
        return f'\\begin{{cases}}{new_inner}\\end{{cases}}'

    # Regex bắt khối cases
    content = re.sub(r'\\begin\{cases\}([\s\S]*?)\\end\{cases\}', internal_fix, content)
    
    return content

def run_fix(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    old_content = f.read()
                
                new_content = fix_cases_syntax(old_content)
                
                if new_content != old_content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Fixed Cases: {path}")

if __name__ == "__main__":
    run_fix("/Users/pixibox/Aero-HowtoLLMs/docs")
