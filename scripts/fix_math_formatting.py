import os
import re

def fix_content_structure(content):
    # 1. Loại bỏ các khối ```md và ``` bao ngoài
    lines = content.split('\n')
    if len(lines) > 5:
        filtered_lines = []
        for line in lines:
            if line.strip() in ['```md', '```']:
                continue
            filtered_lines.append(line)
        content = '\n'.join(filtered_lines)

    # 2. Xóa tham chiếu oaicite
    content = re.sub(r':contentReference\[oaicite:\d+\]\{index=\d+\}', '', content)
    
    # 3. Chuyển \[ \] sang $$
    content = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", content)
    
    return content

def fix_math_ultra_clean(content):
    content = fix_content_structure(content)

    # Dọn dẹp sơ bộ các lỗi lặp $$ do các lần chạy trước
    content = re.sub(r'\$\$\s*\$\$', '', content)
    content = re.sub(r'\$\$\s*\n\s*\$\$', '', content)

    lines = content.split('\n')
    new_lines = []
    in_math_block = False
    math_buffer = []
    
    for line in lines:
        stripped = line.strip()
        if stripped == '$$':
            if not in_math_block:
                in_math_block = True
                math_buffer = []
            else:
                in_math_block = False
                # Xử lý nội dung toán học
                formula = " ".join([l.strip() for l in math_buffer if l.strip()])
                if not formula:
                    continue # Bỏ qua khối rỗng
                
                # Biến đổi chuẩn hóa
                formula = re.sub(r'={3,}', '=', formula)
                formula = re.sub(r'P\(([^)]*?)\|([^)]*?)\)', r'P(\1 \\mid \2)', formula)
                formula = re.sub(r'\{(.*?)\<(.*?)\}', r'{\1\\lt \2}', formula)
                formula = formula.replace('^*', '^{\\ast}')
                formula = formula.replace('$', '') # Xóa dấu $ kẹt bên trong
                
                if new_lines and new_lines[-1].strip() != '':
                    new_lines.append('')
                
                new_lines.append('$$')
                new_lines.append(formula)
                new_lines.append('$$')
                new_lines.append('')
                math_buffer = []
        else:
            if in_math_block:
                math_buffer.append(line)
            else:
                # Sửa inline math: các ký tự lén lút
                def fix_inline(m):
                    inner = m.group(0)
                    inner = inner.replace('|', '\\mid')
                    inner = re.sub(r'\{(.*?)\<(.*?)\}', r'{\1\\lt \2}', inner)
                    inner = inner.replace('^*', '^{\\ast}')
                    return inner
                
                line = re.sub(r'\$.*?\$', fix_inline, line)
                new_lines.append(line)
    
    content = '\n'.join(new_lines)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Sửa lỗi đặc biệt cho file BERT Character Counts mà tác giả viết kiểu P$L=k$
    # Ta chỉ sửa nếu thấy signature "P$..." hoặc "O$..."
    content = re.sub(r'P\$([\s\S]*?)\$', r'$P(\1)$', content)
    content = re.sub(r'O\$([\s\S]*?)\$', r'$O(\1)$', content)
    content = re.sub(r'\\ell\$([\s\S]*?)\$', r'\\ell(\1)', content)
    
    content = content.replace("Ave18_rage", "Average")
    
    return content

def run_fix(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        old_content = f.read()
                    new_content = fix_math_ultra_clean(old_content)
                    if new_content != old_content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        count += 1
                        print(f"Purified: {path}")
                except Exception as e:
                    print(f"Error {path}: {e}")
    print(f"Done. Updated {count} files.")

if __name__ == "__main__":
    run_fix("/Users/pixibox/Aero-HowtoLLMs/docs")
