import os
import re

def fix_content_structure(content):
    # 1. Loại bỏ các khối ```md và ``` bao ngoài
    # Chỉ xóa nếu nó bao quanh phần lớn nội dung
    lines = content.split('\n')
    if len(lines) > 10:
        # Nếu dòng đầu là ```md hoặc ``` và dòng cuối (hoặc gần cuối) cũng thế
        # Chúng ta sẽ xóa chúng.
        if lines[0].strip() in ['```md', '```']:
            lines = lines[1:]
        if lines[-1].strip() in ['```md', '```']:
            lines = lines[:-1]
        elif len(lines) > 1 and lines[-2].strip() in ['```md', '```'] and lines[-1].strip() == '':
            lines = lines[:-2]
            
        content = '\n'.join(lines)

    # 2. Xóa tham chiếu oaicite
    content = re.sub(r':contentReference\[oaicite:\d+\]\{index=\d+\}', '', content)
    
    # 3. Chuyển \[ \] sang $$
    content = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", content)
    
    return content

def fix_math_ultra_clean(content):
    content = fix_content_structure(content)

    # Dọn dẹp sơ bộ các lỗi lặp $$
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
                
                # CHẾ ĐỘ THÔNG MINH: Giữ xuống dòng nếu có \\ hoặc \begin
                math_content = "\n".join(math_buffer).strip()
                if not math_content:
                    continue
                
                if '\\\\' in math_content or '\\begin' in math_content:
                    # Giữ nguyên cấu trúc dòng cho cases, align, v.v.
                    formula = "\n".join([l.strip() for l in math_buffer if l.strip()])
                else:
                    # Ghép thành một dòng cho các công thức đơn giản
                    formula = " ".join([l.strip() for l in math_buffer if l.strip()])
                
                # Biến đổi chuẩn hóa
                formula = re.sub(r'={3,}', '=', formula)
                
                # SỬA LỖI \mid: Chỉ thay | nếu là xác suất P(A|B) hoặc đứng giữa khoảng trắng
                formula = re.sub(r'P\(([^)]*?)\|([^)]*?)\)', r'P(\1 \\mid \2)', formula)
                formula = re.sub(r'(?<=\s)\|(?=\s)', r'\\mid', formula)
                
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
                # Sửa inline math
                def fix_inline(m):
                    inner = m.group(0)
                    # Không thay | bừa bãi trong inline vì hay gặp |V|
                    inner = re.sub(r'P\(([^)]*?)\|([^)]*?)\)', r'P(\1 \\mid \2)', inner)
                    inner = re.sub(r'\{(.*?)\<(.*?)\}', r'{\1\\lt \2}', inner)
                    inner = inner.replace('^*', '^{\\ast}')
                    return inner
                
                line = re.sub(r'\$.*?\$', fix_inline, line)
                new_lines.append(line)
    
    content = '\n'.join(new_lines)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Sửa lỗi lười
    content = re.sub(r'P\$([\s\S]*?)\$', r'$P(\1)$', content)
    content = re.sub(r'O\$([\s\S]*?)\$', r'$O(\1)$', content)
    content = re.sub(r'\\ell\$([\s\S]*?)\$', r'\\ell(\1)', content)
    
    content = content.replace("Ave18_rage", "Average")
    content = content.replace("\\midV", "|V") # Sửa lỗi cũ
    content = content.replace("\\mid V", "|V") # Sửa lỗi cũ
    
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
