import os
import re

def fix_content_structure(content):
    # 1. Loại bỏ các khối code block markdown bao ngoài (```md ... ```)
    lines = content.split('\n')
    if len(lines) > 5:
        new_lines = []
        for line in lines:
            if line.strip() in ['```md', '```']:
                continue
            new_lines.append(line)
        content = '\n'.join(new_lines)

    # 2. Xóa bỏ các tham chiếu rác :contentReference[oaicite:...]
    content = re.sub(r':contentReference\[oaicite:\d+\]\{index=\d+\}', '', content)
    
    # 3. Chuyển đổi các khối \[ \] sang $$
    content = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", content)
    
    return content

def fix_math_ultra_clean(content):
    content = fix_content_structure(content)

    # Xử lý theo dòng để tránh lỗi regex matching nhầm vào bên trong $$
    lines = content.split('\n')
    new_lines = []
    in_math_block = False
    math_buffer = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == '$$':
            if not in_math_block:
                in_math_block = True
                math_buffer = []
            else:
                in_math_block = False
                # Dọn dẹp nội dung toán học
                formula = " ".join([l.strip() for l in math_buffer if l.strip()])
                formula = re.sub(r'={3,}', '=', formula)
                
                # Sửa lỗi phổ biến khiến GitHub render sai
                # 1. Thay | bằng \mid trong các xác suất P(...)
                formula = re.sub(r'P\(([^)]*?)\|([^)]*?)\)', r'P(\1 \mid \2)', formula)
                # 2. Thay < bằng \lt trong subscript
                formula = re.sub(r'(\{.*?)\<(.*?)\}', r'\1\\lt \2}', formula)
                # 3. Đảm bảo \ast được dùng thay cho * thô
                formula = formula.replace('^*', '^{\\ast}')
                
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
                # Sửa inline math x_{<t} -> x_{\lt t}
                line = re.sub(r'(\$.*?\{.*?)\<(.*?\}.*?\$)', r'\1\\lt \2', line)
                # Sửa inline math x_{t} | x_{<t} -> x_{t} \mid x_{\lt t}
                line = re.sub(r'(\$.*?)\|(.*?\$)', r'\1\\mid \2', line)
                new_lines.append(line)
    
    content = '\n'.join(new_lines)
    content = re.sub(r'\n{3,}', '\n\n', content)
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
