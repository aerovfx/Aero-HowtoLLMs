import os
import re

def fix_math_ultra_clean(content):
    # 1. Chuyển đổi các khối [ ] hoặc \[ \] sang $$ trước
    content = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", content)
    
    # 2. Xử lý các dòng "=====" bên trong khối $$ (biến thành single =)
    # Tìm tất cả khối $$ ... $$
    def replace_long_equals(match):
        inner = match.group(1)
        # Thay thế các dòng chỉ chứa dấu = (ít nhất 3 dấu) bằng một dấu =
        # Lưu ý: cần xử lý cả trường hợp có khoảng trắng
        inner = re.sub(r'\n\s*={3,}\s*\n', '\n = \n', inner)
        return f"$$\n{inner.strip()}\n$$"

    content = re.sub(r'\$\$\s*([\s\S]*?)\s*\$\$', replace_long_equals, content)

    # 3. Thuật toán xử lý dòng trống nội bộ cực mạnh (như đã làm trước đó)
    lines = content.split('\n')
    new_lines = []
    in_math_block = False
    math_buffer = []
    
    for line in lines:
        stripped = line.strip()
        if stripped == '$$':
            if not in_math_block:
                in_math_block = True
                math_buffer = ['$$']
            else:
                in_math_block = False
                math_buffer.append('$$')
                cleaned_block = [math_buffer[0]]
                for m_line in math_buffer[1:-1]:
                    if m_line.strip():
                        cleaned_block.append(m_line)
                cleaned_block.append(math_buffer[-1])
                
                if new_lines and new_lines[-1].strip():
                    new_lines.append('')
                
                new_lines.extend(cleaned_block)
                new_lines.append('')
                math_buffer = []
        else:
            if in_math_block:
                math_buffer.append(line)
            else:
                new_lines.append(line)
    
    content = '\n'.join(new_lines)

    # 4. Dọn dẹp dòng trống dư thừa
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 5. Sửa lỗi chính tả
    content = content.replace("Ave18_rage", "Average")
    
    return content

def run_fix(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                fixed = fix_math_ultra_clean(content)
                
                if fixed != content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(fixed)
                    count += 1
                    print(f"Ultra-Cleaned & Fixed Long Equals: {path}")
    print(f"Total files updated: {count}")

if __name__ == "__main__":
    run_fix("/Users/pixibox/Aero-HowtoLLMs/docs")
