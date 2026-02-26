import os
import re

def fix_math_ultra_clean(content):
    # 1. Chuyển đổi các khối [ ] hoặc \[ \] sang $$ trước
    content = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", content)
    
    # 2. Thuật toán xử lý dòng trống nội bộ cực mạnh
    lines = content.split('\n')
    new_lines = []
    in_math_block = False
    math_buffer = []
    
    for line in lines:
        stripped = line.strip()
        if stripped == '$$':
            if not in_math_block:
                # Bắt đầu khối
                in_math_block = True
                math_buffer = ['$$']
            else:
                # Kết thúc khối
                in_math_block = False
                math_buffer.append('$$')
                # Làm sạch buffer: chỉ giữ lại $$ ở đầu/cuối và các dòng có nội dung ở giữa
                cleaned_block = [math_buffer[0]] # Mở $$
                for m_line in math_buffer[1:-1]:
                    if m_line.strip(): # Chỉ giữ dòng có chữ
                        cleaned_block.append(m_line)
                cleaned_block.append(math_buffer[-1]) # Đóng $$
                
                # Đảm bảo có dòng trống trước khối nếu chưa có
                if new_lines and new_lines[-1].strip():
                    new_lines.append('')
                
                new_lines.extend(cleaned_block)
                # Đảm bảo có dòng trống sau khối
                new_lines.append('')
                math_buffer = []
        else:
            if in_math_block:
                math_buffer.append(line)
            else:
                new_lines.append(line)
    
    # Ghép lại
    content = '\n'.join(new_lines)

    # 3. Dọn dẹp dòng trống dư thừa (> 2 dòng)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 4. Sửa lỗi chính tả
    content = content.replace("Ave18_rage", "Average")
    
    return content

def run_fix(directory):
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
                    print(f"Ultra-Cleaned: {path}")

if __name__ == "__main__":
    run_fix("/Users/pixibox/Aero-HowtoLLMs/docs")
