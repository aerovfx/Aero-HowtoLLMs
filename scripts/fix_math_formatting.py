import os
import re

def fix_content_structure(content):
    # 1. Loại bỏ các khối code block markdown bao ngoài (```md ... ```)
    # Chúng ta quét toàn bộ file và nếu thấy cấu trúc này bao quanh phần lớn nội dung, ta bóc nó ra.
    lines = content.split('\n')
    if len(lines) > 10: # Chỉ xử lý file đủ lớn
        # Kiểm tra xem có dấu đóng mở ```md / ``` rải rác không
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
                # Dọn dẹp nội dung toán học: ghép thành 1 dòng, xóa khoảng trắng thừa
                formula = " ".join([l.strip() for l in math_buffer if l.strip()])
                formula = re.sub(r'={3,}', '=', formula)
                
                # Đảm bảo có dòng trống TRƯỚC khối $$
                if new_lines and new_lines[-1].strip() != '':
                    new_lines.append('')
                
                new_lines.append('$$')
                new_lines.append(formula)
                new_lines.append('$$')
                
                # Đảm bảo có dòng trống SAU khối $$ (sẽ được thêm ở vòng lặp sau hoặc cuối)
                new_lines.append('')
                math_buffer = []
        else:
            if in_math_block:
                math_buffer.append(line)
            else:
                new_lines.append(line)
    
    # Ghép lại và dọn dẹp dòng trống trùng lặp
    content = '\n'.join(new_lines)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Sửa lỗi chính tả
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
                        print(f"Verified & Fixed: {path}")
                except Exception as e:
                    print(f"Error {path}: {e}")
    print(f"Finished. Updated {count} files.")

if __name__ == "__main__":
    run_fix("/Users/pixibox/Aero-HowtoLLMs/docs")
