import os
import re

def fix_content_structure(content):
    # 1. Loại bỏ các khối ```md và ``` bao ngoài nếu chúng không phục vụ mục đích code block thực tế
    # Đối với các file tài liệu, ta thường muốn nội dung hiển thị trực tiếp.
    lines = content.split('\n')
    new_lines = []
    
    # Những dòng markdown "giả" cần xóa
    for line in lines:
        s = line.strip()
        if s == '```md' or s == '```':
            # Kiểm tra xem đây là mở một code block thực tế hay chỉ là bao ngoài
            # Trong ngữ cảnh này, ta tàm thời xóa hết các thẻ md bao ngoài
            continue
        new_lines.append(line)
    
    content = '\n'.join(new_lines)

    # 2. Xóa bỏ các tham chiếu rác :contentReference[oaicite:...]
    content = re.sub(r':contentReference\[oaicite:\d+\]\{index=\d+\}', '', content)
    
    # 3. Chuyển đổi các khối [ ] hoặc \[ \] sang $$
    # Dùng regex cẩn thận để tránh bắt nhầm
    content = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", content)
    
    return content

def fix_math_ultra_clean(content):
    # Xử lý cấu trúc bao và rác trước
    content = fix_content_structure(content)

    # Thuật toán xử lý dòng trống nội bộ và GHÉP DÒNG toán học
    # Tìm tất cả khối $$ ... $$
    def clean_math(match):
        inner = match.group(1).strip()
        # Ghép thành 1 dòng
        formula = " ".join([l.strip() for l in inner.split('\n') if l.strip()])
        # Sửa lỗi dấu bằng ASCII
        formula = re.sub(r'={3,}', '=', formula)
        # Sửa lỗi attention specific (đảm bảo softmax dính liền hoặc cách 1 khoảng)
        formula = formula.replace('softmax (', 'softmax(')
        return f"$$\n{formula}\n$$"

    content = re.sub(r'\$\$\s*([\s\S]*?)\s*\$\$', clean_math, content)

    # Đảm bảo có dòng trống BÊN NGOÀI $$ (trước và sau)
    content = re.sub(r'([^\n])\n*\s*\$\$', r'\1\n\n$$', content)
    content = re.sub(r'\$\$\n*\s*([^\n])', r'$$\n\n\1', content)

    # Dọn dẹp dòng trống dư thừa
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
                        content = f.read()
                    
                    fixed = fix_math_ultra_clean(content)
                    
                    if fixed != content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(fixed)
                        count += 1
                        print(f"Fixed: {path}")
                except Exception as e:
                    print(f"Error processing {path}: {e}")
    print(f"Total files updated: {count}")

if __name__ == "__main__":
    run_fix("/Users/pixibox/Aero-HowtoLLMs/docs")
