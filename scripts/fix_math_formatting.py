import os
import re

def fix_math_final_v2(content):
    # 1. Chuyển đổi các khối [ \n content \n ] sang dạng $$
    def convert_bracket_block(match):
        inner = match.group(1).strip()
        if not inner: return match.group(0)
        return f"\n\n$$\n{inner}\n$$\n\n"

    # Xử lý khối [ trên dòng riêng và ] trên dòng riêng
    content = re.sub(r'\n\[\s*\n([\s\S]*?)\n\s*\](?=\n|$)', convert_bracket_block, content)
    
    # 2. Xử lý khối \[ ... \]
    content = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", content)

    # 3. CHỮA LỖI QUAN TRỌNG: Loại bỏ dòng trống BÊN TRONG $$
    # Tìm tất cả các khối $$ ... $$ và làm sạch nội dung bên trong
    def clean_math_content(match):
        # match.group(1) là nội dung giữa hai cặp $$
        inner = match.group(1).strip()
        return f"$$\n{inner}\n$$"

    # Regex này bắt mọi thứ giữa $$ và $$ (bao gồm cả xuống dòng)
    content = re.sub(r'\$\$\s*([\s\S]*?)\s*\$\$', clean_math_content, content)
    
    # 4. Đảm bảo có dòng trống BÊN NGOÀI $$
    # Trước $$
    content = re.sub(r'([^\n])\n\s*\$\$', r'\1\n\n$$', content)
    # Sau $$ (khép)
    content = re.sub(r'\$\$\n([^\n])', r'$$\n\n\1', content)

    # 5. Dọn dẹp dòng trống dư thừa (> 2 dòng)
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    # 6. Sửa lỗi chính tả
    content = content.replace("Ave18_rage", "Average")
    
    return content

def process_and_push(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    old_content = f.read()
                
                new_content = fix_math_final_v2(old_content)
                
                if new_content != old_content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    count += 1
                    print(f"Verified & Fixed: {path}")
    
    print(f"Finished. Updated {count} files.")

if __name__ == "__main__":
    process_and_push("/Users/pixibox/Aero-HowtoLLMs/docs")
