import os
import re

def fix_content_structure(content):
    # 1. Loại bỏ các khối ```md và ``` bao ngoài
    lines = content.split('\n')
    new_lines = []
    # Chỉ xóa nếu dòng đó đúng bằng ```md hoặc ``` và không có dấu hiệu là code block thực sự (như có nội dung bên cạnh)
    for line in lines:
        s = line.strip()
        if s == '```md' or s == '```':
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
                
                # Xử lý nội dung bên trong $$
                # 1. Loại bỏ các dòng trống
                clean_lines = [l.strip() for l in math_buffer if l.strip()]
                
                # 2. Kết nối các dòng: 
                # Nếu một dòng kết thúc bằng \\ thì giữ nguyên xuống dòng (cho cases, v.v.)
                # Nếu không, nối bằng khoảng trắng
                final_formula_lines = []
                current_segment = []
                for l in clean_lines:
                    current_segment.append(l)
                    if l.endswith('\\\\') or l.endswith('\\'):
                        # Nếu kết thúc bằng \ hoặc \\, ta coi như kết thúc một dòng logic trong LaTeX
                        # Lưu ý: GitHub LaTeX dùng \\ cho xuống dòng trong cases
                        # Ta chuẩn hóa: nếu là \ thì đổi thành \\
                        line_text = " ".join(current_segment)
                        if line_text.endswith('\\') and not line_text.endswith('\\\\'):
                            line_text += '\\'
                        final_formula_lines.append(line_text)
                        current_segment = []
                
                if current_segment:
                    final_formula_lines.append(" ".join(current_segment))
                
                formula = "\n".join(final_formula_lines)
                
                # Sửa lỗi dấu bằng ASCII
                formula = re.sub(r'={3,}', '=', formula)
                
                # Đảm bảo có dòng trống TRƯỚC khối $$
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
                        print(f"Verified & Fixed: {path}")
                except Exception as e:
                    print(f"Error {path}: {e}")
    print(f"Finished. Updated {count} files.")

if __name__ == "__main__":
    run_fix("/Users/pixibox/Aero-HowtoLLMs/docs")
