import os
import re

def fix_content_structure(content):
    # 1. Loại bỏ các khối code block markdown bao ngoài (```md ... ```)
    lines = content.split('\n')
    if len(lines) > 5:
        filtered_lines = []
        for line in lines:
            if line.strip() in ['```md', '```']:
                continue
            filtered_lines.append(line)
        content = '\n'.join(filtered_lines)

    # 2. Xóa bỏ các tham chiếu rác :contentReference[oaicite:...]
    content = re.sub(r':contentReference\[oaicite:\d+\]\{index=\d+\}', '', content)
    
    # 3. Chuyển đổi các khối \[ \] sang $$
    content = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", content)
    
    return content

def fix_lazy_math(content):
    # Sửa lỗi P$L=k$ -> $P(L=k)$
    content = re.sub(r'P\$([\s\S]*?)\$', r'$P(\1)$', content)
    
    # Sửa lỗi \ell$t$ -> $\ell(t)$
    content = re.sub(r'\\ell\$([\s\S]*?)\$', r'\\ell(\1)', content)
    
    # Sửa lỗi O$m^2$ -> $O(m^2)$
    content = re.sub(r'O\$([\s\S]*?)\$', r'$O(\1)$', content)

    # Sửa các dòng toán học thô (không có delimiter)
    # Tìm các biểu thức có chứa \, _, ^, \approx, \ge, v.v. nhưng không nằm trong $ hoặc $$
    
    latex_symbols = [r'\\approx', r'\\ge', r'\\le', r'\\prod', r'\\sum', r'\\mathbb', r'\\mathcal', r'\\ell', r'\\log', r'\\infty', r'\\propto', r'\\partial', r'\\nabla']
    
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Nếu dòng chứa symbol LaTeX thô
        has_symbol = any(re.search(sym, stripped) for sym in latex_symbols)
        # Hoặc chứa các biểu thức biến số w_i, x_n, m = n
        has_var = re.search(r'[a-zA-Z]_[0-9a-zA-Z]', stripped) or re.search(r'[a-zA-Z]\s*=\s*[a-zA-Z]', stripped)
        
        # Nếu dòng trông như toán học nhưng chưa có $$
        if (has_symbol or has_var) and not stripped.startswith('$') and not stripped.startswith('#') and not stripped.startswith('-') and not stripped.startswith('|') and len(stripped) < 100:
            if '=' in stripped or r'\approx' in stripped or r'\ge' in stripped or r'\le' in stripped:
                line = f"$$\n{stripped}\n$$"
        
        # Xử lý inline symbols kẹt trong text: \approx, \ge, v.v.
        for sym in latex_symbols:
            # Nếu symbol đứng độc lập (không có $)
            line = re.sub(f'(?<!\\$){sym}', f'${sym}$', line)
            
        new_lines.append(line)
        
    return '\n'.join(new_lines)

def fix_math_ultra_clean(content):
    # 1. Sửa lỗi "lười" trước
    content = fix_lazy_math(content)
    # 2. Xử lý cấu trúc
    content = fix_content_structure(content)

    lines = content.split('\n', -1)
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
                # Ghép nội dung toán học
                formula = " ".join([l.strip() for l in math_buffer if l.strip()])
                formula = re.sub(r'={3,}', '=', formula)
                
                # CHUẨN HÓA LATEX CHO GITHUB RENDERER
                formula = re.sub(r'P\(([^)]*?)\|([^)]*?)\)', r'P(\1 \\mid \2)', formula)
                formula = re.sub(r'\{(.*?)\<(.*?)\}', r'{\1\\lt \2}', formula)
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
                # Sửa inline math
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
