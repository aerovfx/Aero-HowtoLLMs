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
    content = re.sub(r'\\ell\$([\s\S]*?)\$', r'\\ell(\1)', content)
    content = re.sub(r'O\$([\s\S]*?)\$', r'$O(\1)$', content)

    latex_symbols = [r'\\approx', r'\\ge', r'\\le', r'\\prod', r'\\sum', r'\\mathbb', r'\\mathcal', r'\\ell', r'\\log', r'\\infty', r'\\propto', r'\\partial', r'\\nabla', r'\\mathrm', r'\\mathbb', r'\\mathcal']
    
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
            
        # Tìm xem có symbol LaTeX nào không nằm trong $ hay không
        # Chúng ta dùng một regex để tìm các symbol không bắt đầu bằng $
        
        # 1. Xử lý các dòng trông như phương trình toán học
        has_symbol = any(re.search(sym, stripped) for sym in latex_symbols)
        has_eq = '=' in stripped or r'\approx' in stripped or r'\ge' in stripped or r'\le' in stripped
        is_shorthand = re.search(r'^[a-zA-Z]_[0-9a-zA-Z]\s*=', stripped) or re.search(r'^[a-zA-Z]\s*=\s*[a-zA-Z0-9]', stripped)
        
        if (has_symbol or has_eq or is_shorthand) and not stripped.startswith('$') and not stripped.startswith('#') and not stripped.startswith('!') and not stripped.startswith('|') and not stripped.startswith('-') and len(stripped) < 150:
            # Wrap nguyên dòng trong $$
            # Trước đó hãy xóa các phím $ lẻ tẻ trong dòng
            clean_stripped = stripped.replace('$', '')
            line = f"$$\n{clean_stripped}\n$$"
        else:
            # Xử lý inline symbols kẹt trong text
            for sym in latex_symbols:
                # Tìm symbol KHÔNG nằm giữa $...$
                # Regex này tìm symbol và đảm bảo phía trước/sau không có $ rải rác gần đó
                # Cách đơn giản: replace (?<!\$)symbol(?!\$) -> $symbol$
                line = re.sub(f'(?<!\\$){sym}', f'${sym}$', line)
            
            # Sửa các biến số x_i kẹt trong text: (?<!\$)([a-zA-Z]_[0-9a-zA-Z])(?!\$)
            line = re.sub(r'(?<!\$)\b([a-zA-Z]_[0-9a-zA-Z\(\)])\b(?!\$)', r'$\1$', line)
            
        new_lines.append(line)
        
    content = '\n'.join(new_lines)
    
    # Dọn dẹp: $$ \n $math$ \n $$ -> $$ \n math \n $$
    content = re.sub(r'\$\$\n\s*\$(.*?)\$\n\s*\$\$', r'$$\n\1\n$$', content)
    
    return content

def fix_math_ultra_clean(content):
    content = fix_lazy_math(content)
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
                formula = " ".join([l.strip() for l in math_buffer if l.strip()])
                formula = re.sub(r'={3,}', '=', formula)
                formula = re.sub(r'P\(([^)]*?)\|([^)]*?)\)', r'P(\1 \\mid \2)', formula)
                formula = re.sub(r'\{(.*?)\<(.*?)\}', r'{\1\\lt \2}', formula)
                formula = formula.replace('^*', '^{\\ast}')
                
                # Xóa bất kỳ dấu $ nào kẹt trong formula $$
                formula = formula.replace('$', '')
                
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
