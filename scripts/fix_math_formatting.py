import os
import re

def fix_math_formats(content):
    placeholders = []
    
    def protect(match):
        placeholders.append(match.group(0))
        return f"___MATH_PROTECT_{len(placeholders)-1}___"

    # 1. Protect code blocks and existing math
    content = re.sub(r'```.*?```', protect, content, flags=re.DOTALL)
    content = re.sub(r'`.*?`', protect, content)
    content = re.sub(r'\$\$.*?\$\$', protect, content, flags=re.DOTALL)
    content = re.sub(r'\$.*?\$', protect, content)

    # 2. Convert block math [ \n formula \n ]
    def block_converter(match):
        inner = match.group(1).strip()
        return f"\n$$\n{inner}\n$$\n"

    # Block patterns
    content = re.sub(r'\n\s*\[\s*\n(.*?)\n\s*\]', block_converter, content, flags=re.DOTALL)
    content = re.sub(r'\\\[(.*?)\\\]', block_converter, content, flags=re.DOTALL)
    
    # Handle the side-by-side case from the user screenshot if they are literal on one line
    # [ X_{out} = ... ] [ Y = ... ]
    content = re.sub(r'(?<!\w)\[ ([^\]\n]+?=.*?\\text\{.*?\}|[^\]\n]+?=.*?_{.*?\}|[^\]\n]+?=.*?\\times) \]', 
                     lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", content)

    # Protect newly created math blocks
    content = re.sub(r'\$\$.*?\$\$', protect, content, flags=re.DOTALL)

    # 3. Handle inline math in parentheses ( ... )
    def inline_heuristic(match):
        inner = match.group(1).strip()
        
        # Guard against empty or huge
        if not inner or len(inner) > 150:
            return f"({match.group(1)})"

        # Exclude common code/programming patterns
        if '[' in inner or ']' in inner or "'" in inner or '"' in inner:
            return f"({match.group(1)})"
        
        # Check for code keywords
        code_keywords = ['torch', 'model', 'tokenizer', 'batch', 'layer', 'loss', 'print', 'return', 'def ', 'class ', 'import ']
        if any(word in inner.lower() for word in code_keywords):
            return f"({match.group(1)})"

        is_math = False
        # LaTeX commands
        if '\\' in inner:
            is_math = True
        # Single letter variables
        elif len(inner) == 1 and inner.isalpha():
            is_math = True
        # Subscripts/Superscripts
        elif '_' in inner or '^' in inner:
            # Check if it looks like a math variable with subscript (e.g. z_i, W_Q)
            if re.match(r'^[A-Za-z]_[A-Za-z0-9]$|^[A-Za-z]_{[^{}]+}$|^[A-Za-z]\^[A-Za-z0-9]$|^[A-Za-z]\^{ [^{}]+ }$', inner):
                is_math = True
            elif len(inner) < 10 and not any(c.isdigit() for c in inner): # simple x_out etc
                 is_math = True
        # Math operators
        elif any(op in inner for op in [' \in ', ' \approx ', ' \times ', ' \cdot ', ' \pm ', ' = ']):
            is_math = True
        elif re.match(r'^[A-Za-z0-9\s\+\-\*\/\!\(\)\=\.]{1,20}$', inner) and any(c in inner for c in '+*/='):
            is_math = True
            
        if is_math:
            return f"${inner}$"
        return f"({match.group(1)})"

    # Target (variable) or (expression)
    content = re.sub(r'\(([^\)\n]+?)\)', inline_heuristic, content)
    
    # 4. Handle escaped inline math \( ... \)
    content = re.sub(r'\\\((.*?)\\\)', r'$\1$', content)

    # 5. Restore placeholders
    for i in range(len(placeholders)-1, -1, -1):
        content = content.replace(f"___MATH_PROTECT_{i}___", placeholders[i])

    return content

def fix_typos(content):
    content = content.replace("Ave18_rage", "Average")
    return content

def process_docs(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if '.git' in dirs: dirs.remove('.git')
        for file in files:
            if file.endswith('.md'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    new_content = fix_math_formats(content)
                    new_content = fix_typos(new_content)
                    
                    if new_content != content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Fixed: {path}")
                except Exception as e:
                    print(f"Error: {path} - {e}")

if __name__ == "__main__":
    process_docs("/Users/pixibox/Aero-HowtoLLMs/docs")
