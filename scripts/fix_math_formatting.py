import os
import re

def fix_math_formats(content):
    # 1. Convert block math \[ ... \] to $$ ... $$
    # Ensure it's on a new line for better compatibility
    content = re.sub(r'\\\[(.*?)\\\]', r'\n$$\n\1\n$$\n', content, flags=re.DOTALL)
    
    # 2. Convert inline math \( ... \) to $ ... $
    # Ensure no spaces inside $ for better compatibility ($ math $ -> $math$)
    def inline_replacer(match):
        inner = match.group(1).strip()
        return f"${inner}$"
    
    content = re.sub(r'\\\((.*?)\\\)', inline_replacer, content)
    
    # 3. Clean up existing $$ math $$ if they are on the same line and shouldn't be
    # (Optional: for simplicity we can just ensure newlines around all $$ blocks)
    # content = re.sub(r'(?<!\n)\s*\$\$(.*?)\$\$', r'\n$$\n\1\n$$', content, flags=re.DOTALL)
    
    return content

def fix_typos(content):
    # Fix the weird "Ave18_rage" corruption
    content = content.replace("Ave18_rage", "Average")
    content = content.replace("Ave18_RAGe", "Average") # If it exists
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
