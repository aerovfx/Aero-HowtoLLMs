import os
import re

def ensure_math_spacing(content):
    """
    GitHub requires $$ math blocks to:
    1. Have a blank line BEFORE the opening $$
    2. Have a blank line AFTER the closing $$
    3. $$ must be on its own line (not inline with text)
    
    This script fixes all $$ blocks to ensure proper spacing.
    """
    
    # Step 1: Normalize $$ blocks that are on the same line as content
    # e.g., "text: $$\nformula\n$$" -> "text:\n\n$$\nformula\n$$"
    # Match text before $$ on the same line
    content = re.sub(r'([^\n])\s*\$\$\s*\n', r'\1\n\n$$\n', content)
    
    # Match text after $$ on the same line  
    content = re.sub(r'\n\$\$\s*([^\n\$])', r'\n$$\n\n\1', content)
    
    # Step 2: Ensure blank line BEFORE $$
    # Replace "\n$$\n" with "\n\n$$\n" when there's no blank line before
    content = re.sub(r'([^\n])\n\$\$\n', r'\1\n\n$$\n', content)
    
    # Step 3: Ensure blank line AFTER $$
    # Replace "\n$$\n" (closing) with "\n$$\n\n" when followed by non-blank
    content = re.sub(r'\n\$\$\n([^\n])', r'\n$$\n\n\1', content)
    
    # Step 4: Clean up excessive blank lines (more than 2 consecutive)
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    return content


def fix_bracket_math(content):
    """Convert remaining [ formula ] style math to $$ formula $$"""
    
    def convert_block(match):
        inner = match.group(1).strip()
        if not inner:
            return match.group(0)
        if any(c in inner for c in ['\\', '_', '^', '=']):
            return f"\n\n$$\n{inner}\n$$\n\n"
        return match.group(0)
    
    content = re.sub(r'\n\[\s*\n([\s\S]*?)\n\]\s*(?=\n)', convert_block, content)
    content = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", content)
    content = re.sub(r'\\\((.*?)\\\)', lambda m: f"${m.group(1).strip()}$", content)
    
    return content


def fix_typos(content):
    content = content.replace("Ave18_rage", "Average")
    return content


def process_docs(base_dir):
    count = 0
    for root, dirs, files in os.walk(base_dir):
        if '.git' in dirs: dirs.remove('.git')
        if 'node_modules' in dirs: dirs.remove('node_modules')
        for file in files:
            if file.endswith('.md'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    new_content = fix_bracket_math(content)
                    new_content = fix_typos(new_content)
                    new_content = ensure_math_spacing(new_content)
                    
                    if new_content != content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        count += 1
                        print(f"Fixed: {os.path.relpath(path, base_dir)}")
                except Exception as e:
                    print(f"Error: {path} - {e}")
    print(f"\nTotal files fixed: {count}")


if __name__ == "__main__":
    process_docs("/Users/pixibox/Aero-HowtoLLMs/docs")
