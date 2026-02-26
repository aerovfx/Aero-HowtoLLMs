import os
import re

def fix_math_formats(content):
    """
    Fix math formulas for GitHub Markdown rendering.
    Converts [ formula ] and \[ formula \] to $$ formula $$ blocks.
    """
    
    # === PHASE 0: Fix [ \n formula \n ] BEFORE protecting code blocks ===
    # This pattern is unambiguous: a line with just "[", math content, then a line with just "]" 
    def convert_bracket_block(match):
        inner = match.group(1).strip()
        if not inner:
            return match.group(0)
        # Check for math-like content
        if any(c in inner for c in ['\\', '_', '^', '=']):
            return f"\n$$\n{inner}\n$$\n"
        return match.group(0)
    
    # [ on its own line, content, ] on its own line
    content = re.sub(r'\n\[\s*\n([\s\S]*?)\n\]\s*(?=\n)', convert_bracket_block, content)
    
    # === PHASE 1: Protect code blocks and existing valid constructs ===
    protections = []
    
    def protect(match):
        protections.append(match.group(0))
        return f"__PROT_{len(protections)-1}__"
    
    # Protect fenced code blocks
    content = re.sub(r'```[\s\S]*?```', protect, content)
    # Protect inline code
    content = re.sub(r'`[^`\n]+`', protect, content)
    # Protect existing $$ blocks
    content = re.sub(r'\$\$[\s\S]*?\$\$', protect, content)
    # Protect existing $ inline math
    content = re.sub(r'\$[^\$\n]+?\$', protect, content)
    # Protect markdown links
    content = re.sub(r'\[([^\]]*?)\]\([^\)]*?\)', protect, content)
    # Protect reference citations [1], [2]
    content = re.sub(r'\[\d+\]', protect, content)
    
    # === PHASE 2: Convert remaining \[ \] and \( \) ===
    content = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", content)
    content = re.sub(r'\\\((.*?)\\\)', lambda m: f"${m.group(1).strip()}$", content)
    
    # === PHASE 3: Fix typos ===
    content = content.replace("Ave18_rage", "Average")
    
    # === PHASE 4: Restore protections ===
    for i in range(len(protections) - 1, -1, -1):
        content = content.replace(f"__PROT_{i}__", protections[i])
    
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
                    
                    new_content = fix_math_formats(content)
                    
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
