import os
import re

def scan_math(base_dir):
    patterns = {
        'block': re.compile(r'\\\[(.*?)\\\]', re.DOTALL),
        'inline': re.compile(r'\\\((.*?)\\\)'),
        'dollar_block': re.compile(r'\$\$(.*?)\$\$', re.DOTALL),
        'dollar_inline': re.compile(r'\$([^\$].*?)\$')
    }
    
    counts = {k: 0 for k in patterns}
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if not file.endswith('.md'): continue
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                for key, p in patterns.items():
                    finds = p.findall(content)
                    if finds:
                        counts[key] += len(finds)
                        if counts[key] < 5: # Show some examples
                            print(f"Found {key} in {path}: {finds[0][:100]}")
            except:
                pass
                
    print("\nSummary counts:", counts)

if __name__ == "__main__":
    scan_math("/Users/pixibox/Aero-HowtoLLMs/docs")
