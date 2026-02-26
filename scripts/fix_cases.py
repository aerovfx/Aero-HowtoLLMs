import os
import re

def fix_cases_syntax(content):
    # Tìm các khối cases: \begin{cases} ... \end{cases}
    def internal_fix(match):
        inner = match.group(1)
        
        # 1. Sửa lỗi nhân đôi backslash cho các ký tự toán học (như \le, \infty) 
        # mà bị script trước đó làm hỏng thành \\le, \\infty
        # Chúng ta chỉ muốn \\ ở cuối dòng logic.
        
        # Đầu tiên, đưa tất cả \\ về \ ngoại trừ những chỗ thực sự là xuống dòng
        # Nhưng làm vậy hơi khó. Hãy thử pattern:
        # Nếu thấy \\ tiếp theo là một chữ cái (a-z) thì khả năng cao là symbol
        inner = re.sub(r'\\\\([a-zA-Z])', r'\\\1', inner)
        
        # 2. Bây giờ, tìm các điểm ngắt dòng thực sự. 
        # Trong các file bị hỏng, chúng thường là " \ " hoặc " \\ " (nhưng bị dính vào symbol)
        # Hoặc đơn giản là ta cần một dấu \\ giữa các điều kiện.
        
        # Giả sử cấu trúc là: giá trị & điều kiện \ giá trị & điều kiện
        # Ta tìm các dấu \ nằm giữa các cụm điều kiện.
        # Một dấu hiệu tốt là \ theo sau bởi một dấu trừ (cho -\infty) hoặc một con số.
        inner = re.sub(r'\s\\\s(-?\d+|-\\infty|\\infty)', r' \\\\ \1', inner)
        
        return f'\\begin{{cases}}{inner}\\end{{cases}}'

    content = re.sub(r'\\begin\{cases\}([\s\S]*?)\\end\{cases\}', internal_fix, content)
    
    return content

def run_fix(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    old_content = f.read()
                
                new_content = fix_cases_syntax(old_content)
                
                if new_content != old_content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Refined Cases: {path}")

if __name__ == "__main__":
    run_fix("/Users/pixibox/Aero-HowtoLLMs/docs")
