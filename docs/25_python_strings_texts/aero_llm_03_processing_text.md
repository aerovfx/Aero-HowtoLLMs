
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [25 python strings texts](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Nhập môn Python: Kỹ thuật Xử lý và Phân tích Văn bản (Processing Text)

## Tóm tắt (Abstract)

Báo cáo này nghiên cứu các kỹ thuật xử lý văn bản (text processing) trong Python — bước tiền xử lý (preprocessing) thiết yếu trước khi đưa dữ liệu ngôn ngữ vào các mô hình Transformer và LLM. Chúng ta phân tích cơ chế phân tách chuỗi bằng `.split()`, kỹ thuật làm sạch văn bản (text cleaning) với các phương thức chuỗi tích hợp như `.strip()`, `.lower()`, `.replace()`, và cách tận dụng thư viện chuẩn `string` để phân loại ký tự. Nghiên cứu cũng trình bày việc xây dựng quy trình tiền xử lý hoàn chỉnh: từ văn bản thô (raw text) đến danh sách token sạch (clean token list) — nền tảng cho mọi pipeline NLP hiện đại.

---

## 1. Phân tách Văn bản — Phương thức `.split()`

### 1.1. Cơ chế hoạt động

Python xem toàn bộ câu văn là **một chuỗi ký tự liên tục duy nhất**. Để làm việc với từng từ riêng biệt, cần phân tách bằng `.split()`:

```python

$$
sentence = "The quick brown fox jumps" words = sentence.split() print(words) # ['The', 'quick', 'brown', 'fox', 'jumps'] print(len(words))   # 5 từ - **Mặc định:** Phân tách tại khoảng trắng (space, tab, newline). - **Tùy chỉnh delimiter:** Truyền ký tự phân cách vào `.split(delimiter)`. ```python csv_line = "Hanoi,Saigon,Danang"
$$

cities = csv_line.split(',')   # ['Hanoi', 'Saigon', 'Danang']

### 1.2. Thống kê Từng Đơn vị Từ

Sau khi phân tách, có thể phân tích từng phần tử bằng vòng lặp `for`:

```python

$$
sentence = "Deep learning transforms language understanding"
$$

for word in sentence.split():
    print(f"'{word}' — {len(word)} ký tự")

Đây là bước khởi đầu cho các tác vụ như:
- Thống kê tần suất từ (word frequency)
- Xây dựng từ vựng (vocabulary building)
- Phân tích phân phối độ dài token

---

## 2. Thư viện `string` — Bộ Hằng số Ký tự Chuẩn

Thư viện `string` (tích hợp sẵn, không cần `pip install`) cung cấp các tập hợp ký tự tiêu chuẩn:

```python
import string

print(string.ascii_lowercase)  # 'abcdefghijklmnopqrstuvwxyz'
print(string.ascii_uppercase)  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
print(string.ascii_letters)    # tất cả chữ cái hoa + thường
print(string.digits)           # '0123456789'

$$
print(string.punctuation)      # '!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
$$

print(string.whitespace)       # space, tab (\t), newline (\n), \r, \f, \v

**Ưu điểm:** Không cần tự định nghĩa thủ công — tránh sai sót và tiết kiệm code.

---

## 3. Kiểm tra Loại Ký tự — Toán tử `in`

Kết hợp `in` với các hằng số của `string` để phân loại ký tự:

```python
import string

$$
char = 'A'
$$

if char in string.ascii_letters:
    print("Là chữ cái")
if char.lower() in string.ascii_lowercase:
    print("Là chữ cái (không phân biệt hoa/thường)")
if char in string.digits:
    print("Là chữ số")
if char in string.punctuation:
    print("Là dấu câu")

**Ứng dụng thực tế — Lọc dấu câu:**

```python

$$
text = "Hello, world! This is NLP." clean = ''.join(c for c in text if c not in string.punctuation)
$$

print(clean)   # 'Hello world This is NLP'

---

## 4. Làm sạch Văn bản — Các Phương thức Chuỗi Quan trọng

### 4.1. Chuẩn hóa chữ hoa/thường

```python

$$
text = "Deep Learning"
$$

print(text.lower())   # 'deep learning'
print(text.upper())   # 'DEEP LEARNING'

Cần thiết vì tokenizer phân biệt 'Apple' ≠ 'apple'. Chuyển về chữ thường để thống nhất.

### 4.2. Loại bỏ khoảng trắng thừa

```python

$$
raw = "  Hello World  \n"
$$

print(raw.strip())     # 'Hello World'  — cắt 2 đầu
print(raw.lstrip())    # 'Hello World  \n'  — chỉ cắt trái
print(raw.rstrip())    # '  Hello World'  — chỉ cắt phải

### 4.3. Thay thế chuỗi con

```python

text = "I love deep_learning and NLP!"

$$
cleaned = text.replace("_", " ").replace("!", "") print(cleaned)   # 'I love deep learning and NLP' ### 4.4. Kiểm tra nội dung chuỗi ```python
$$

word = "Hello123"

$$
print(word.isalpha())    # False — có chứa số print(word.isdigit())    # False — không phải toàn số print(word.isalnum())    # True — chữ cái + số print("  ".isspace())    # True — toàn khoảng trắng --- ## 5. Ký tự Điều khiển (Control Characters) `string.whitespace` bao gồm các ký tự **"vô hình"** thường gây lỗi khi xử lý dữ liệu: | Ký tự | Tên | Ý nghĩa | |-------|-----|---------| | `' '` | Space | Khoảng trắng thông thường | | `'\t'` | Tab | Khoảng cách ngang | | `'\n'` | Newline | Xuống dòng | | `'\r'` | Carriage Return | Về đầu dòng (Windows) | | `'\f'` | Form Feed | Sang trang | | `'\v'` | Vertical Tab | Tab dọc | **Xử lý newline trong văn bản nhiều dòng:** ```python
$$

multiline = "Line 1\nLine 2\nLine 3"

lines = multiline.split('\n')   # ['Line 1', 'Line 2', 'Line 3']

---

## 6. Quy trình Tiền xử lý Hoàn chỉnh

Kết hợp tất cả kỹ thuật trên để xây dựng pipeline:

```python
import string

def preprocess_text(text):
    """Làm sạch văn bản cho NLP pipeline."""
    # 1. Chuyển về chữ thường

text = text.lower()

    # 2. Loại bỏ khoảng trắng đầu cuối

text = text.strip()

    # 3. Tách từ

words = text.split()

    # 4. Loại bỏ dấu câu khỏi từng từ

$$
words = [w.strip(string.punctuation) for w in words]
$$

    # 5. Loại bỏ token rỗng

$$
words = [w for w in words if w]
$$

    return words

$$
raw = "  Hello, World! This is Deep Learning.  " tokens = preprocess_text(raw)
$$

