
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../index.md) > [07 fine tune pretrained models](../../index.md) > [fine tuning](../index.md) > [03 2. utilizing llms with prompt engineering](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../../index.md)
- [📚 Module 01: LLM Course](../../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Demo Prompt Engineering Với FLAN-T5

## Giới Thiệu

Chào mừng mọi người đến với demo đầu tiên của khóa học này. Tất cả các demo trong khóa học này sẽ sử dụng Google Colaboratory.

Google Colab là một nền tảng cho phép chúng ta lưu trữ các file notebook và kết nối miễn phí đến một instance trên Google Cloud Platform nơi chúng ta cũng có thể kết nối GPU. Điều này rất hữu ích, đặc biệt cho việc prototype các ý tưởng.

Truy cập: colab.research.google.com

## Thiết Lập Môi Trường

### Kết Nối Google Colab

1. Truy cập trang web Colab
2. Upload notebook từ Exercise Files
3. Click "Connect" để kết nối với GPU miễn phí

**Lưu ý:** Loại GPU phụ thuộc vào:
- Khả năng sẵn có theo múi giờ
- Tần suất sử dụng GPU gần đây
- Vì là miễn phí nên không đảm bảo được loại GPU cụ thể

## Cài Đặt Thư Viện

```python
# Cài đặt Transformers và TensorFlow
!pip install transformers tensorflow

## Tải Mô Hình FLAN-T5

```python
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Tải tokenizer và model

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

$$
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large") **Lưu ý về warnings:** - Warning về xác thực HuggingFace là bình thường - Warning về việc model được train bằng PyTorch rồi convert sang TensorFlow - độ chính xác 99.9% tương đương ## Quy Trình Prompt Với FLAN-T5 Việc prompt một LLM luôn gồm 4 bước: 1. Định nghĩa prompt 2. Tokenize (chuyển đổi văn bản thành tokens) 3. Model.generate() (tạo output) 4. Tokenizer.decode() (chuyển đổi IDs về văn bản) ### 1. Tóm Tắt Văn Bản (Summarization) ```python # Định nghĩa prompt
$$

prompt = "summarize: Studies show that eating carrots help improve vision..."

$$
# Tokenize inputs = tokenizer(prompt, return_tensors="tf", max_length=512,
$$

truncation=True, padding=True)

# Generate

outputs = model.generate(inputs.input_ids, max_length=50)

# Decode

summary = tokenizer.decode(outputs[0])

print(summary)

**Kết quả:** "eat carrots" - một bản tóm tắt ngắn gọn

### 2. Dịch Thuật (Translation)

```python
# Prompt dịch tiếng Anh sang tiếng Tây Ban Nha

$$
prompt = "translate English to Spanish: cheese is delicious"
$$

# Tokenize

inputs = tokenizer(prompt, return_tensors="tf", max_length=512,

$$
truncation=True, padding=True) # Generate outputs = model.generate(inputs.input_ids, max_length=40) # Decode translation = tokenizer.decode(outputs[0]) print(translation) ### 3. Trả Lời Câu Hỏi (Question Answering) ```python # Context và câu hỏi
$$

context = "The Great Wall of China is over 13,000 miles long."

question = "question: How long is the Great Wall of China?"

prompt = context + " " + question

# Tokenize

inputs = tokenizer(prompt, return_tensors="tf", max_length=512,

$$
truncation=True, padding=True) # Generate outputs = model.generate(inputs.input_ids, max_length=50) # Decode answer = tokenizer.decode(outputs[0])
$$

