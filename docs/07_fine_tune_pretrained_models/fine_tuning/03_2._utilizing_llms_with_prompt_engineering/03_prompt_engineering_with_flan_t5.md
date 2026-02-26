
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
# Prompt Engineering Với FLAN-T5

## Giới Thiệu

Hãy nói về cách sử dụng mô hình đa năng này cho tóm tắt văn bản, dịch thuật và trả lời câu hỏi sử dụng thư viện Hugging Face Transformers và TensorFlow.

Hugging Face là một nền tảng lưu trữ một bộ sưu tập lớn các mô hình pre-trained, bao gồm FLAN-T5, có thể được điều chỉnh cho nhiều tác vụ dựa trên văn bản.

## Cài Đặt Môi Trường

Đầu tiên, chúng ta cần cài đặt môi trường của mình. Điều này bao gồm cài đặt các thư viện transformers và TensorFlow, cung cấp cơ sở hạ tầng và mô hình cần thiết cho các tác vụ của chúng ta.

## Tải FLAN-T5

Sau khi cài đặt, chúng ta sẽ tải FLAN-T5 sử dụng thư viện Transformers. Để làm điều đó, chúng ta sẽ sử dụng:
- **AutoTokenizer:** Xử lý văn bản thành định dạng mà mô hình có thể làm việc, chuyển đổi câu thành chuỗi tokens hoặc biểu diễn số.
- **TFAutoModelForSeq2SeqLM:** Mô hình sẽ diễn giải các tokens này và tạo văn bản dựa trên chúng.

## Tóm Tắt Văn Bản (Text Summarization)

Cho tóm tắt văn bản, chúng ta sẽ cho FLAN-T5 một đoạn văn bản và yêu cầu một bản tóm tắt ngắn gọn.

**Các bước thực hiện:**
1. Đặt prompt (ví dụ: "Summarize the following article about carrots")

2. Tokenize với `return_tensors="tf"` để xuất TensorFlow tensors

$$
3. Giới hạn độ dài với `max_length=512`
$$

