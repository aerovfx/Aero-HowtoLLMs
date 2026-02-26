
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../../index.md) > [07 Fine tune pretrained models](../../../index.md) > [Fine Tuning](../../index.md) > [03   2. Utilizing LLMs with Prompt Engineering](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../../../index.md)
- [📚 Module 01: LLM Course](../../../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../../../20-Python-Colab-notebooks/index.md)
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
3. Giới hạn độ dài với `max_length=512`
4. Sử dụng `model.generate()` để tạo đầu ra
5. Decode kết quả với tokenizer

**Tham số quan trọng:**
- `num_beams`: Kiểm soát beam search
- `early_stopping`: Dừng tạo khi có câu trả lời hài lòng
- `max_length`: Giới hạn số tokens

## Dịch Thuật (Translation)

Cho dịch thuật từ tiếng Anh sang tiếng Pháp:
- Prompt: "translate English to French: [văn bản cần dịch]"
- Tiếp tục với các bước tokenize, generate, và decode

## Trả Lời Câu Hỏi (Question Answering)

Cho trả lời câu hỏi:
- Cung cấp ngữ cảnh: "The Great Wall of China is over 13,000 miles long."
- Đặt câu hỏi: "question: How long is the Great Wall of China?"

**Tham số quan trọng:**
- `num_beams`: Kiểm soát beam search algorithm
- `early_stopping`: Quan trọng trong Q&A, dừng tạo khi có câu trả lời hài lòng

## Kết Luận

Chúng ta đã thấy cách áp dụng FLAN-T5 cho ba tác vụ khác nhau, chứng minh tính linh hoạt và sức mạnh của mô hình. Bằng cách hiểu cách tạo prompts hiệu quả và cấu hình các tham số mô hình, bạn có thể nâng cao khả năng của các ứng dụng của mình, làm cho chúng thông minh và phản hồi nhanh hơn.

## Tài Liệu Tham Khảo

1. Liu, P., et al. (2023). "Prefix Tuning vs. Prompt Tuning: A Comparative Study." *arXiv:2303.13402*.

2. Reynolds, L., & McDonell, K. (2021). "Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm." *arXiv:2102.07350*.

3. Stiennon, N., et al. (2020). "Learning to Summarize with Human Feedback." *Advances in Neural Information Processing Systems*, 33, 3008-3021.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
