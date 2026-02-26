
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
# Học Tập Trong Ngữ Cảnh (In-Context Learning) và Mẫu Few-Shot với FLAN-T5

## Tổng Quan

Trong bài hướng dẫn thực hành này, chúng ta sẽ khám phá cách cải thiện các prompt cho mô hình FLAN-T5 thông qua kỹ thuật **học tập trong ngữ cảnh (In-Context Learning - ICL)** và mẫu **few-shot learning**. Đây là những kỹ thuật quan trọng trong kỹ thuật prompt hiện đại, cho phép mô hình ngôn ngữ lớn (LLM) thực hiện các tác vụ mới mà không cần huấn luyện lại.

## 1. Giới Thiệu về Few-Shot Learning

### 1.1 Khái Niệm Cơ Bản

Few-shot learning là một kỹ thuật trong đó chúng ta cung cấp cho mô hình một số ví dụ minh họa về tác vụ cần thực hiện. Các ví dụ này được đưa vào prompt để mô hình hiểu:

- Loại tác vụ cần thực hiện (ví dụ: tóm tắt, dịch thuật)
- Định dạng đầu ra mong muốn
- Độ dài và phong cách của kết quả

### 1.2 Cơ Chế Hoạt Động

Khi cung cấp các ví dụ few-shot, mô hình học được:

$$\text{Kết quả} = f(\text{ví dụ}_1, \text{ví dụ}_2, ..., \text{ví dụ}_n, \text{đầu vào mới})$$

Trong đó:
- $f$ là mô hình ngôn ngữ
- Các ví dụ cung cấp "ngữ cảnh" để mô hình suy luận
- Mô hình tổng quát hóa từ các ví dụ để xử lý đầu vào mới

## 2. Triển Khai Với FLAN-T5

### 2.1 Tóm Tắt Văn Bản (Summarization)

```python
# Ví dụ few-shot cho tác vụ tóm tắt
few_shot_examples = """
summarize: The quick brown fox jumps over the lazy dog. The dog was not amused by the fox's antics.
The fox jumped over the dog who was not happy.

summarize: Rain in Spain falls mainly on the plain. The weather has been unusual this year.
Weather patterns in Spain are interesting.

summarize: Carrots are rich in vitamin A and are excellent for eye health. They also contain fiber.
"""
```

Kết quả thu được:
- **Không có few-shot**: "eat carrots"
- **Có few-shot**: "Carrots are a great source of vitamin A, which is crucial for maintaining healthy eyesight"

### 2.2 Dịch Thuật (Translation)

```python
# Ví dụ few-shot cho tác vụ dịch Anh - Tây Ban Nha
translation_examples = """
translate English to Spanish: Hello, how are you?
Hola, ¿cómo estás?

translate English to Spanish: Good morning
Buenos días

translate English to Spanish: Cheese is delicious
"""
```

## 3. So Sánh Hiệu Quả

### 3.1 Đánh Giá Định Tính

| Phương pháp | Đầu ra | Chất lượng |
|-------------|---------|-------------|
| Zero-shot | "eat carrots" | Cơ bản |
| Few-shot | "Carrots are a great source of vitamin A..." | Tốt |

### 3.2 Phân Tích Toán Học

Hiệu quả của few-shot learning có thể được biểu diễn:

$$P(y|x, \text{ví dụ}) = \frac{1}{Z} \sum_{i=1}^{n} w_i \cdot \text{sim}(x, x_i) \cdot P(y|x_i)$$

Trong đó:
- $w_i$ là trọng số của ví dụ thứ $i$
- $\text{sim}$ là hàm đo độ tương đồng
- $Z$ là hằng số chuẩn hóa

## 4. Ứng Dụng Thực Tế

### 4.1 Trong Các Lĩnh Vực Khác Nhau

Kỹ thuật few-shot learning có thể áp dụng cho:

1. **Phân Tích Cảm Xúc (Sentiment Analysis)**
2. **Trả Lời Câu Hỏi (Question Answering)**
3. **Dịch Thuật (Translation)**
4. **Tóm Tắt (Summarization)**
5. **Chain-of-Thought Reasoning**

### 4.2 Ví Dụ Trả Lời Câu Hỏi

```python
# Few-shot cho QA
qa_examples = """
Context: The Great Wall of China is over 13,000 miles long.
Question: How long is the Great Wall of China?
Answer: Over 13,000 miles.

Context: Mount Everest is the highest mountain in the world.
Question: What is the highest mountain in the world?
Answer:
"""
```

## 5. Kết Luận

Kỹ thuật học tập trong ngữ cảnh và few-shot learning là những công cụ mạnh mẽ để cải thiện đáng kể hiệu suất của các mô hình ngôn ngữ lớn như FLAN-T5. Bằng cách cung cấp các ví dụ minh họa trong prompt, chúng ta có thể:

- Cải thiện chất lượng đầu ra
- Định dạng kết quả theo yêu cầu
- Giảm thiểu nhu cầu fine-tuning cho các tác vụ mới

## Tài Liệu Tham Khảo

1. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *Advances in Neural Information Processing Systems*, 33, 1877-1901.

2. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *arXiv:2201.11903*.

3. Dong, Q., et al. (2022). "A Survey on In-context Learning." *arXiv:2301.00234*.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
