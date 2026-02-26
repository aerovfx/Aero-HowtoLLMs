
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [20 investigating token embeddings](index.md)

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
# Phân Loại Từ Loại Bằng Thư Viện SpaCy Trong Phân Tích Mechanistic Interpretability

## Tóm tắt (Abstract)
Từ loại (Parts of speech - POS) như danh từ, động từ, tính từ là cột sống của cấu trúc ngôn ngữ. Việc mở rộng phân tích Mechanistic Interpretability lên các bộ dữ liệu quy mô lớn đòi hỏi phải tự động hóa quá trình nhận diện từ loại thay vì dán nhãn thủ công tĩnh. Báo cáo này giới thiệu việc ứng dụng thư viện Xử lý Ngôn ngữ Tự nhiên `spaCy` để dán nhãn POS. Tuy nhiên, một xung đột kỹ thuật lớn nảy sinh giữa thuật toán mã hóa từ phụ (Sub-word Byte-Pair Encoding) của Tokenizer trong LLMs (như GPT-2) và bộ Tokenizer tiêu chuẩn của `spaCy`, dẫn đến việc nhận diện sai không gian trắng (spaces) và chia cắt từ. Tài liệu này thảo luận nguyên nhân lõi và bước đệm xử lý cơ bản trước khi đi vào giải pháp định lượng sâu hơn.

---

## 1. Mở Đầu (Introduction)
Trong các khảo sát trước đây (như phân tích từ `"her"`, `"him"`, `"round"`), ta chủ yếu xây dựng bộ dữ liệu nhỏ và nhặt tay từng vị trí Token. Để thực sự khảo nghiệm cách một LLM như GPT-2 phản ứng với danh từ so với tính từ trên hàng ngàn cụm văn bản biểu đạt tự do, ta cần một bộ máy gán nhãn tự động.
`spaCy` là một thư viện Python mạnh mẽ, tối ưu hóa cho công việc bóc tách thông tin POS. Tuy nhiên, vì kiến trúc cắt chữ của LLMs không tuân theo quy tắc ngôn ngữ học truyền thống, việc ép chéo thư viện POS dán nhãn lên các Model Tokens thô phác sinh ra nhiều hệ lụy kỹ thuật cản trở phân tích.

---

## 2. Tiền Xử Lý: Ứng Dụng spaCy Cơ Bản (Methodology)

### 2.1. Ngữ Cảnh Quyết Định Từ Loại (Context-Dependent POS)
Một lỗi sơ đẳng khi ứng dụng `spaCy` là chia nhỏ câu thành các từ rời rạc (`split()`) và chạy gán nhãn đơn lẻ độc lập.
Ví dụ: Từ `explore` nếu đứng một mình có thể bị `spaCy` ném vào nhãn Danh từ (Noun). Nhưng khi đặt nguyên vẹn trong câu *"a sentence that I will use to **explore**"*, bộ máy phân tích cú pháp của `spaCy` sẽ nhận thức được cấu trúc động từ nguyên thể có `to` và gán chuẩn xác đây là Động từ (Verb). Do đó, nguyên tắc tối thượng là: **Phải đưa toàn bộ nguyên câu (Full Context) vào bộ phận NLP Object của `spaCy` phân tích một lần.**

### 2.2. Nghịch Lý Giữa LLM Tokenizer và rào cản Từ Phụ (Subwords)
Sự kết nối giữa LLMs và `spaCy` thất bại khi đối diện thuật toán BPE (Byte-Pair Encoding) của GPT.
- **Vấn đề dấu cách:** GPT-2 ghim chặt khoảng trắng (space character, thường ký hiệu là đằng trước chữ cái) vào trong Token. Khi ném thẳng token như `[ Ġsentence]` vào `spaCy`, hệ thống POS lập tức đánh giá ký tự đầu là dấu ngắt khoảng trắng và trả về kết quả rác (Space) thay vì (Noun). Giải pháp sơ cứu cục bộ là dùng chuỗi hàm `.strip()` gọt bỏ khoảng trắng thừa.
- **Vấn đề phân mảnh từ vựng:** GPT-2 cắt những chữ lạ thành nhiều token con. Ví dụ: từ `spacy` biến thành 2 tokens: `spa` và `cy`. Khi cắt đứt như vậy, `spaCy` mất hoàn toàn khái niệm từ gốc để dán nhãn.

---

## 3. Khảo Sát Đánh Giá: Thử Nghiệm Kết Trích Văn Bản (Analysis) 
Để trình diễn khả năng ứng dụng, thuật toán sử dụng bộ khởi tạo GPT-2 Small sinh ra 400 Token ngẫu nhiên tiếp nối câu mồi: *"I think the world could be better if"*.
Kết quả được chạy qua vòng lặp quét của `spaCy` Tokenizer:
```python

if token.pos_ == 'NOUN': count_noun += 1

$$
if token.pos_ == 'VERB': count_verb += 1
$$

