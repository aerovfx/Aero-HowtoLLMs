
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [20 investigating token embeddings](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../20_python_colab_notebooks/index.md)
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
if token.pos_ == 'VERB': count_verb += 1
```
Kết quả trả ra hoạt động khá tròn nhiệm vụ, gom nhặt được danh sách dài các danh từ và động từ. Tuy nhiên vẫn tồn tại nhiễu do tách khoảng trắng/tách từ. Biện pháp an toàn nhất để tránh sai lầm rác tác động lên phương sai Toán học (Statistics Noise) là: **Chỉ dung nạp và thống kê những Tokens đại diện trọn vẹn cho đúng một Từ nguyên vẹn (Full Words) trong bộ từ điển của Mô hình**, và sàng lọc vứt bỏ các Token vỡ vụn hoặc độ tự tin thấp.

---

## 4. Kết Luận
Việc dùng bộ dán nhãn từ loại song song như `spaCy` là cánh cửa bắt buộc để tự động hóa Mechanistic Interpretability. Mặc dù sự xung khắc giữa hai hệ Tokenizer (BPE vs Standard Linguistic) gây ra rườm rà trong việc làm sạch dữ liệu, nhưng chỉ cần ta chuẩn bị tập Text Database đủ lớn, việc thanh lọc mạnh tay (chỉ giữ Full Word Tokens hợp chuẩn) hoàn toàn đủ khả năng cung cấp một bộ mẫu dung lượng khổng lồ. Cách làm này sẽ được triển khai thực nghiệm đo lường góc và phương sai trong chặng phân tích so sánh Danh Từ - Tính Từ tiếp theo.

---

## Tài Liệu Tham Khảo (Citations)
1. Lý thuyết ứng dụng cài đặt NLP object bằng `spacy.load()` và cơ chế gán `.pos_` của phương thức `spaCy` Token dựa trên source code `aero_LLM_07_Parts of speech with SpaCy library.md`.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Phân Khảo Token Embeddings: Đo Lường Góc Quay Của Vector Biểu Diễn](aero_llm_01_calculating_rotations_of_embeddings_vectors.md) | [Xem bài viết →](aero_llm_01_calculating_rotations_of_embeddings_vectors.md) |
| [Thử Thách Lập Trình (Code Challenge): Tiến Hóa Đa Tầng Của Các Điều Chỉnh Góc Quay Tuần Tự](aero_llm_02_codechallenge_laminar_evolution_of_sequential_angular_adjustments.md) | [Xem bài viết →](aero_llm_02_codechallenge_laminar_evolution_of_sequential_angular_adjustments.md) |
| [Đo Lường Độ Dài Đường Dẫn (Path Length) Sự Tương Quan Với Dự Đoán Token](aero_llm_03_path_length_and_logit_token_prediction.md) | [Xem bài viết →](aero_llm_03_path_length_and_logit_token_prediction.md) |
| [Thử Thách Lập Trình: Phân Rã Độ Dài Đường Dẫn Luồng Số Dư (Phần 1)](aero_llm_04_codechallenge_residual_stream_path_length_decomposition_part_1_.md) | [Xem bài viết →](aero_llm_04_codechallenge_residual_stream_path_length_decomposition_part_1_.md) |
| [Thử Thách Lập Trình: Phân Rã Độ Dài Đường Dẫn Luồng Số Dư (Phần 2)](aero_llm_05_codechallenge_residual_stream_path_length_decomposition_part_2_.md) | [Xem bài viết →](aero_llm_05_codechallenge_residual_stream_path_length_decomposition_part_2_.md) |
| [Quỹ Đạo Không Gian Trạng Thái (State-Space Trajectories) Của Hệ Vector Ngôn Ngữ](aero_llm_06_state_space_trajectories_through_embedding_space.md) | [Xem bài viết →](aero_llm_06_state_space_trajectories_through_embedding_space.md) |
| 📌 **[Phân Loại Từ Loại Bằng Thư Viện SpaCy Trong Phân Tích Mechanistic Interpretability](aero_llm_07_parts_of_speech_with_spacy_library.md)** | [Xem bài viết →](aero_llm_07_parts_of_speech_with_spacy_library.md) |
| [Thử Thách Lập Trình: So Sánh Độ Dài Quỹ Đạo Của Danh Từ Và Tính Từ (Phần 1)](aero_llm_08_codechallenge_do_nouns_or_adjectives_have_longer_trajectories_part_1_.md) | [Xem bài viết →](aero_llm_08_codechallenge_do_nouns_or_adjectives_have_longer_trajectories_part_1_.md) |
| [Thử Thách Lập Trình: So Sánh Độ Dài Quỹ Đạo Của Danh Từ Và Tính Từ (Phần 2)](aero_llm_09_codechallenge_do_nouns_or_adjectives_have_longer_trajectories_part_2_.md) | [Xem bài viết →](aero_llm_09_codechallenge_do_nouns_or_adjectives_have_longer_trajectories_part_2_.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
