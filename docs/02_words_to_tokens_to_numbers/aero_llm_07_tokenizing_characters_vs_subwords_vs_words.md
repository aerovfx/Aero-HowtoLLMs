
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [02 words to tokens to numbers](index.md)

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
# So sánh Tokenization mức ký tự, từ và subword: Phân tích lý thuyết và mô hình toán học

---

## Tóm tắt

Bài báo này phân tích và so sánh ba chiến lược tokenization phổ biến trong xử lý ngôn ngữ tự nhiên: **character-level**, **word-level** và **subword-level**. Dựa trên nền tảng kiến trúc Attention Is All You Need và các mô hình ngôn ngữ lớn như GPT-4 do OpenAI phát triển, bài viết mô hình hóa toán học sự khác biệt giữa các phương pháp, phân tích entropy, độ phức tạp tính toán và tác động đến self-attention.

---

# 1. Giới thiệu

Tokenization là quá trình ánh xạ:

$$
\tau: \Sigma^* \rightarrow V^*
$$

trong đó:

* (\Sigma): tập ký tự
* (V): tập token
* (\Sigma^*): chuỗi ký tự
* (V^*): chuỗi token

Ba chiến lược chính:

1. Character-level
2. Word-level
3. Subword-level (BPE, Unigram LM)

Mỗi phương pháp tạo ra độ dài chuỗi (T) và kích thước từ vựng (|V|) khác nhau.

---

# 2. Tokenization mức ký tự (Character-Level)

## 2.1 Định nghĩa

Mỗi token là một ký tự:

$$
V = \Sigma
$$

Chuỗi:

$$
X = (c_1, c_2, \dots, c_n)
$$

Số token:

$$
T = n
$$

---

## 2.2 Ưu điểm

* Không có OOV:

$$
\forall x \in \Sigma^*, \tau(x) \text{ luôn tồn tại}
$$

* Kích thước từ vựng nhỏ:

$$
|V| \approx 100 - 500
$$

---

## 2.3 Nhược điểm

Self-attention có độ phức tạp:

$$
\mathcal{O}(T^2 d)
$$

Vì (T = n) lớn → chi phí tăng mạnh.

Ví dụ: văn bản 1000 ký tự

$$
T_{char} = 1000
$$

Chi phí attention:

$$
\propto 1000^2 = 10^6
$$

---

# 3. Tokenization mức từ (Word-Level)

## 3.1 Định nghĩa

Chuỗi:

$$
X = (w_1, w_2, \dots, w_m)
$$

với:

$$
m < n
$$

Tập từ vựng:

$$
V = { w }
$$

---

## 3.2 Đặc điểm thống kê

Phân bố tần suất từ tuân theo định luật Zipf:

$$
f(w_r) \propto \frac{1}{r}
$$

trong đó (r) là thứ hạng.

Entropy:

$$
H(W) = -\sum_{w} P(w)\log P(w)
$$

---

## 3.3 Nhược điểm

Xác suất OOV:

$$
P(\text{OOV}) = 1 - \sum_{w \in V} P(w)
$$

Vì từ vựng hữu hạn.

Kích thước từ vựng lớn:

$$
|V| \approx 30,000 - 200,000
$$

Embedding matrix:

$$
E \in \mathbb{R}^{|V| \times d}
$$

→ tiêu tốn bộ nhớ.

---

# 4. Tokenization mức Subword

Subword kết hợp ưu điểm của hai phương pháp trên.

## 4.1 Byte Pair Encoding (BPE)

BPE lặp lại:

$$
(a^*, b^*) = \arg\max_{a,b} f(a,b)
$$

Cập nhật từ vựng:

$$
V_{k+1} = V_k \cup {ab}
$$

---

## 4.2 Unigram Language Model

Tối ưu:

$$
\max_{\theta} \prod_i \sum_{z \in \mathcal{Z}(x_i)} P(z|\theta)
$$

Trong đó:

* (z): một phân tách hợp lệ
* (\mathcal{Z}(x_i)): tập các phân tách

---

## 4.3 Độ dài chuỗi trung bình

Giả sử:

* Character-level: (T_c = n)
* Word-level: (T_w = m)
* Subword-level: (T_s)

Thông thường:

$$
m < T_s < n
$$

Do đó:

$$
T_s^2 < T_c^2
$$

và

$$
|V_s| < |V_w|
$$

---

# 5. So sánh độ phức tạp

| Phương pháp | Độ dài (T) | Từ vựng (|V|) | OOV | Chi phí attention |
|-------------|--------------|-----------------|------|-------------------|
| Character | Lớn | Nhỏ | Không | Rất cao |
| Word | Nhỏ | Rất lớn | Có | Thấp |
| Subword | Trung bình | Trung bình | Không | Trung bình |

Self-attention:

$$
\text{Cost} = \mathcal{O}(T^2 d)
$$

Embedding memory:

$$
\mathcal{O}(|V| d)
$$

Subword tối ưu cân bằng hai yếu tố.

---

# 6. Phân tích thông tin

Theo định lý Shannon:

$$
H(X) = -\sum_x P(x)\log P(x)
$$

Chiều dài mã tối ưu:

$$
L \approx \frac{H(X)}{\log |V|}
$$

Subword giúp:

* Giảm chiều dài chuỗi
* Giảm entropy điều kiện

---

# 7. Ảnh hưởng đến Transformer

Mô hình Transformer tính:

$$
Z = \text{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Vì attention phụ thuộc (T):

* Character-level → khó mở rộng
* Word-level → vấn đề OOV
* Subword → cân bằng tối ưu

Các mô hình GPT hiện đại sử dụng biến thể byte-level BPE.

---

# 8. Thảo luận thực nghiệm

Trong thực tế:

* Character-level phù hợp cho dữ liệu nhiễu
* Word-level phù hợp cho corpora nhỏ
* Subword-level phù hợp cho LLM quy mô lớn

Giả sử chuỗi 1000 ký tự:

$$
T_c = 1000
$$
$$
T_s \approx 250 - 400
$$
$$
T_w \approx 150 - 250
$$

Chi phí attention giảm theo bình phương độ dài.

---

# 9. Kết luận

Tokenization có thể xem là bài toán tối ưu đa mục tiêu:

$$
\min_{V} \left( \alpha T^2 + \beta |V| \right)
$$

Trong đó:

* (T): độ dài chuỗi
* (|V|): kích thước từ vựng
* (\alpha, \beta): trọng số chi phí

Subword tokenization là nghiệm cân bằng gần tối ưu trong thực tế.

---

# Tài liệu tham khảo

1. Attention Is All You Need
2. Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*.
3. Kudo, T. (2018). *Subword Regularization*.
4. Shannon, C. (1948). *A Mathematical Theory of Communication*.
5. Jurafsky, D., Martin, J. (2023). *Speech and Language Processing*.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Tại sao văn bản cần được đánh số?](aero_llm_01_why_text_needs_to_be_numbered.md) | [Xem bài viết →](aero_llm_01_why_text_needs_to_be_numbered.md) |
| [Phân tích và chuyển đổi văn bản thành chuỗi token được đánh số: Cơ sở toán học và ứng dụng trong mô hình ngôn ngữ lớn](aero_llm_02_parsing_text_to_numbered_tokens.md) | [Xem bài viết →](aero_llm_02_parsing_text_to_numbered_tokens.md) |
| [Tạo và trực quan hóa Token trong mô hình ngôn ngữ lớn: Cơ sở toán học và phân tích biểu diễn](aero_llm_03_codechallenge_create_and_visualize_tokens_part_1_.md) | [Xem bài viết →](aero_llm_03_codechallenge_create_and_visualize_tokens_part_1_.md) |
| [Tạo và trực quan hóa Token (Phần 2): Phân tích hình học không gian embedding và Attention Map trong mô hình Transformer](aero_llm_04_codechallenge_create_and_visualize_tokens_part_2_.md) | [Xem bài viết →](aero_llm_04_codechallenge_create_and_visualize_tokens_part_2_.md) |
| [Chuẩn bị văn bản cho Tokenization trong mô hình Transformer: Cơ sở lý thuyết và phân tích toán học](aero_llm_05_preparing_text_for_tokenization.md) | [Xem bài viết →](aero_llm_05_preparing_text_for_tokenization.md) |
| [Phân tích quy trình Tokenization qua ví dụ *The Time Machine*: Cơ sở thuật toán và mô hình hóa toán học](aero_llm_06_codechallenge_tokenizing_the_time_machine.md) | [Xem bài viết →](aero_llm_06_codechallenge_tokenizing_the_time_machine.md) |
| 📌 **[So sánh Tokenization mức ký tự, từ và subword: Phân tích lý thuyết và mô hình toán học](aero_llm_07_tokenizing_characters_vs_subwords_vs_words.md)** | [Xem bài viết →](aero_llm_07_tokenizing_characters_vs_subwords_vs_words.md) |
| [aero llm 08 byte pair encoding algorithm](aero_llm_08_byte_pair_encoding_algorithm.md) | [Xem bài viết →](aero_llm_08_byte_pair_encoding_algorithm.md) |
| [Thuật toán Byte Pair Encoding (BPE) và Bài toán Tối ưu Kích thước Từ vựng trong Mô hình Ngôn ngữ](aero_llm_09_codechallenge_byte_pair_encoding_to_a_desired_vocab_size.md) | [Xem bài viết →](aero_llm_09_codechallenge_byte_pair_encoding_to_a_desired_vocab_size.md) |
| [aero llm 10 exploring chatgpt4 s tokenizer](aero_llm_10_exploring_chatgpt4_s_tokenizer.md) | [Xem bài viết →](aero_llm_10_exploring_chatgpt4_s_tokenizer.md) |
| [aero llm 11 codechallenge token count by subword length part 1](aero_llm_11_codechallenge_token_count_by_subword_length_part_1_.md) | [Xem bài viết →](aero_llm_11_codechallenge_token_count_by_subword_length_part_1_.md) |
| [aero llm 12 codechallenge token count by subword length part 2](aero_llm_12_codechallenge_token_count_by_subword_length_part_2_.md) | [Xem bài viết →](aero_llm_12_codechallenge_token_count_by_subword_length_part_2_.md) |
| [aero llm 13 how many rs in strawberry](aero_llm_13_how_many_rs_in_strawberry.md) | [Xem bài viết →](aero_llm_13_how_many_rs_in_strawberry.md) |
| [aero llm 14 codechallenge create your algorithmic rapper name](aero_llm_14_codechallenge_create_your_algorithmic_rapper_name_.md) | [Xem bài viết →](aero_llm_14_codechallenge_create_your_algorithmic_rapper_name_.md) |
| [aero llm 15 tokenization in bert](aero_llm_15_tokenization_in_bert.md) | [Xem bài viết →](aero_llm_15_tokenization_in_bert.md) |
| [aero llm 16 codechallenge character counts in bert tokens](aero_llm_16_codechallenge_character_counts_in_bert_tokens.md) | [Xem bài viết →](aero_llm_16_codechallenge_character_counts_in_bert_tokens.md) |
| [aero llm 17 translating between tokenizers](aero_llm_17_translating_between_tokenizers.md) | [Xem bài viết →](aero_llm_17_translating_between_tokenizers.md) |
| [aero llm 18 codechallenge more on token translation](aero_llm_18_codechallenge_more_on_token_translation.md) | [Xem bài viết →](aero_llm_18_codechallenge_more_on_token_translation.md) |
| [aero llm 19 codechallenge tokenization compression ratios](aero_llm_19_codechallenge_tokenization_compression_ratios.md) | [Xem bài viết →](aero_llm_19_codechallenge_tokenization_compression_ratios.md) |
| [aero llm 20 tokenization in different languages](aero_llm_20_tokenization_in_different_languages.md) | [Xem bài viết →](aero_llm_20_tokenization_in_different_languages.md) |
| [aero llm 21 codechallenge zipf s law in characters and tokens](aero_llm_21_codechallenge_zipf_s_law_in_characters_and_tokens.md) | [Xem bài viết →](aero_llm_21_codechallenge_zipf_s_law_in_characters_and_tokens.md) |
| [aero llm 22 word variations in claude tokenizer](aero_llm_22_word_variations_in_claude_tokenizer.md) | [Xem bài viết →](aero_llm_22_word_variations_in_claude_tokenizer.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
