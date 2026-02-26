
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
# Phân tích quy trình Tokenization qua ví dụ *The Time Machine*: Cơ sở thuật toán và mô hình hóa toán học

---

## Tóm tắt

Bài viết này trình bày một phân tích khoa học về quy trình tokenization thông qua ví dụ văn bản *The Time Machine* của H. G. Wells. Nội dung tập trung vào cách tiền xử lý văn bản, xây dựng từ vựng, ánh xạ token sang chỉ số và chuyển đổi sang không gian vector phục vụ huấn luyện mô hình Transformer. Bài viết bổ sung các nền tảng lý thuyết từ kiến trúc Attention Is All You Need và các mô hình GPT do OpenAI phát triển, kèm theo các công thức toán học minh họa cho quá trình rời rạc hóa và biểu diễn liên tục.

---

# 1. Giới thiệu

Trong các mô hình ngôn ngữ hiện đại, dữ liệu văn bản phải được chuyển đổi từ dạng ký tự sang dạng token trước khi đưa vào mạng nơ-ron.

Cho văn bản đầu vào:

$$
X = (c_1, c_2, \dots, c_n), \quad c_i \in \Sigma
$$

Tokenization thực hiện ánh xạ:

$$
\tau : \Sigma^* \rightarrow V^*
$$

trong đó:

* $\Sigma$: tập ký tự
* $V$: từ vựng token
* $V^*$: chuỗi token

Ví dụ với *The Time Machine*:

The Time Machine by H. G. Wells

Sau xử lý có thể thành:

$$

$$

\text{"the"}, \text{"time"}, \text{"machine"}, \text{"by"}, \text{"h"}, \text{"g"}, \text{"wells"}

$$

$$

---

# 2. Tiền xử lý văn bản

## 2.1 Chuẩn hóa chữ thường

$$
f_{lower}(x) = \text{lower}(x)
$$

Giúp giảm kích thước từ vựng:

$$
|V_{raw}| > |V_{normalized}|
$$

---

## 2.2 Loại bỏ ký tự đặc biệt

Hàm lọc:

$$
f_{clean}(x) = x \setminus { \text{punctuation} }
$$

Mục tiêu:

* Giảm nhiễu
* Chuẩn hóa cấu trúc

---

# 3. Tokenization mức từ (Word-level Tokenization)

Sau khi tách theo khoảng trắng:

$$
X = (w_1, w_2, \dots, w_T)
$$

Số lượng token:

$$
T \leq n
$$

Tần suất xuất hiện của từ $w$:

$$
f(w) = \sum_{i=1}^{T} \mathbf{1}(w_i = w)
$$

---

# 4. Xây dựng từ vựng (Vocabulary Construction)

Tập từ vựng:

$$
V = { w \mid f(w) \geq \delta }
$$

với $\delta$ là ngưỡng tối thiểu.

Kích thước từ vựng:

$$
|V| = M
$$

Ánh xạ:

$$
w \rightarrow id(w) \in {0, 1, \dots, M-1}
$$

---

# 5. Biểu diễn One-Hot

Token $w_i$ được biểu diễn:

$$
x_i \in \mathbb{R}^{M}
$$

với:

$$
x_{ij} = \begin{cases} 1 & \text{nếu } j = id(w_i) \ 0 & \text{ngược lại} \end{cases}
$$

Nhược điểm:

* Kích thước lớn
* Không phản ánh ngữ nghĩa

---

# 6. Embedding Vector

Embedding matrix:

$$
E \in \mathbb{R}^{M \times d}
$$

Vector embedding:

$$
e_i = E^T x_i
$$

Do đó:

$$
e_i \in \mathbb{R}^{d}
$$

Khoảng cách cosine:

$$
\cos(e_i, e_j) = \frac{e_i \cdot e_j}{|e_i||e_j|}
$$

Giúp đo mức độ tương đồng ngữ nghĩa.

---

# 7. Mô hình hóa xác suất ngôn ngữ

Theo mô hình tự hồi quy:

$$
P(X) = \prod_{t=1}^{T} P(w_t \mid w_{<t})
$$

Mạng Transformer tính:

$$
Z = \text{Transformer}(e_1, \dots, e_T)
$$

Logits:

$$
z_t = W_{out} h_t
$$

Softmax:

$$
P(w_t = j \mid w_{<t}) = \frac{\exp(z_{tj})} {\sum_{k=1}^{M} \exp(z_{tk})}
$$

---

# 8. Độ phức tạp tính toán

Self-attention:

$$
\mathcal{O}(T^2 d)
$$

Nếu văn bản dài như *The Time Machine* (~30,000 từ), chi phí tăng theo bình phương độ dài chuỗi.

Do đó, tokenization tối ưu giúp:

* Giảm $T$
* Giảm bộ nhớ
* Tăng tốc huấn luyện

---

# 9. Phân tích thống kê văn bản

Entropy của tập từ:

$$
H(W) = - \sum_{w \in V} P(w) \log P(w)
$$

Với:

$$
P(w) = \frac{f(w)}{T}
$$

Nếu phân bố Zipf:

$$
f(w_r) \propto \frac{1}{r}
$$

trong đó $r$ là thứ hạng tần suất.

Điều này cho thấy:

* Số ít từ xuất hiện rất nhiều
* Nhiều từ hiếm xuất hiện

---

# 10. So sánh với Subword Tokenization

Word-level tokenization có nhược điểm:

$$
P(\text{OOV}) > 0
$$

Giải pháp: Byte Pair Encoding (BPE).

Tập hợp phân rã:

$$
w = s_1 s_2 \dots s_k
$$

với $s_i \in V_{subword}$

Đảm bảo:

$$
\forall w, \exists \text{ decomposition}
$$

---

# 11. Thảo luận

Tokenization là quá trình:

$$
\text{Text} \rightarrow \text{Discrete Representation} \rightarrow \text{Continuous Geometry}
$$

Về mặt toán học:

* Là ánh xạ từ chuỗi ký tự sang không gian vector
* Là bước nén thông tin
* Ảnh hưởng trực tiếp đến phân phối xác suất

---

# 12. Kết luận

Thông qua ví dụ *The Time Machine*, ta thấy:

1. Tokenization quyết định cấu trúc dữ liệu đầu vào
2. Vocabulary ảnh hưởng đến kích thước embedding
3. Biểu diễn vector quyết định khả năng học ngữ nghĩa
4. Độ dài chuỗi ảnh hưởng đến độ phức tạp Transformer

Toàn bộ quá trình có thể được mô hình hóa:

$$
\Sigma^* \xrightarrow{\tau} V^* \xrightarrow{E} \mathbb{R}^{T \times d}
$$

đóng vai trò nền tảng cho mọi mô hình Transformer hiện đại.

---

# Tài liệu tham khảo

1. Attention Is All You Need
2. Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*.
3. Shannon, C. (1948). *A Mathematical Theory of Communication*.
4. Jurafsky, D., Martin, J. (2023). *Speech and Language Processing*.
5. Manning, C., Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Tại sao văn bản cần được đánh số?](aero_llm_01_why_text_needs_to_be_numbered.md) | [Xem bài viết →](aero_llm_01_why_text_needs_to_be_numbered.md) |
| [Phân tích và chuyển đổi văn bản thành chuỗi token được đánh số: Cơ sở toán học và ứng dụng trong mô hình ngôn ngữ lớn](aero_llm_02_parsing_text_to_numbered_tokens.md) | [Xem bài viết →](aero_llm_02_parsing_text_to_numbered_tokens.md) |
| [Tạo và trực quan hóa Token trong mô hình ngôn ngữ lớn: Cơ sở toán học và phân tích biểu diễn](aero_llm_03_codechallenge_create_and_visualize_tokens_part_1_.md) | [Xem bài viết →](aero_llm_03_codechallenge_create_and_visualize_tokens_part_1_.md) |
| [Tạo và trực quan hóa Token (Phần 2): Phân tích hình học không gian embedding và Attention Map trong mô hình Transformer](aero_llm_04_codechallenge_create_and_visualize_tokens_part_2_.md) | [Xem bài viết →](aero_llm_04_codechallenge_create_and_visualize_tokens_part_2_.md) |
| [Chuẩn bị văn bản cho Tokenization trong mô hình Transformer: Cơ sở lý thuyết và phân tích toán học](aero_llm_05_preparing_text_for_tokenization.md) | [Xem bài viết →](aero_llm_05_preparing_text_for_tokenization.md) |
| 📌 **[Phân tích quy trình Tokenization qua ví dụ *The Time Machine*: Cơ sở thuật toán và mô hình hóa toán học](aero_llm_06_codechallenge_tokenizing_the_time_machine.md)** | [Xem bài viết →](aero_llm_06_codechallenge_tokenizing_the_time_machine.md) |
| [So sánh Tokenization mức ký tự, từ và subword: Phân tích lý thuyết và mô hình toán học](aero_llm_07_tokenizing_characters_vs_subwords_vs_words.md) | [Xem bài viết →](aero_llm_07_tokenizing_characters_vs_subwords_vs_words.md) |
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
