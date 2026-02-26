
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
# Tạo và trực quan hóa Token trong mô hình ngôn ngữ lớn: Cơ sở toán học và phân tích biểu diễn

---

## Tóm tắt

Quá trình tạo (create) và trực quan hóa (visualize) token là bước trung gian quan trọng giữa văn bản thô và không gian vector trong các mô hình ngôn ngữ lớn (LLMs). Bài viết này phân tích cơ sở toán học của tokenization, embedding, và các kỹ thuật trực quan hóa không gian đặc trưng (feature space visualization) như PCA và t-SNE. Phân tích dựa trên kiến trúc Transformer của Vaswani et al. (2017) và các mô hình GPT do OpenAI phát triển.

---

# 1. Giới thiệu

Mô hình ngôn ngữ không xử lý văn bản trực tiếp mà xử lý:
$$
\text{Text} \rightarrow \text{Token IDs} \rightarrow \text{Embedding vectors}
$$


Việc trực quan hóa token giúp:

* Hiểu cấu trúc không gian embedding
* Phân tích quan hệ ngữ nghĩa
* Kiểm tra tính chất học được của mô hình

---

# 2. Tạo Token (Token Creation)

## 2.1. Tokenization

Cho văn bản $x$, hàm tokenization:
$$
T: \mathcal{X} \rightarrow V^T
$$


Trong đó:

* $V$: từ vựng có kích thước $|V| = N$
* (T(x) = (t_1, t_2, ..., t_T))

Mỗi token $t_i \in {1,2,...,N}$

---

## 2.2. Embedding

Ma trận embedding:
$$
E \in \mathbb{R}^{N \times d}
$$


Vector của token thứ $i$:
$$
e_i = E[t_i]
$$


Chuỗi đầu vào:
$$
Z = (e_1, e_2, ..., e_T)
$$


---

# 3. Thêm thông tin vị trí

Transformer không có RNN hay CNN nên cần positional encoding:
$$
z_i = e_i + p_i
$$


Trong GPT:
$$
p_i \in \mathbb{R}^d
$$


được học trực tiếp.

---

# 4. Trực quan hóa không gian token

Embedding có chiều cao $ví dụ ( d = 768, 1024, 1280$).
Để trực quan hóa, ta cần giảm chiều.

---

## 4.1. Principal Component Analysis (PCA)

Cho ma trận embedding:
$$
X \in \mathbb{R}^{T \times d}
$$


Ma trận hiệp phương sai:
$$
\Sigma = \frac{1}{T} X^T X
$$


Giải bài toán trị riêng:
$$
\Sigma v = \lambda v
$$


Chọn 2 trị riêng lớn nhất → chiếu xuống 2D:
$$
X_{2D} = X W_{2}
$$


---

## 4.2. t-SNE

t-SNE tối thiểu hóa KL-divergence giữa phân phối khoảng cách cao chiều và thấp chiều:
$$
\min_{Y}
D_{KL}(P | Q)
$$


Trong đó:
$$
D_{KL}(P|Q)
===========

\sum_{i,j}
P_{ij}
\log
\frac{P_{ij}}{Q_{ij}}
$$


---

# 5. Quan hệ ngữ nghĩa trong không gian embedding

Embedding học được tính chất tuyến tính.

Ví dụ:
$$
\text{King} - \text{Man} + \text{Woman} \approx \text{Queen}
$$


Về mặt vector:
$$
e_{king} - e_{man} + e_{woman}
\approx e_{queen}
$$


Điều này cho thấy embedding mã hóa cấu trúc ngữ nghĩa.

---

# 6. Self-Attention và tương tác token

Attention:
$$
\text{Attention}(Q,K,V)
=======================

\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$


Ma trận attention:
$$
A_{ij}
======

\frac
{\exp(q_i k_j / \sqrt{d_k})}
{\sum_j \exp(q_i k_j / \sqrt{d_k})}
$$


Trực quan hóa attention giúp hiểu:

* Token nào ảnh hưởng token nào
* Quan hệ phụ thuộc dài hạn

---

# 7. Độ phức tạp tính toán

Self-attention:
$$
\mathcal{O}(T^2 d)
$$


Nếu số token tăng:
$$
T \uparrow \Rightarrow \text{Memory} \uparrow
$$


Việc tạo token hiệu quả giúp:

* Giảm chiều dài chuỗi
* Giảm chi phí huấn luyện

---

# 8. Ví dụ minh họa quy trình

Cho câu:

> "Transformers process tokens"

Bước 1: Tokenization
$$
[1245, 5432, 987]
$$


Bước 2: Embedding
$$
Z \in \mathbb{R}^{3 \times d}
$$


Bước 3: Attention
$$
Z' = \text{Transformer}(Z)
$$


Bước 4: Visualization

* PCA → 2D
* t-SNE → cụm ngữ nghĩa

---

# 9. Ứng dụng trong huấn luyện GPT

Mô hình GPT tối ưu:
$$
P(x) = \prod_{t=1}^{T} P(x_t | x_{<t})
$$


Token là đơn vị cơ bản của xác suất.

Loss:
$$
\mathcal{L}
===========

-\sum_{t=1}^{T}
\log P(x_t | x_{<t})
$$


Nếu tokenization không tốt:

* Chuỗi dài
* Gradient nhiễu
* Hiệu suất giảm

---

# 10. Thảo luận

Tạo và trực quan hóa token giúp:

1. Hiểu cấu trúc embedding
2. Phát hiện bias
3. Phân tích clustering ngữ nghĩa
4. Kiểm tra alignment

Token không chỉ là ID — chúng là điểm trong không gian vector cao chiều.

---

# 11. Kết luận

Quá trình:
$$
\text{Text}
\rightarrow
\text{Token IDs}
\rightarrow
\text{Embedding}
\rightarrow
\text{Attention}
$$


là nền tảng của mọi mô hình ngôn ngữ hiện đại.

Trực quan hóa giúp:

* Giải thích mô hình
* Phân tích hành vi
* Cải thiện hiệu năng

---

# Tài liệu tham khảo

1. Vaswani, A. et al. (2017). *Attention Is All You Need*.
2. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*.
3. van der Maaten, L., Hinton, G. (2008). *Visualizing Data using t-SNE*.
4. Jolliffe, I. (2002). *Principal Component Analysis*.

-
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Tại sao văn bản cần được đánh số?](aero_llm_01_why_text_needs_to_be_numbered.md) | [Xem bài viết →](aero_llm_01_why_text_needs_to_be_numbered.md) |
| [Phân tích và chuyển đổi văn bản thành chuỗi token được đánh số: Cơ sở toán học và ứng dụng trong mô hình ngôn ngữ lớn](aero_llm_02_parsing_text_to_numbered_tokens.md) | [Xem bài viết →](aero_llm_02_parsing_text_to_numbered_tokens.md) |
| 📌 **[Tạo và trực quan hóa Token trong mô hình ngôn ngữ lớn: Cơ sở toán học và phân tích biểu diễn](aero_llm_03_codechallenge_create_and_visualize_tokens_part_1_.md)** | [Xem bài viết →](aero_llm_03_codechallenge_create_and_visualize_tokens_part_1_.md) |
| [Tạo và trực quan hóa Token (Phần 2): Phân tích hình học không gian embedding và Attention Map trong mô hình Transformer](aero_llm_04_codechallenge_create_and_visualize_tokens_part_2_.md) | [Xem bài viết →](aero_llm_04_codechallenge_create_and_visualize_tokens_part_2_.md) |
| [Chuẩn bị văn bản cho Tokenization trong mô hình Transformer: Cơ sở lý thuyết và phân tích toán học](aero_llm_05_preparing_text_for_tokenization.md) | [Xem bài viết →](aero_llm_05_preparing_text_for_tokenization.md) |
| [Phân tích quy trình Tokenization qua ví dụ *The Time Machine*: Cơ sở thuật toán và mô hình hóa toán học](aero_llm_06_codechallenge_tokenizing_the_time_machine.md) | [Xem bài viết →](aero_llm_06_codechallenge_tokenizing_the_time_machine.md) |
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
