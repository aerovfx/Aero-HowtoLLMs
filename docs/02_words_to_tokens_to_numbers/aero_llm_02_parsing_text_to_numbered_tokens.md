
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [02 words to tokens to numbers](../index.md)

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
# Phân tích và chuyển đổi văn bản thành chuỗi token được đánh số: Cơ sở toán học và ứng dụng trong mô hình ngôn ngữ lớn

---

## Tóm tắt

Quá trình phân tích (parsing) văn bản thành các token được đánh số là bước nền tảng trong huấn luyện và suy luận của các mô hình ngôn ngữ lớn (LLMs). Bài viết này trình bày cơ sở lý thuyết của tokenization, đánh số vị trí (positional indexing), và vai trò của chúng trong kiến trúc Transformer. Phân tích dựa trên các công trình nền tảng như Transformer của Vaswani et al. (2017) và các mô hình GPT do OpenAI phát triển. Các công thức toán học minh họa quá trình ánh xạ văn bản sang không gian vector và cách mô hình xử lý chuỗi có thứ tự.

---

# 1. Giới thiệu

Máy tính không xử lý trực tiếp “từ” hay “câu” như con người, mà xử lý **chuỗi số**.

Do đó, văn bản phải được:

1. Tách thành token
2. Ánh xạ thành chỉ số (ID)
3. Chuyển thành vector embedding
4. Đánh số theo vị trí trong chuỗi

Ví dụ:

> "AI is powerful"

Sau tokenization có thể trở thành:

[
["AI", " is", " powerful"]
]

Và được ánh xạ thành:

[
[50256, 318, 3665]
]

---

# 2. Tokenization: Cơ sở toán học

Giả sử tập từ vựng (V) có kích thước:

[
|V| = N
]

Hàm tokenization:

[
T: \mathcal{X} \to V^T
]

với:

* ( \mathcal{X} ): không gian văn bản
* (V^T): chuỗi các token ID

Nếu chuỗi văn bản là (x), ta có:

[
T(x) = (t_1, t_2, ..., t_T)
]

Mỗi (t_i \in {1,2,...,N})

---

# 3. Byte Pair Encoding (BPE)

GPT sử dụng BPE để xử lý từ hiếm.

Giả sử ban đầu ta có tập ký tự (C).
Thuật toán lặp:

1. Tìm cặp ký tự xuất hiện nhiều nhất
2. Gộp thành token mới
3. Thêm vào từ vựng

Quá trình tối ưu hóa nhằm giảm entropy:

[
H(X) = -\sum_x P(x)\log P(x)
]

BPE giúp:

* Giảm độ dài chuỗi (T)
* Tăng hiệu quả tính toán

---

# 4. Đánh số token (Positional Indexing)

Sau tokenization:

[
(t_1, t_2, ..., t_T)
]

Ta cần biểu diễn thứ tự:

[
i = 1,2,...,T
]

Nếu không có chỉ số vị trí, mô hình Transformer sẽ bất biến hoán vị.

---

## 4.1. Biểu diễn embedding

Mỗi token ID được ánh xạ:

[
e_i = E(t_i)
]

Vector đầu vào cuối cùng:

[
z_i = e_i + p_i
]

Trong đó:

* (p_i): vector vị trí

---

# 5. Self-Attention và vai trò của thứ tự

Attention được định nghĩa:

[
\text{Attention}(Q,K,V)
=======================

\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
]

Nếu không có positional encoding:

[
\text{Attention}(PX) = P\text{Attention}(X)
]

→ Không phân biệt thứ tự.

Khi thêm (p_i):

[
Z = E + P
]

ma trận attention phản ánh quan hệ phụ thuộc có hướng.

---

# 6. Causal Masking

Trong mô hình tự hồi quy (GPT):

[
P(x) = \prod_{t=1}^{T} P(x_t | x_{<t})
]

Mask:

[
M_{ij} =
\begin{cases}
0 & j \le i \
-\infty & j > i
\end{cases}
]

Ma trận attention thực tế:

[
A = \text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}} + M
\right)
]

Đánh số token cho phép xác định chính xác vị trí (i).

---

# 7. Độ phức tạp tính toán

Self-attention:

[
\mathcal{O}(T^2 d)
]

Nếu chiều dài chuỗi tăng gấp đôi:

[
\text{Compute} \approx 4\times
]

Do đó việc tokenization hiệu quả giúp:

* Giảm (T)
* Giảm chi phí huấn luyện

---

# 8. Ví dụ minh họa

Giả sử câu:

> "Machine learning is amazing"

Tokenization:

[
[1543, 4673, 318, 4996]
]

Embedding:

[
E \in \mathbb{R}^{|V| \times d}
]

Đầu vào:

[
Z \in \mathbb{R}^{T \times d}
]

Qua attention:

[
Z' = \text{Transformer}(Z)
]

---

# 9. Liên hệ với Reinforcement Learning from Human Feedback

Trong RLHF:

[
x = [\text{Prompt}; \text{Response}]
]

Đánh số cho phép:

* Phân biệt đoạn cần tối ưu
* Mask loss chính xác

Loss:

[
\mathcal{L} = - \sum_{t \in R} \log P(x_t | x_{<t})
]

---

# 10. Thảo luận

Quá trình parsing text to numbered tokens là:

* Bước đầu tiên của NLP pipeline
* Điều kiện cần cho Transformer hoạt động
* Yếu tố quyết định hiệu suất tính toán

Nếu bỏ bước này:

[
\text{Model} \to \text{Không thể huấn luyện}
]

---

# 11. Kết luận

Chuyển đổi văn bản thành chuỗi token được đánh số là:

1. Nền tảng của mô hình ngôn ngữ
2. Cơ sở cho self-attention
3. Điều kiện để thực hiện causal modeling

Toán học cho thấy thứ tự là thành phần thiết yếu trong biểu diễn ngôn ngữ.

---

# Tài liệu tham khảo

1. Vaswani, A. et al. (2017). *Attention Is All You Need*.
2. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*.
3. Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*.
4. Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.

---
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Tại sao văn bản cần được đánh số?](aero_llm_01_why_text_needs_to_be_numbered.md) | [Xem bài viết →](aero_llm_01_why_text_needs_to_be_numbered.md) |
| 📌 **[Phân tích và chuyển đổi văn bản thành chuỗi token được đánh số: Cơ sở toán học và ứng dụng trong mô hình ngôn ngữ lớn](aero_llm_02_parsing_text_to_numbered_tokens.md)** | [Xem bài viết →](aero_llm_02_parsing_text_to_numbered_tokens.md) |
| [Tạo và trực quan hóa Token trong mô hình ngôn ngữ lớn: Cơ sở toán học và phân tích biểu diễn](aero_llm_03_codechallenge_create_and_visualize_tokens_part_1_.md) | [Xem bài viết →](aero_llm_03_codechallenge_create_and_visualize_tokens_part_1_.md) |
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
