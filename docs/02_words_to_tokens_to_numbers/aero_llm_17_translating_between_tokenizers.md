
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
Chuyển đổi giữa các Tokenizer trong Mô hình Ngôn ngữ Lớn:

Phân tích Lý thuyết, Biểu diễn Hình thức và Hệ quả Tính toán

⸻

Tóm tắt

Tokenization là bước tiền xử lý cốt lõi trong các mô hình ngôn ngữ lớn (LLMs). Tuy nhiên, sự khác biệt giữa các thuật toán token hóa như WordPiece, BPE và Unigram LM tạo ra những thách thức khi cần chuyển đổi giữa các tokenizer khác nhau. Bài viết này, dựa trên tài liệu đính kèm về Translating between tokenizers, trình bày một khung lý thuyết hình thức cho bài toán ánh xạ giữa hai hệ tokenizer, phân tích điều kiện tồn tại ánh xạ song ánh, ước lượng sai số thông tin, và đánh giá tác động đến độ phức tạp tính toán trong Transformer. Các ví dụ được minh họa với tokenizer của BERT, GPT-2, và thư viện SentencePiece.

⸻

1. Giới thiệu

Mỗi mô hình ngôn ngữ định nghĩa một hàm token hóa:

\mathcal{T}: \Sigma^{\ast} \rightarrow V^{\ast}

Trong đó:
	•	\Sigma^{\ast}: tập tất cả chuỗi ký tự
	•	V: từ vựng token
	•	V^{\ast}: chuỗi token

Hai tokenizer khác nhau \mathcal{T}_A và \mathcal{T}_B sẽ tạo ra hai biểu diễn khác nhau cho cùng một chuỗi đầu vào x:

\mathcal{T}_A(x) \neq \mathcal{T}_B(x)

Vấn đề đặt ra:
Làm thế nào để ánh xạ chuỗi token từ không gian V_A^{\ast} sang V_B^{\ast} mà không mất thông tin?

⸻

2. Các Hệ Tokenizer Phổ biến

2.1 WordPiece

Được sử dụng trong BERT do Google phát triển.

Thuật toán tối đa hóa xác suất:

\arg\max_{s_1,\dots,s_k} \prod_{i=1}^{k} P(s_i)

⸻

2.2 Byte Pair Encoding (BPE)

Được sử dụng trong GPT-2 bởi OpenAI.

Quá trình lặp:

(\alpha, \beta) = \arg\max_{(u,v)} \text{freq}(uv)

Sau đó thay thế cặp phổ biến nhất.

⸻

2.3 Unigram Language Model

Áp dụng trong SentencePiece.

Tối ưu hóa:

\max_{V} \sum_{x \in D} \log \sum_{s \in \mathcal{S}(x)} \prod_{i} P(s_i)

⸻

3. Mô hình Toán học của Bài toán Chuyển đổi

Giả sử:

\mathcal{T}_A: \Sigma^{\ast} \rightarrow V_A^{\ast}

\mathcal{T}_B: \Sigma^{\ast} \rightarrow V_B^{\ast}

Ta cần xây dựng:

\Phi: V_A^{\ast} \rightarrow V_B^{\ast}

3.1 Điều kiện tồn tại ánh xạ chính xác

Nếu tồn tại hàm giải mã:

\mathcal{D}_A: V_A^{\ast} \rightarrow \Sigma^{\ast}

thì:

\Phi = \mathcal{T}_B \circ \mathcal{D}_A

Khi đó:

\Phi(\mathcal{T}_A(x)) = \mathcal{T}_B(x)

⸻

4. Phân tích Sai số Thông tin

Nếu tokenizer không khả nghịch hoàn toàn, ta có sai số:

\epsilon = d(\mathcal{D}_A(\mathcal{T}_A(x)), x)

Trong đó d là khoảng cách Levenshtein.

Entropy trước và sau:

H_A = - \sum p(t_i)\log p(t_i)

H_B = - \sum p(u_j)\log p(u_j)

Độ chênh entropy:

\Delta H = |H_A - H_B|

Nếu \Delta H lớn → thay đổi cấu trúc phân bố token đáng kể.

⸻

5. Ảnh hưởng đến Độ dài Chuỗi và Self-Attention

Giả sử văn bản có n ký tự.

Số token:

m_A = \frac{n}{\mathbb{E}[L_A]}

m_B = \frac{n}{\mathbb{E}[L_B]}

Self-attention có độ phức tạp:

O(m^2)

Tỷ lệ chi phí:

\frac{C_A}{C_B} = \left(\frac{m_A}{m_B}\right)^2

Nếu tokenizer B tạo token dài hơn:

\mathbb{E}[L_B] > \mathbb{E}[L_A]
\Rightarrow C_B < C_A

⸻

6. Bài toán Căn chỉnh Token (Token Alignment)

Giả sử:

\mathcal{T}_A(x) = (a_1, a_2, \dots, a_m)

\mathcal{T}_B(x) = (b_1, b_2, \dots, b_k)

Ta cần tìm ánh xạ căn chỉnh:

\pi: \{1,\dots,m\} \rightarrow \{1,\dots,k\}

Tối ưu hóa:

\min_{\pi} \sum_{i=1}^{m} d(\text{span}(a_i), \text{span}(b_{\pi(i)}))

Đây tương đương bài toán căn chỉnh chuỗi động (dynamic programming).

⸻

7. Biểu diễn Ma trận Ánh xạ

Ta có thể định nghĩa ma trận chuyển đổi:

M \in \mathbb{R}^{|V_A| \times |V_B|}

Trong đó:

M_{ij} = P(b_j \mid a_i)

Nếu ánh xạ xác định:

M_{ij} \in \{0,1\}

Nếu ánh xạ xác suất:

\sum_j M_{ij} = 1

⸻

8. Ứng dụng Thực tiễn
	1.	Chuyển embedding giữa hai mô hình
	2.	Fine-tune chéo tokenizer
	3.	Distillation giữa hai LLM
	4.	Interoperability giữa hệ sinh thái NLP

⸻

9. Thảo luận

Sự khác biệt giữa tokenizer không chỉ ảnh hưởng đến:
	•	Độ dài chuỗi
	•	Chi phí attention
	•	Entropy hệ biểu diễn

Mà còn ảnh hưởng đến:
	•	Phân bố gradient
	•	Ổn định huấn luyện
	•	Tính chuyển giao embedding

Bài toán chuyển đổi tokenizer thực chất là bài toán ánh xạ giữa hai hệ mã hóa rời rạc có cấu trúc phân cấp.

⸻

10. Kết luận

Việc chuyển đổi giữa hai tokenizer có thể được mô hình hóa hình thức bằng:

\Phi = \mathcal{T}_B \circ \mathcal{D}_A

Sai số thông tin được đo bằng:

\epsilon = d(\mathcal{D}_A(\mathcal{T}_A(x)), x)

Độ phức tạp tính toán phụ thuộc vào:

O\left(\left(\frac{n}{\mathbb{E}[L]}\right)^2\right)

Thiết kế tokenizer không chỉ là vấn đề tiền xử lý mà là một thành phần cấu trúc của toàn bộ kiến trúc Transformer.

⸻

Tài liệu tham khảo
	1.	Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	2.	Radford et al. (2019). Language Models are Unsupervised Multitask Learners.
	3.	Kudo & Richardson (2018). SentencePiece: A simple and language independent subword tokenizer.
	4.	Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
	5.	Shannon, C. (1948). A Mathematical Theory of Communication.
	6.	Vaswani et al. (2017). Attention Is All You Need.
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
| 📌 **[aero llm 17 translating between tokenizers](aero_llm_17_translating_between_tokenizers.md)** | [Xem bài viết →](aero_llm_17_translating_between_tokenizers.md) |
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
