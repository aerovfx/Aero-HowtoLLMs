
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
# Chuẩn bị văn bản cho Tokenization trong mô hình Transformer: Cơ sở lý thuyết và phân tích toán học

---

## Tóm tắt

Bài báo này trình bày một cách hệ thống quy trình **chuẩn bị văn bản trước khi tokenization** trong các mô hình ngôn ngữ lớn dựa trên kiến trúc Attention Is All You Need. Nội dung phân tích các bước tiền xử lý (text normalization, cleaning, encoding), cơ chế mã hóa Byte Pair Encoding (BPE), và vai trò của tokenizer trong các mô hình GPT do OpenAI phát triển. Bài viết bổ sung các công thức toán học mô tả xác suất chuỗi, ánh xạ rời rạc–liên tục và cấu trúc đại số của quá trình mã hóa.

---

# 1. Giới thiệu

Trong các mô hình Transformer hiện đại như GPT-4, văn bản đầu vào không được xử lý trực tiếp dưới dạng ký tự mà phải trải qua quá trình:

$$
\text{Raw Text} \rightarrow \text{Normalization} \rightarrow \text{Tokenization} \rightarrow \text{Embedding}
$$

Tokenization đóng vai trò là cầu nối giữa:

* Không gian rời rạc của ký tự
* Không gian vector liên tục của embedding

Nếu gọi chuỗi văn bản ban đầu là:

$$
X = (c_1, c_2, \dots, c_n)
$$

thì tokenizer ánh xạ:

$$
\tau: \Sigma^* \rightarrow \mathbb{Z}^m
$$

với (\Sigma) là bảng chữ cái và (\mathbb{Z}^m) là chuỗi ID token.

---

# 2. Chuẩn hóa văn bản (Text Normalization)

Chuẩn hóa giúp đảm bảo tính nhất quán của dữ liệu huấn luyện.

## 2.1 Lowercasing

Ánh xạ:

$$
f_{lower}(c) = \text{lowercase}(c)
$$

Ví dụ:

$$
\text{"ChatGPT"} \rightarrow \text{"chatgpt"}
$$

## 2.2 Unicode Normalization

Văn bản Unicode có thể biểu diễn cùng một ký tự theo nhiều cách.

Chuẩn NFC:

$$
\text{é} = e + \acute{}
$$

Chuẩn hóa đảm bảo:

$$
NFC(x_1) = NFC(x_2)
$$

nếu hai chuỗi tương đương về mặt ngữ nghĩa.

---

# 3. Tokenization: Cơ sở xác suất

Mô hình ngôn ngữ tối ưu xác suất:

$$
P(X) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

Tuy nhiên, nếu làm việc ở mức ký tự:

$$
T = n
$$

Số bước dự đoán lớn → chi phí cao.

Giải pháp:

Chia thành token:

$$
X = (w_1, w_2, \dots, w_m), \quad m < n
$$

Giảm độ dài chuỗi và tăng tính biểu diễn.

---

# 4. Byte Pair Encoding (BPE)

BPE được giới thiệu cho NLP bởi Sennrich et al. (2016).

## 4.1 Thuật toán

Ban đầu:

$$
V_0 = { \text{tập ký tự đơn} }
$$

Lặp:

1. Tìm cặp ký tự xuất hiện nhiều nhất
2. Gộp thành token mới
3. Cập nhật từ vựng

Giả sử tần suất cặp ((a,b)):

$$
f(a,b) = \sum_{i} \mathbb{I}[(a,b) \in X_i]
$$

Chọn:

$$
(a^*, b^*) = \arg\max_{a,b} f(a,b)
$$

Cập nhật:

$$
V_{k+1} = V_k \cup {ab}
$$

---

# 5. Không gian rời rạc và ánh xạ embedding

Sau tokenization:

$$
w_i \rightarrow id_i \in {1, \dots, |V|}
$$

Embedding matrix:

$$
E \in \mathbb{R}^{|V| \times d}
$$

Ánh xạ:

$$
e_i = E[id_i]
$$

Toàn bộ chuỗi:

$$
X \rightarrow (e_1, e_2, \dots, e_m)
$$

---

# 6. Phân tích độ phức tạp

Nếu:

* (N) là số ký tự
* (V) là kích thước từ vựng

Chi phí xây dựng BPE:

$$
\mathcal{O}(N \log V)
$$

Chi phí suy luận tokenization:

$$
\mathcal{O}(m)
$$

---

# 7. Vấn đề Out-of-Vocabulary (OOV)

Không như Word2Vec truyền thống, BPE đảm bảo:

$$
\forall x \in \Sigma^*, \exists \text{ decomposition into subwords}
$$

Ví dụ:

```
tokenization → token + ization
```

Điều này đảm bảo:

$$
P(x) > 0
$$

cho mọi chuỗi hợp lệ.

---

# 8. So sánh với các phương pháp khác

| Phương pháp     | Đặc điểm        | Hạn chế           |
| --------------- | --------------- | ----------------- |
| Word-level      | Ngắn, dễ hiểu   | OOV cao           |
| Character-level | Không OOV       | Chuỗi dài         |
| BPE             | Cân bằng        | Phụ thuộc dữ liệu |
| Unigram LM      | Xác suất tối ưu | Tính toán cao     |

Unigram Language Model tối ưu:

$$
\max_{\theta} \prod_i \sum_{z \in \mathcal{Z}(x_i)} P(z|\theta)
$$

---

# 9. Tác động đến Attention

Độ dài chuỗi ảnh hưởng trực tiếp đến chi phí self-attention:

$$
\text{Complexity} = \mathcal{O}(T^2 d)
$$

Nếu tokenization kém → (T) lớn → chi phí tăng.

Do đó, tokenizer tối ưu giúp:

* Giảm memory footprint
* Tăng tốc inference
* Cải thiện chất lượng ngữ nghĩa

---

# 10. Liên hệ thực tế trong GPT

Các mô hình GPT sử dụng biến thể của BPE hoặc byte-level BPE.

Xác suất sinh token:

$$
P(w_t | w_{<t}) =
\frac{\exp(z_t W_{out})}
{\sum_j \exp(z_j W_{out})}
$$

Chất lượng tokenization ảnh hưởng trực tiếp đến phân phối logits.

---

# 11. Thảo luận

Chuẩn bị văn bản không chỉ là bước tiền xử lý kỹ thuật mà còn là:

* Bài toán tối ưu thông tin
* Bài toán mã hóa nguồn (source coding)
* Bài toán nén dữ liệu

Theo định lý Shannon:

$$
H(X) = - \sum_x P(x) \log P(x)
$$

Tokenizer tốt giúp:

$$
\text{Length}(X_{tokens}) \approx \frac{H(X)}{\log |V|}
$$

---

# 12. Kết luận

Quy trình chuẩn bị văn bản cho tokenization bao gồm:

1. Chuẩn hóa Unicode
2. Làm sạch dữ liệu
3. Áp dụng BPE hoặc Unigram LM
4. Ánh xạ sang embedding

Toán học cho thấy tokenization là quá trình:

$$
\Sigma^* \rightarrow V^*
$$

giúp tối ưu:

* Độ dài chuỗi
* Độ phức tạp tính toán
* Biểu diễn ngữ nghĩa

---

# Tài liệu tham khảo

1. Attention Is All You Need
2. Sennrich, R., Haddow, B., Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*.
3. Kudo, T. (2018). *Subword Regularization: Improving Neural Network Translation Models*.
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
| 📌 **[Chuẩn bị văn bản cho Tokenization trong mô hình Transformer: Cơ sở lý thuyết và phân tích toán học](aero_llm_05_preparing_text_for_tokenization.md)** | [Xem bài viết →](aero_llm_05_preparing_text_for_tokenization.md) |
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
