
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
# Thuật toán Byte Pair Encoding (BPE) và Bài toán Tối ưu Kích thước Từ vựng trong Mô hình Ngôn ngữ

## Tóm tắt

Trong các mô hình ngôn ngữ hiện đại, đặc biệt là các hệ thống dựa trên kiến trúc Transformer, việc xây dựng tokenizer đóng vai trò nền tảng quyết định hiệu suất và chi phí tính toán. Tài liệu đính kèm trình bày một bài toán thực hành: **triển khai thuật toán Byte Pair Encoding (BPE) để đạt kích thước từ vựng mong muốn**. Bài viết này phân tích cơ sở lý thuyết của BPE, mô hình hóa toán học quá trình gộp cặp, bài toán tối ưu kích thước từ vựng, độ phức tạp tính toán và mối liên hệ với huấn luyện mô hình ngôn ngữ lớn (LLM).

---

## 1. Giới thiệu

Các mô hình học sâu xử lý văn bản thông qua ánh xạ:

$$
\text{text} \rightarrow \text{tokens} \rightarrow \text{embedding vectors}
$$

Một tokenizer hiệu quả cần:

- Giảm kích thước từ vựng $V$
- Hạn chế token ngoài tập huấn luyện (OOV)
- Giữ độ dài chuỗi $L$ ở mức hợp lý

Byte Pair Encoding (BPE) được đề xuất ban đầu cho nén dữ liệu (Gage, 1994) và được áp dụng cho NLP bởi Sennrich et al. (2016). Hiện nay, nhiều mô hình ngôn ngữ lớn sử dụng biến thể của BPE.

---

## 2. Mô hình hóa bài toán BPE

### 2.1 Biểu diễn dữ liệu

Giả sử tập dữ liệu huấn luyện:

$$
\mathcal{D} = \{w_1, w_2, \dots, w_N\}
$$

Mỗi từ được biểu diễn thành chuỗi ký tự:

$$
w_i = (c_1, c_2, \dots, c_m)
$$

Tập token ban đầu:

$$
V_0 = \{ \text{tất cả ký tự xuất hiện} \}
$$

---

### 2.2 Hàm đếm tần suất cặp token

Tại bước $k$, tập token là $V_k$.

Tập các cặp token liền kề:

$$
P_k = \{(t_i, t_{i+1})\}
$$

Hàm tần suất:

$$
f_k(p) = \sum_{w \in \mathcal{D}} \text{count}(p, w)
$$

Chọn cặp tối ưu:

$$
p_k^* = \arg\max_{p \in P_k} f_k(p)
$$

Sau đó cập nhật:

$$
V_{k+1} = V_k \cup \{ t_{new} \}
$$

Quá trình dừng khi:

$$
|V_k| = V_{target}
$$

---

## 3. Bài toán đạt kích thước từ vựng mong muốn

Giả sử:

- Từ vựng ban đầu: $|V_0| = C$
- Số vòng gộp: $M$

Khi đó:

$$
|V_M| = C + M
$$

Nếu muốn:

$$
|V_M| = V_{target}
$$

Ta cần:

$$
M = V_{target} - C
$$

Như vậy, bài toán trở thành:

> Thực hiện chính xác $M$ phép gộp có tần suất cao nhất.

---

## 4. Phân tích độ phức tạp tính toán

### 4.1 Mỗi vòng lặp

- Đếm tần suất tất cả cặp:

$$
\mathcal{O}(T)
$$

với $T$ là tổng số token trong tập dữ liệu.

- Chọn cặp lớn nhất:

$$
\mathcal{O}(|P_k|)
$$

### 4.2 Tổng thể

Với $M$ vòng lặp:

$$
\mathcal{O}(M \cdot T)
$$

Trong thực tế:

$$
T \approx 10^9 - 10^{12}
$$

Do đó cần:
- Cấu trúc heap
- Cập nhật tần suất cục bộ
- Phân mảnh dữ liệu (sharding)

---

## 5. Ảnh hưởng đến Mô hình Ngôn ngữ

### 5.1 Số tham số embedding

Ma trận embedding:

$$
E \in \mathbb{R}^{V \times d}
$$

Số tham số:

$$
\text{Params} = V \times d
$$

Ví dụ:

- $V = 50,000$
- $d = 4096$

$$
\text{Params} = 204,800,000
$$

Nếu tăng $V$ lên 100,000:

$$
\text{Params} = 409,600,000
$$

Chi phí tăng gấp đôi.

---

### 5.2 Ảnh hưởng đến Attention

Attention có độ phức tạp:

$$
\mathcal{O}(L^2 \cdot d)
$$

Trong đó:
- $L$ là chiều dài chuỗi token.

Nếu token quá nhỏ (character-level):

$$
L \uparrow \Rightarrow \text{Chi phí tăng}
$$

Nếu token quá lớn (word-level):

- OOV tăng
- Mất khả năng phân tích hình thái

BPE cân bằng hai yếu tố này.

---

## 6. So sánh BPE với WordPiece và Unigram LM

| Thuật toán | Tiêu chí tối ưu | Cơ chế |
|------------|----------------|---------|
| BPE | Tần suất cặp | Gộp lặp |
| WordPiece | Likelihood | Chọn cặp tối đa hóa xác suất |
| Unigram LM | Xác suất mô hình | Loại bỏ token kém |

BPE là phương pháp tham lam (greedy):

$$
\max f_k(p)
$$

Trong khi WordPiece tối ưu:

$$
\max \log P(\mathcal{D} | V_k)
$$

---

## 7. Mối liên hệ với Huấn luyện LLM

Giả sử:

- Tổng token huấn luyện: $T$
- Kích thước mô hình: $d$
- Số lớp: $L$

Chi phí huấn luyện xấp xỉ:

$$
\mathcal{O}(T \cdot L \cdot d^2)
$$

Việc chọn tokenizer ảnh hưởng trực tiếp đến:

- $T$ (số token sau phân tách)
- Hiệu quả tổng quát hóa
- Khả năng biểu diễn từ hiếm

---

## 8. Hạn chế

- Không xét ngữ nghĩa khi gộp
- Phụ thuộc dữ liệu huấn luyện ban đầu
- Có thể tạo token không trực quan

---

## 9. Kết luận

Thuật toán Byte Pair Encoding cung cấp một cơ chế phân tách từ hiệu quả, đặc biệt trong bối cảnh mô hình ngôn ngữ lớn. Bài toán đạt kích thước từ vựng mong muốn có thể được mô hình hóa thành việc thực hiện chính xác số vòng gộp cần thiết:

$$
M = V_{target} - |V_0|
$$

Việc tối ưu hóa BPE không chỉ là bước tiền xử lý, mà còn ảnh hưởng trực tiếp đến:

- Bộ nhớ
- Thời gian huấn luyện
- Chất lượng mô hình

Trong tương lai, các phương pháp tokenizer thích nghi động (adaptive tokenization) có thể thay thế BPE truyền thống nhằm tối ưu hóa tốt hơn theo mục tiêu huấn luyện.

---

## Tài liệu tham khảo

1. Gage, P. (1994). *A New Algorithm for Data Compression.*
2. Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.*
3. Vaswani, A. et al. (2017). *Attention Is All You Need.*
4. Kudo, T. (2018). *Subword Regularization.*
5. Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers.*

---
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
| 📌 **[Thuật toán Byte Pair Encoding (BPE) và Bài toán Tối ưu Kích thước Từ vựng trong Mô hình Ngôn ngữ](aero_llm_09_codechallenge_byte_pair_encoding_to_a_desired_vocab_size.md)** | [Xem bài viết →](aero_llm_09_codechallenge_byte_pair_encoding_to_a_desired_vocab_size.md) |
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
