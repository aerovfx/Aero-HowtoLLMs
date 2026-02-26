
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
Phân tích Thống kê Số lượng Token theo Độ dài Subword (Phần 2): Mô hình hóa Toán học và Hàm Phân bố

⸻

Tóm tắt

Bài báo này tiếp tục phân tích thống kê số lượng token theo độ dài subword trong bộ tokenizer của GPT-4, dựa trên dữ liệu thực nghiệm từ tài liệu đính kèm (phần 2). Chúng tôi xây dựng mô hình toán học cho phân bố độ dài, kiểm định giả thuyết phân bố mũ và luật Zipf, đồng thời phân tích tác động của cấu trúc token đến độ phức tạp tính toán trong kiến trúc Transformer của OpenAI. Kết quả cho thấy phân bố độ dài subword có xu hướng suy giảm phi tuyến, gần với hàm mũ hoặc log-linear, và có mối liên hệ chặt chẽ với entropy hệ token.

⸻

1. Giới thiệu

Trong các mô hình ngôn ngữ lớn (LLMs), tokenization là bước ánh xạ văn bản thô thành chuỗi token rời rạc:

$$
S = (c_1, c_2, ..., c_n)
$$

$$
T = (t_1, t_2, ..., t_m)
$$

Với:

$$
m \le n
$$

Mỗi token t_i có độ dài \ell(t_i) tính theo byte hoặc ký tự Unicode.

Phần 2 của dữ liệu thực nghiệm tập trung vào:
	•	Phân bố chi tiết ở các độ dài lớn hơn
	•	Sự suy giảm số lượng token khi độ dài tăng
	•	Quan hệ giữa độ dài và tần suất xuất hiện

⸻

2. Mô hình hóa Phân bố Độ dài Subword

2.1 Phân bố xác suất rời rạc

Gọi:
	•	V: tập từ vựng
	•	N_k: số token có độ dài k

Khi đó:

$$
P(L = k) = \frac{N_k}{|V|}
$$

Và:

$$
\sum_{k=1}^{K_{\max}} P(L = k) = 1
$$

⸻

2.2 Giả thuyết phân bố mũ

Dữ liệu thực nghiệm cho thấy:

N_k \approx Ae^{-\lambda k}

Suy ra:

$$
P(L = k) = \frac{Ae^{-\lambda k}}{\sum_{j=1}^{K_{\max}} Ae^{-\lambda j}}
$$

Chuẩn hóa:

$$
P(L = k) = (1 - e^{-\lambda}) e^{-\lambda (k-1)}
$$

Đây là phân bố hình học rời rạc.

⸻

2.3 Kỳ vọng và Phương sai

Kỳ vọng:

$$
\mathbb{E}[L] = \frac{1}{1 - e^{-\lambda}}
$$

Phương sai:

$$
\mathrm{Var}(L) = \frac{e^{-\lambda}}{(1 - e^{-\lambda})^2}
$$

Điều này cho thấy khi \lambda nhỏ:
	•	Đuôi phân bố dài hơn
	•	Tồn tại nhiều token dài

⸻

3. Liên hệ với Luật Zipf

Tần suất token theo thứ hạng:

$$
f(r) \propto \frac{1}{r^\alpha}
$$

Trong đó:
	•	r: thứ hạng
	•	\alpha \approx 1

Kết hợp hai quan sát:
	•	Token ngắn → tần suất cao
	•	Token dài → tần suất thấp

Ta có mô hình kết hợp:

$$
P(t) \propto e^{-\beta \ell(t)} \cdot \frac{1}{r^\alpha}
$$

⸻

4. Ảnh hưởng đến Độ dài Chuỗi và Chi phí Attention

Giả sử văn bản có tổng số ký tự n.

Số token:

$$
m = \frac{n}{\mathbb{E}[L]}
$$

Self-attention có độ phức tạp:

$$
O(m^2)
$$

Thay vào:

$$
O\left(\left(\frac{n}{\mathbb{E}[L]}\right)^2\right)
$$

Do đó:
	•	Nếu \mathbb{E}[L] \uparrow \Rightarrow m \downarrow \Rightarrow \text{Cost} \downarrow
	•	Nếu token quá dài → vocabulary lớn → tăng chi phí embedding

⸻

5. Entropy của Hệ Token

Entropy:

$$
H = - \sum_{t \in V} P(t) \log P(t)
$$

Thay mô hình mũ:

$$
H \approx - \sum_{k} P(L=k) \log P(L=k)
$$

Với phân bố hình học:

$$
H = - \sum_{k=1}^{\infty} (1-q) q^{k-1} \log[(1-q) q^{k-1}]
$$

Trong đó:

$$
q = e^{-\lambda}
$$

Entropy tối ưu khi:
	•	Không quá tập trung vào token cực ngắn
	•	Không quá phân tán ở token dài

⸻

6. Kiểm định Phù hợp Mô hình

Để kiểm tra giả thuyết phân bố mũ, có thể sử dụng:

6.1 Hồi quy log-linear

$$
\log N_k = \log A - \lambda k
$$

Nếu đồ thị \log N_k theo k tuyến tính → xác nhận mô hình mũ.

⸻

6.2 Kiểm định Chi-square

$$
\chi^2 = \sum_{k} \frac{(N_k - \hat{N}_k)^2}{\hat{N}_k}
$$

So sánh với phân bố lý thuyết.

⸻

7. Hàm Tối ưu Hóa Ngầm trong Tokenizer

Tokenizer BPE thực chất tối ưu xấp xỉ:

$$
\min_{V} \left( \mathbb{E}[m] + \lambda |V| \right)
$$

Trong đó:
	•	\mathbb{E}[m]: số token trung bình
	•	|V|: kích thước từ vựng
	•	\lambda: hệ số điều chỉnh

Đây là bài toán cân bằng giữa:
	•	Độ nén chuỗi
	•	Kích thước embedding matrix

⸻

8. Thảo luận

Phần 2 của dữ liệu thực nghiệm cho thấy:
	•	Phân bố không hoàn toàn tuyến tính
	•	Có đuôi dài nhẹ (heavy-tail)
	•	Một số token đặc biệt dài đại diện cho chuỗi phổ biến

Điều này phù hợp với lý thuyết:
	•	Ngôn ngữ tự nhiên có cấu trúc fractal
	•	Zipf và phân bố mũ thường xuất hiện trong hệ thống thông tin

⸻

9. Kết luận

Phân bố độ dài subword có thể được mô hình hóa gần đúng bằng phân bố mũ rời rạc:

$$
P(L = k) \sim e^{-\lambda k}
$$

Tác động trực tiếp đến:

$$
m = \frac{n}{\mathbb{E}[L]}
$$

\text{Attention Cost} \sim O(m^2)

H = - \sum P(t)\log P(t)

Do đó, thiết kế tokenizer là bài toán tối ưu đa mục tiêu giữa:
	•	Độ dài chuỗi
	•	Kích thước từ vựng
	•	Entropy thông tin
	•	Chi phí tính toán

⸻

Tài liệu tham khảo
	1.	Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units.
	2.	Vaswani, A. et al. (2017). Attention Is All You Need.
	3.	Shannon, C. (1948). A Mathematical Theory of Communication.
	4.	Kudo, T. (2018). Subword Regularization.
	5.	Brown, T. et al. (2020). Language Models are Few-Shot Learners.
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
| 📌 **[aero llm 12 codechallenge token count by subword length part 2](aero_llm_12_codechallenge_token_count_by_subword_length_part_2_.md)** | [Xem bài viết →](aero_llm_12_codechallenge_token_count_by_subword_length_part_2_.md) |
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
