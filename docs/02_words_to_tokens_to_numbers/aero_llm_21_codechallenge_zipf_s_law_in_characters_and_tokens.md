
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
Luật Zipf trong Phân bố Ký tự và Token:

Phân tích Định lượng và Hệ quả đối với Tokenization trong Mô hình Ngôn ngữ

⸻

Tóm tắt

Dựa trên tài liệu đính kèm “Zipf’s Law in Characters and Tokens”, bài viết này phân tích sự xuất hiện của luật Zipf trong phân bố tần suất ký tự và token trong văn bản tự nhiên. Chúng tôi xây dựng mô hình toán học cho phân bố thứ hạng–tần suất, so sánh hành vi giữa mức ký tự và mức token (subword), và phân tích tác động đến thiết kế tokenizer cũng như chi phí tính toán của kiến trúc Transformer. Các ví dụ được minh họa với tokenizer của BERT và GPT-2.

⸻

1. Giới thiệu

Trong ngôn ngữ tự nhiên, tần suất xuất hiện của đơn vị ngôn ngữ (ký tự, từ, token) không phân bố đều mà tuân theo quy luật lũy thừa, được biết đến là Luật Zipf, do George Kingsley Zipf đề xuất.

Nếu r là thứ hạng của một đơn vị (1 là phổ biến nhất), thì tần suất f(r) được xấp xỉ bởi:

f(r) \propto \frac{1}{r^\alpha}

với:

\alpha \approx 1

Luật này xuất hiện ở cả mức ký tự và mức token.

⸻

2. Luật Zipf ở Mức Ký tự

Gọi:
	•	\Sigma: bảng chữ cái
	•	|\Sigma| = K

Sắp xếp ký tự theo tần suất giảm dần.

f_c(r) = C r^{-\alpha_c}

Tổng xác suất:

\sum_{r=1}^{K} f_c(r) = 1

Chuẩn hóa:

C = \left( \sum_{r=1}^{K} r^{-\alpha_c} \right)^{-1}

Với tiếng Anh:

\alpha_c \approx 1

Do bảng chữ cái nhỏ (26–100 ký tự), phân bố có đuôi ngắn.

⸻

3. Luật Zipf ở Mức Token

Với token (subword), kích thước từ vựng:

|V| \approx 30{,}000

Phân bố:

f_t(r) = C' r^{-\alpha_t}

Thông thường:

\alpha_t \in [0.8, 1.2]

Phân bố token có đuôi dài hơn nhiều so với ký tự.

⸻

4. So sánh Entropy

Entropy ký tự:

H_c = - \sum_{r=1}^{K} f_c(r)\log f_c(r)

Entropy token:

H_t = - \sum_{r=1}^{|V|} f_t(r)\log f_t(r)

Với phân bố Zipf:

H \approx \log Z(\alpha) + \frac{\alpha}{Z(\alpha)} \sum_{r} r^{-\alpha}\log r

Trong đó:

Z(\alpha) = \sum_{r=1}^{N} r^{-\alpha}

Vì |V| \gg K, nên:

H_t > H_c

⸻

5. Ảnh hưởng đến Tỷ lệ Nén

Giả sử văn bản có:
	•	n ký tự
	•	m token

Compression ratio:

R = \frac{n}{m}

Theo bảo toàn thông tin:

n H_c \approx m H_t

Suy ra:

R \approx \frac{H_t}{H_c}

Nếu H_t tăng (do đuôi dài của Zipf), R tăng → chuỗi token ngắn hơn.

⸻

6. Hệ quả đối với Transformer

Self-attention có độ phức tạp:

O(m^2)

Thay m = \frac{n}{R}:

O\left(\frac{n^2}{R^2}\right)

Vì luật Zipf tạo ra:
	•	Ít token cực kỳ phổ biến
	•	Nhiều token hiếm

Gradient trong huấn luyện sẽ:

\text{Var}(\nabla) \uparrow

đối với token hiếm.

⸻

7. Phân tích Phổ Tần suất (Frequency Spectrum)

Tổng số lần xuất hiện của token thứ hạng r:

N_r = N_1 r^{-\alpha}

Tổng số token trong corpus:

T = \sum_{r=1}^{|V|} N_r

Xấp xỉ tích phân:

T \approx N_1 \int_1^{|V|} r^{-\alpha} dr

Nếu \alpha = 1:

T \approx N_1 \log |V|

Điều này giải thích tại sao:
	•	Tăng từ vựng → tăng nhẹ tổng khối lượng thông tin
	•	Đuôi dài vẫn chiếm phần đáng kể

⸻

8. So sánh giữa Ký tự và Token trong Thực tế

8.1 Ở mức ký tự
	•	Bảng chữ cái nhỏ
	•	Phân bố ít cực đoan

8.2 Ở mức token (WordPiece/BPE)

Áp dụng trong BERT và GPT-2:
	•	Một số token cực phổ biến (“the”, “##ing”)
	•	Nhiều token xuất hiện rất hiếm

Đuôi dài mạnh hơn → phù hợp luật Zipf.

⸻

9. Ảnh hưởng đến Thiết kế Tokenizer

Nếu từ vựng quá nhỏ:

|V| \downarrow \Rightarrow \alpha_t \uparrow

Phân bố dốc hơn → token phổ biến chiếm ưu thế.

Nếu từ vựng quá lớn:

|V| \uparrow \Rightarrow \text{đuôi dài mạnh}

Tối ưu hóa:

\min_{|V|} \left( \frac{n^2}{R^2} + \lambda |V| \right)

⸻

10. Thảo luận

Luật Zipf cho thấy:
	1.	Ngôn ngữ tự nhiên có cấu trúc tự tổ chức
	2.	Tokenization kế thừa tính chất lũy thừa
	3.	Phân bố đuôi dài ảnh hưởng đến huấn luyện
	4.	Thiết kế tokenizer phải cân bằng giữa nén và phân bố tần suất

Các hệ như Google và OpenAI đã chọn kích thước từ vựng nhằm cân bằng giữa entropy và chi phí tính toán.

⸻

11. Kết luận

Luật Zipf trong ký tự và token được mô tả bởi:

f(r) \propto r^{-\alpha}

Entropy:

H = -\sum f(r)\log f(r)

Compression ratio:

R \approx \frac{H_t}{H_c}

Chi phí attention:

O\left(\frac{n^2}{R^2}\right)

Do đó, phân bố lũy thừa không chỉ là hiện tượng ngôn ngữ học mà còn ảnh hưởng trực tiếp đến hiệu năng tính toán của mô hình ngôn ngữ.

⸻

Tài liệu tham khảo
	1.	Zipf, G. K. (1935). The Psycho-Biology of Language.
	2.	Shannon, C. (1948). A Mathematical Theory of Communication.
	3.	Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	4.	Radford et al. (2019). GPT-2: Language Models are Unsupervised Multitask Learners.
	5.	Vaswani et al. (2017). Attention Is All You Need.
	6.	Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
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
| [aero llm 17 translating between tokenizers](aero_llm_17_translating_between_tokenizers.md) | [Xem bài viết →](aero_llm_17_translating_between_tokenizers.md) |
| [aero llm 18 codechallenge more on token translation](aero_llm_18_codechallenge_more_on_token_translation.md) | [Xem bài viết →](aero_llm_18_codechallenge_more_on_token_translation.md) |
| [aero llm 19 codechallenge tokenization compression ratios](aero_llm_19_codechallenge_tokenization_compression_ratios.md) | [Xem bài viết →](aero_llm_19_codechallenge_tokenization_compression_ratios.md) |
| [aero llm 20 tokenization in different languages](aero_llm_20_tokenization_in_different_languages.md) | [Xem bài viết →](aero_llm_20_tokenization_in_different_languages.md) |
| 📌 **[aero llm 21 codechallenge zipf s law in characters and tokens](aero_llm_21_codechallenge_zipf_s_law_in_characters_and_tokens.md)** | [Xem bài viết →](aero_llm_21_codechallenge_zipf_s_law_in_characters_and_tokens.md) |
| [aero llm 22 word variations in claude tokenizer](aero_llm_22_word_variations_in_claude_tokenizer.md) | [Xem bài viết →](aero_llm_22_word_variations_in_claude_tokenizer.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
