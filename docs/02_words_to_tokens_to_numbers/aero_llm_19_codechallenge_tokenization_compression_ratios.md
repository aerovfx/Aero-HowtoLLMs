
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
Phân tích Tỷ lệ Nén trong Tokenization:

Mô hình Toán học và Ảnh hưởng đến Hiệu năng Mô hình Ngôn ngữ

⸻

Tóm tắt

Dựa trên tài liệu đính kèm “Tokenization Compression Ratios”, bài viết này phân tích tỷ lệ nén (compression ratio) của các phương pháp tokenization trong mô hình ngôn ngữ lớn (LLMs). Chúng tôi xây dựng mô hình toán học cho tỷ lệ nén giữa không gian ký tự và không gian token, phân tích mối quan hệ với entropy và độ phức tạp self-attention, đồng thời so sánh các cơ chế token hóa như WordPiece và Byte Pair Encoding (BPE). Các ví dụ minh họa được trình bày với tokenizer của BERT và GPT-2.

⸻

1. Giới thiệu

Tokenization là quá trình ánh xạ một chuỗi ký tự:

x \in \Sigma^*

thành chuỗi token:

\mathcal{T}(x) = (t_1, t_2, \dots, t_m)

Tỷ lệ nén của tokenizer phản ánh mức độ giảm số đơn vị biểu diễn khi chuyển từ ký tự sang token.

⸻

2. Định nghĩa Tỷ lệ Nén

Giả sử:
	•	Văn bản có n ký tự
	•	Sau tokenization thu được m token

2.1 Compression Ratio

R = \frac{n}{m}

Nếu R > 1, tokenization đạt hiệu ứng nén.

⸻

2.2 Độ dài Token Trung bình

Gọi L_i là số ký tự trong token t_i.

\bar{L} = \frac{1}{m} \sum_{i=1}^{m} L_i

Ta có:

n = \sum_{i=1}^{m} L_i

Suy ra:

R = \bar{L}

Tỷ lệ nén chính là độ dài ký tự trung bình trên mỗi token.

⸻

3. Phân tích Xác suất

Gọi P(L=k) là xác suất token có độ dài k.

Kỳ vọng:

\mathbb{E}[L] = \sum_{k} k P(L=k)

Tỷ lệ nén trung bình:

R = \mathbb{E}[L]

Nếu phân bố độ dài tuân theo phân bố hình học:

P(L=k) = (1-q)q^{k-1}

thì:

\mathbb{E}[L] = \frac{1}{1-q}

⸻

4. Liên hệ với Entropy

Theo lý thuyết của Claude Shannon (1948), entropy của nguồn ký tự:

H_c = -\sum_{c \in \Sigma} p(c)\log p(c)

Entropy trên token:

H_t = -\sum_{t \in V} p(t)\log p(t)

Tỷ lệ nén lý thuyết tối ưu:

R_{\text{opt}} = \frac{H_c}{H_t}

Nếu tokenizer tối ưu theo nghĩa thông tin, thì:

m H_t \approx n H_c

⸻

5. Ảnh hưởng đến Self-Attention

Trong kiến trúc Transformer:

\text{Cost} = O(m^2)

Thay m = \frac{n}{R}:

\text{Cost} = O\left(\left(\frac{n}{R}\right)^2\right)

Do đó:
	•	R \uparrow \Rightarrow chi phí giảm theo bình phương.

Ví dụ:
	•	Nếu R = 4, chi phí giảm 16 lần so với character-level.

⸻

6. So sánh Các Phương pháp Tokenization

6.1 WordPiece

Áp dụng trong BERT.

Tối ưu xác suất chuỗi subword:

\arg\max_{s_1,\dots,s_k} \prod_i P(s_i)

Có xu hướng tạo token trung bình 3–5 ký tự.

⸻

6.2 Byte Pair Encoding (BPE)

Sử dụng trong GPT-2 bởi OpenAI.

Thuật toán lặp:

(u,v) = \arg\max \text{freq}(uv)

Gộp cặp xuất hiện nhiều nhất.

⸻

6.3 Character-level

R = 1

Không nén → chi phí attention cao nhất.

⸻

7. Phân tích Giới hạn Lý thuyết

Giả sử kích thước từ vựng |V|.

Dung lượng embedding:

W \in \mathbb{R}^{|V| \times d}

Tổng tham số:

|V|d

Bài toán tối ưu đa mục tiêu:

\min_{V} \left( \frac{n}{R} \right)^2 + \lambda |V|

Trong đó:
	•	Thành phần đầu: chi phí attention
	•	Thành phần sau: chi phí bộ nhớ embedding

⸻

8. Phân tích Tỷ lệ Nén Thực nghiệm

Trong thực tế:
	•	Văn bản tiếng Anh: R \approx 3-4
	•	Văn bản có nhiều ký tự Unicode: R thấp hơn
	•	Ngôn ngữ chắp dính (agglutinative): R biến thiên mạnh

Do đó:

R = f(\text{ngôn ngữ}, |V|, thuật toán)

⸻

9. Bàn luận

Tokenization đóng vai trò như cơ chế nén tiền xử lý cho Transformer.

Có thể xem tokenization như bài toán mã hóa:

\Sigma^* \rightarrow V^*

Mục tiêu:
	1.	Giảm độ dài chuỗi (tăng R)
	2.	Giữ entropy thông tin
	3.	Hạn chế tăng kích thước từ vựng

Sự cân bằng này giải thích vì sao các hệ như Google và OpenAI chọn từ vựng khoảng 30k–50k token.

⸻

10. Kết luận

Tỷ lệ nén trong tokenization được xác định bởi:

R = \frac{n}{m} = \mathbb{E}[L]

Ảnh hưởng trực tiếp đến:

\text{Attention Cost} = O\left(\left(\frac{n}{R}\right)^2\right)

Và chịu ràng buộc bởi:

m H_t \approx n H_c

Tokenization có thể được xem như bước nén thông tin có kiểm soát nhằm tối ưu hóa hiệu năng và chi phí tính toán của mô hình ngôn ngữ.

⸻

Tài liệu tham khảo
	1.	Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	2.	Radford et al. (2019). GPT-2: Language Models are Unsupervised Multitask Learners.
	3.	Vaswani et al. (2017). Attention Is All You Need.
	4.	Shannon, C. (1948). A Mathematical Theory of Communication.
	5.	Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
	6.	Kudo & Richardson (2018). SentencePiece: A simple and language independent subword tokenizer.
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
| 📌 **[aero llm 19 codechallenge tokenization compression ratios](aero_llm_19_codechallenge_tokenization_compression_ratios.md)** | [Xem bài viết →](aero_llm_19_codechallenge_tokenization_compression_ratios.md) |
| [aero llm 20 tokenization in different languages](aero_llm_20_tokenization_in_different_languages.md) | [Xem bài viết →](aero_llm_20_tokenization_in_different_languages.md) |
| [aero llm 21 codechallenge zipf s law in characters and tokens](aero_llm_21_codechallenge_zipf_s_law_in_characters_and_tokens.md) | [Xem bài viết →](aero_llm_21_codechallenge_zipf_s_law_in_characters_and_tokens.md) |
| [aero llm 22 word variations in claude tokenizer](aero_llm_22_word_variations_in_claude_tokenizer.md) | [Xem bài viết →](aero_llm_22_word_variations_in_claude_tokenizer.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
