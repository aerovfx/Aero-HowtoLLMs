
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [02 Words to tokens to numbers](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
Phân tích Thống kê Số lượng Token theo Độ dài Subword (Phần 2): Mô hình hóa Toán học và Hàm Phân bố

⸻

Tóm tắt

Bài báo này tiếp tục phân tích thống kê số lượng token theo độ dài subword trong bộ tokenizer của GPT-4, dựa trên dữ liệu thực nghiệm từ tài liệu đính kèm (phần 2). Chúng tôi xây dựng mô hình toán học cho phân bố độ dài, kiểm định giả thuyết phân bố mũ và luật Zipf, đồng thời phân tích tác động của cấu trúc token đến độ phức tạp tính toán trong kiến trúc Transformer của OpenAI. Kết quả cho thấy phân bố độ dài subword có xu hướng suy giảm phi tuyến, gần với hàm mũ hoặc log-linear, và có mối liên hệ chặt chẽ với entropy hệ token.

⸻

1. Giới thiệu

Trong các mô hình ngôn ngữ lớn (LLMs), tokenization là bước ánh xạ văn bản thô thành chuỗi token rời rạc:

S = (c_1, c_2, ..., c_n)

T = (t_1, t_2, ..., t_m)

Với:

m \le n

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

P(L = k) = \frac{N_k}{|V|}

Và:

\sum_{k=1}^{K_{\max}} P(L = k) = 1

⸻

2.2 Giả thuyết phân bố mũ

Dữ liệu thực nghiệm cho thấy:

N_k \approx Ae^{-\lambda k}

Suy ra:

P(L = k) = \frac{Ae^{-\lambda k}}{\sum_{j=1}^{K_{\max}} Ae^{-\lambda j}}

Chuẩn hóa:

P(L = k) = (1 - e^{-\lambda}) e^{-\lambda (k-1)}

Đây là phân bố hình học rời rạc.

⸻

2.3 Kỳ vọng và Phương sai

Kỳ vọng:

\mathbb{E}[L] = \frac{1}{1 - e^{-\lambda}}

Phương sai:

\mathrm{Var}(L) = \frac{e^{-\lambda}}{(1 - e^{-\lambda})^2}

Điều này cho thấy khi \lambda nhỏ:
	•	Đuôi phân bố dài hơn
	•	Tồn tại nhiều token dài

⸻

3. Liên hệ với Luật Zipf

Tần suất token theo thứ hạng:

f(r) \propto \frac{1}{r^\alpha}

Trong đó:
	•	r: thứ hạng
	•	\alpha \approx 1

Kết hợp hai quan sát:
	•	Token ngắn → tần suất cao
	•	Token dài → tần suất thấp

Ta có mô hình kết hợp:

P(t) \propto e^{-\beta \ell(t)} \cdot \frac{1}{r^\alpha}

⸻

4. Ảnh hưởng đến Độ dài Chuỗi và Chi phí Attention

Giả sử văn bản có tổng số ký tự n.

Số token:

m = \frac{n}{\mathbb{E}[L]}

Self-attention có độ phức tạp:

O(m^2)

Thay vào:

O\left(\left(\frac{n}{\mathbb{E}[L]}\right)^2\right)

Do đó:
	•	Nếu \mathbb{E}[L] \uparrow \Rightarrow m \downarrow \Rightarrow \text{Cost} \downarrow
	•	Nếu token quá dài → vocabulary lớn → tăng chi phí embedding

⸻

5. Entropy của Hệ Token

Entropy:

H = - \sum_{t \in V} P(t) \log P(t)

Thay mô hình mũ:

H \approx - \sum_{k} P(L=k) \log P(L=k)

Với phân bố hình học:

H = - \sum_{k=1}^{\infty} (1-q) q^{k-1} \log[(1-q) q^{k-1}]

Trong đó:

q = e^{-\lambda}

Entropy tối ưu khi:
	•	Không quá tập trung vào token cực ngắn
	•	Không quá phân tán ở token dài

⸻

6. Kiểm định Phù hợp Mô hình

Để kiểm tra giả thuyết phân bố mũ, có thể sử dụng:

6.1 Hồi quy log-linear

\log N_k = \log A - \lambda k

Nếu đồ thị \log N_k theo k tuyến tính → xác nhận mô hình mũ.

⸻

6.2 Kiểm định Chi-square

\chi^2 = \sum_{k} \frac{(N_k - \hat{N}_k)^2}{\hat{N}_k}

So sánh với phân bố lý thuyết.

⸻

7. Hàm Tối ưu Hóa Ngầm trong Tokenizer

Tokenizer BPE thực chất tối ưu xấp xỉ:

\min_{V} \left( \mathbb{E}[m] + \lambda |V| \right)

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

P(L = k) \sim e^{-\lambda k}

Tác động trực tiếp đến:

m = \frac{n}{\mathbb{E}[L]}

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
| [Tại sao văn bản cần được đánh số?](aero_LLM_01_Why text needs to be numbered.md) | [Xem bài viết →](aero_LLM_01_Why text needs to be numbered.md) |
| [Phân tích và chuyển đổi văn bản thành chuỗi token được đánh số: Cơ sở toán học và ứng dụng trong mô hình ngôn ngữ lớn](aero_LLM_02_Parsing text to numbered tokens.md) | [Xem bài viết →](aero_LLM_02_Parsing text to numbered tokens.md) |
| [Tạo và trực quan hóa Token trong mô hình ngôn ngữ lớn: Cơ sở toán học và phân tích biểu diễn](aero_LLM_03_CodeChallenge Create and visualize tokens (part 1).md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Create and visualize tokens (part 1).md) |
| [Tạo và trực quan hóa Token (Phần 2): Phân tích hình học không gian embedding và Attention Map trong mô hình Transformer](aero_LLM_04_CodeChallenge Create and visualize tokens (part 2).md) | [Xem bài viết →](aero_LLM_04_CodeChallenge Create and visualize tokens (part 2).md) |
| [Chuẩn bị văn bản cho Tokenization trong mô hình Transformer: Cơ sở lý thuyết và phân tích toán học](aero_LLM_05_Preparing text for tokenization.md) | [Xem bài viết →](aero_LLM_05_Preparing text for tokenization.md) |
| [Phân tích quy trình Tokenization qua ví dụ *The Time Machine*: Cơ sở thuật toán và mô hình hóa toán học](aero_LLM_06_CodeChallenge Tokenizing The Time Machine.md) | [Xem bài viết →](aero_LLM_06_CodeChallenge Tokenizing The Time Machine.md) |
| [So sánh Tokenization mức ký tự, từ và subword: Phân tích lý thuyết và mô hình toán học](aero_LLM_07_Tokenizing characters vs. subwords vs. words.md) | [Xem bài viết →](aero_LLM_07_Tokenizing characters vs. subwords vs. words.md) |
| [aero_LLM_08_Byte-pair encoding algorithm.md](aero_LLM_08_Byte-pair encoding algorithm.md) | [Xem bài viết →](aero_LLM_08_Byte-pair encoding algorithm.md) |
| [Thuật toán Byte Pair Encoding (BPE) và Bài toán Tối ưu Kích thước Từ vựng trong Mô hình Ngôn ngữ](aero_LLM_09_CodeChallenge Byte-pair encoding to a desired vocab size.md) | [Xem bài viết →](aero_LLM_09_CodeChallenge Byte-pair encoding to a desired vocab size.md) |
| [aero_LLM_10_Exploring ChatGPT4's tokenizer.md](aero_LLM_10_Exploring ChatGPT4's tokenizer.md) | [Xem bài viết →](aero_LLM_10_Exploring ChatGPT4's tokenizer.md) |
| [aero_LLM_11_CodeChallenge Token count by subword length (part 1).md](aero_LLM_11_CodeChallenge Token count by subword length (part 1).md) | [Xem bài viết →](aero_LLM_11_CodeChallenge Token count by subword length (part 1).md) |
| 📌 **[aero_LLM_12_CodeChallenge Token count by subword length (part 2).md](aero_LLM_12_CodeChallenge Token count by subword length (part 2).md)** | [Xem bài viết →](aero_LLM_12_CodeChallenge Token count by subword length (part 2).md) |
| [aero_LLM_13_How many rs in strawberry.md](aero_LLM_13_How many rs in strawberry.md) | [Xem bài viết →](aero_LLM_13_How many rs in strawberry.md) |
| [aero_LLM_14_CodeChallenge Create your algorithmic rapper name ).md](aero_LLM_14_CodeChallenge Create your algorithmic rapper name ).md) | [Xem bài viết →](aero_LLM_14_CodeChallenge Create your algorithmic rapper name ).md) |
| [aero_LLM_15_Tokenization in BERT.md](aero_LLM_15_Tokenization in BERT.md) | [Xem bài viết →](aero_LLM_15_Tokenization in BERT.md) |
| [aero_LLM_16_CodeChallenge Character counts in BERT tokens.md](aero_LLM_16_CodeChallenge Character counts in BERT tokens.md) | [Xem bài viết →](aero_LLM_16_CodeChallenge Character counts in BERT tokens.md) |
| [aero_LLM_17_Translating between tokenizers.md](aero_LLM_17_Translating between tokenizers.md) | [Xem bài viết →](aero_LLM_17_Translating between tokenizers.md) |
| [aero_LLM_18_CodeChallenge More on token translation.md](aero_LLM_18_CodeChallenge More on token translation.md) | [Xem bài viết →](aero_LLM_18_CodeChallenge More on token translation.md) |
| [aero_LLM_19_CodeChallenge Tokenization compression ratios.md](aero_LLM_19_CodeChallenge Tokenization compression ratios.md) | [Xem bài viết →](aero_LLM_19_CodeChallenge Tokenization compression ratios.md) |
| [aero_LLM_20_Tokenization in different languages.md](aero_LLM_20_Tokenization in different languages.md) | [Xem bài viết →](aero_LLM_20_Tokenization in different languages.md) |
| [aero_LLM_21_CodeChallenge Zipf's law in characters and tokens.md](aero_LLM_21_CodeChallenge Zipf's law in characters and tokens.md) | [Xem bài viết →](aero_LLM_21_CodeChallenge Zipf's law in characters and tokens.md) |
| [aero_LLM_22_Word variations in Claude tokenizer.md](aero_LLM_22_Word variations in Claude tokenizer.md) | [Xem bài viết →](aero_LLM_22_Word variations in Claude tokenizer.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
