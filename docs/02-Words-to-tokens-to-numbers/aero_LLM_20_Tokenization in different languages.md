Tokenization trong Các Ngôn ngữ Khác nhau:

Phân tích Toán học về Tỷ lệ Nén, Hình thái học và Ảnh hưởng đến Transformer

⸻

Tóm tắt

Dựa trên tài liệu đính kèm “Tokenization in Different Languages”, bài viết này phân tích sự khác biệt trong hành vi tokenization giữa các ngôn ngữ có đặc điểm hình thái và hệ chữ viết khác nhau. Chúng tôi xây dựng mô hình toán học cho tỷ lệ nén, entropy và độ dài chuỗi token, đồng thời phân tích tác động đến độ phức tạp tính toán trong kiến trúc Transformer. Các ví dụ minh họa được trình bày với tokenizer của BERT, mBERT và thư viện SentencePiece.

⸻

1. Giới thiệu

Tokenization ánh xạ chuỗi ký tự:

x \in \Sigma^*

thành chuỗi token:

\mathcal{T}(x) = (t_1, t_2, \dots, t_m)

Tuy nhiên, đặc điểm của ngôn ngữ (morphology, hệ chữ viết, khoảng trắng) ảnh hưởng mạnh đến:
	•	Độ dài trung bình của token
	•	Tỷ lệ nén
	•	Kích thước từ vựng
	•	Chi phí attention

⸻

2. Phân loại Ngôn ngữ theo Đặc điểm Tokenization

2.1 Ngôn ngữ phân tích (Analytic languages)

Ví dụ: tiếng Anh.
Từ thường tách bằng khoảng trắng.

Tokenizer như WordPiece (trong BERT) hoạt động hiệu quả.

⸻

2.2 Ngôn ngữ chắp dính (Agglutinative languages)

Ví dụ: tiếng Thổ Nhĩ Kỳ, tiếng Phần Lan.
Một từ có thể chứa nhiều hậu tố.

Nếu một từ có cấu trúc:

w = r + s_1 + s_2 + \dots + s_k

Độ dài ký tự tăng tuyến tính theo k.

Tokenizer phải chia nhỏ hơn:

m \uparrow

⸻

2.3 Ngôn ngữ không phân tách bằng khoảng trắng

Ví dụ: tiếng Trung.

Chuỗi ký tự:

x = c_1 c_2 \dots c_n

Mỗi ký tự có thể là một đơn vị nghĩa.

Trong trường hợp này:

R \approx 1

(trừ khi tokenizer gộp nhiều ký tự thành một token).

⸻

3. Mô hình Tỷ lệ Nén

Giả sử:
	•	n: số ký tự
	•	m: số token

3.1 Compression Ratio

R = \frac{n}{m}

Tương đương:

R = \mathbb{E}[L]

trong đó L là độ dài token.

⸻

3.2 So sánh giữa Ngôn ngữ

Giả sử:

R_{\text{EN}} = 4

R_{\text{ZH}} = 1.5

Chi phí attention:

C = O(m^2) = O\left(\left(\frac{n}{R}\right)^2\right)

Tỷ lệ chi phí:

\frac{C_{\text{ZH}}}{C_{\text{EN}}}
=
\left(\frac{R_{\text{EN}}}{R_{\text{ZH}}}\right)^2

Nếu R_{\text{EN}} = 4, R_{\text{ZH}} = 2:

= \left(\frac{4}{2}\right)^2 = 4

Tiếng Trung tốn gấp 4 lần chi phí attention cho cùng số ký tự.

⸻

4. Entropy theo Ngôn ngữ

Theo lý thuyết của Claude Shannon:

Entropy ký tự:

H_c = -\sum p(c)\log p(c)

Entropy token:

H_t = -\sum p(t)\log p(t)

Bảo toàn thông tin:

n H_c \approx m H_t

Suy ra:

R \approx \frac{H_t}{H_c}

Ngôn ngữ có bảng chữ cái lớn (như tiếng Trung) có:

H_c \uparrow
\Rightarrow R \downarrow

⸻

5. Tác động đến Mô hình Đa ngôn ngữ

5.1 mBERT

mBERT dùng chung từ vựng ~110k token cho nhiều ngôn ngữ.

Phân bố token không đồng đều:

p_{\text{lang}}(t) \neq \text{uniform}

Ngôn ngữ có ít dữ liệu → ít token chuyên biệt.

⸻

5.2 Tối ưu hóa Từ vựng

Bài toán:

\min_{V} \sum_{\ell} \alpha_\ell \left(\frac{n_\ell}{R_\ell}\right)^2 + \lambda |V|

Trong đó:
	•	\ell: ngôn ngữ
	•	\alpha_\ell: trọng số dữ liệu
	•	R_\ell: compression ratio của ngôn ngữ đó

⸻

6. Phân bố Độ dài Token

Gọi:

P_\ell(L=k)

Kỳ vọng:

\mathbb{E}_\ell[L] = \sum_k k P_\ell(L=k)

Ngôn ngữ chắp dính có:

\text{Var}(L) \uparrow

vì từ dài bị chia thành nhiều subword không đều.

⸻

7. Ảnh hưởng đến Độ phức tạp Huấn luyện

Transformer:

\text{Cost} = O(m^2 d)

Thay m = \frac{n}{R}:

\text{Cost} = O\left(\frac{n^2}{R^2} d\right)

Ngôn ngữ có R nhỏ làm tăng:
	•	Bộ nhớ GPU
	•	Thời gian huấn luyện
	•	Độ trễ suy luận

⸻

8. Phân tích Hình thái học

Nếu số hậu tố trung bình mỗi từ là k:

|w| \sim O(k)

Tokenizer tối ưu sẽ cố gắng học các đơn vị có xác suất cao:

\arg\max_{s} P(s)

Trong ngôn ngữ chắp dính, xác suất hậu tố phân tán → khó đạt nén cao.

⸻

9. Thảo luận

Khác biệt giữa các ngôn ngữ dẫn đến:
	1.	Compression ratio khác nhau
	2.	Chi phí attention khác nhau
	3.	Phân bố gradient khác nhau
	4.	Hiệu năng mô hình không đồng đều

Các hệ như Google và OpenAI phải cân bằng giữa:
	•	Bao phủ đa ngôn ngữ
	•	Kích thước từ vựng
	•	Chi phí tính toán

⸻

10. Kết luận

Tokenization phụ thuộc mạnh vào cấu trúc ngôn ngữ.

Các hệ thức quan trọng:

R = \frac{n}{m}

n H_c \approx m H_t

\text{Cost} = O\left(\frac{n^2}{R^2}\right)

Ngôn ngữ có compression ratio thấp sẽ chịu chi phí tính toán cao hơn trong Transformer.

Do đó, thiết kế tokenizer đa ngôn ngữ là bài toán tối ưu đa mục tiêu giữa:
	•	Entropy
	•	Kích thước từ vựng
	•	Phân bố dữ liệu
	•	Chi phí attention

⸻

Tài liệu tham khảo
	1.	Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	2.	Vaswani et al. (2017). Attention Is All You Need.
	3.	Shannon, C. (1948). A Mathematical Theory of Communication.
	4.	Kudo & Richardson (2018). SentencePiece.
	5.	Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units.

