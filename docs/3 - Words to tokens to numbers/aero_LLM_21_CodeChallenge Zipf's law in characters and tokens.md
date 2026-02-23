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

