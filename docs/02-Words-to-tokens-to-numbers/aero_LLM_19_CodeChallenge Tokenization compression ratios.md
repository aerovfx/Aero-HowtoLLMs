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

