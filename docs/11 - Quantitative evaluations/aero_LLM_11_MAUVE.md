MAUVE: Đo lường chất lượng và đa dạng của mô hình sinh ngôn ngữ thông qua hình học phân phối

Phân tích lý thuyết, công thức toán học và ứng dụng trong đánh giá LLM

⸻

Tóm tắt

Bài viết này trình bày phương pháp MAUVE – một thước đo hiện đại để đánh giá mô hình sinh ngôn ngữ dựa trên so sánh hình học giữa hai phân phối xác suất: phân phối dữ liệu thật và phân phối do mô hình sinh ra. Nội dung được phát triển dựa trên tài liệu đính kèm và mở rộng từ công trình của Krishna Pillutla et al. (2021), nền tảng lý thuyết phân kỳ thông tin của Solomon Kullback và Richard Leibler, cùng ứng dụng trong các mô hình ngôn ngữ lớn tại OpenAI.

⸻

1. Giới thiệu

Đánh giá mô hình sinh ngôn ngữ (text generation) là bài toán khó vì cần cân bằng:
	•	Chất lượng (quality): câu có hợp lý, trôi chảy?
	•	Đa dạng (diversity): mô hình có sinh lặp lại không?

Các thước đo truyền thống như:
	•	Perplexity
	•	BLEU
	•	ROUGE

không phản ánh đầy đủ sự khác biệt phân phối toàn cục.

MAUVE giải quyết bằng cách:
	•	So sánh phân phối embedding của văn bản thật và văn bản sinh
	•	Xây dựng đường cong trade-off giữa precision và recall

⸻

2. Cơ sở lý thuyết

Giả sử:
	•	P: phân phối dữ liệu thật
	•	Q: phân phối mô hình sinh

Ta muốn đo mức gần nhau giữa P và Q.

⸻

3. KL Divergence và hạn chế

Phân kỳ KL:

D_{KL}(P \| Q) = \sum_x P(x)\log \frac{P(x)}{Q(x)}

Vấn đề:
	•	Không đối xứng
	•	Không đo đồng thời precision và recall
	•	Không phản ánh hình học phân phối

⸻

4. Ý tưởng của MAUVE

MAUVE dựa trên họ phân kỳ:

D_\lambda(P \| Q)

Tạo phân phối trộn:

R_\lambda = \lambda P + (1-\lambda) Q

Sau đó tính:

D_{KL}(P \| R_\lambda)
\quad \text{và} \quad
D_{KL}(Q \| R_\lambda)

Khi thay đổi \lambda \in [0,1], ta thu được một đường cong trong không gian hai chiều.

⸻

5. Precision–Recall Curve trong không gian phân phối

MAUVE xây dựng đồ thị:

x(\lambda) = D_{KL}(P \| R_\lambda)
y(\lambda) = D_{KL}(Q \| R_\lambda)

Diện tích dưới đường cong này được chuẩn hoá thành điểm MAUVE:

MAUVE \in [0,1]

Giá trị gần 1 → phân phối gần nhau.

⸻

6. Triển khai thực tế

6.1 Embedding

Văn bản được ánh xạ vào không gian embedding:

x_i = f_{\text{LM}}(text_i)

Trong đó f_{\text{LM}} là encoder từ Transformer của Ashish Vaswani et al.

⸻

6.2 Rời rạc hoá không gian

Không gian embedding được phân cụm (k-means):

\min \sum_{i=1}^{N} ||x_i - c_{z_i}||^2

Sau đó ước lượng phân phối rời rạc trên các cluster.

⸻

7. So sánh với Perplexity

Perplexity:

PP = \exp\left(- \frac{1}{N} \sum \log P(w_i)\right)

Perplexity:
	•	Đo chất lượng token-level
	•	Không đo đa dạng toàn cục

MAUVE:
	•	Đo phân phối toàn văn bản
	•	Cân bằng precision–recall

⸻

8. Phân tích hình học

Giả sử:
	•	P = Q

→ Với mọi \lambda:

D_{KL}(P \| R_\lambda) = D_{KL}(Q \| R_\lambda)

→ MAUVE = 1

Nếu:
	•	Q collapse (mode collapse)

→ D_{KL}(P \| Q) lớn
→ MAUVE giảm mạnh.

⸻

9. Phân tích giới hạn

9.1 Khi Q thiếu đa dạng

Recall thấp:

D_{KL}(P \| R_\lambda) \uparrow

⸻

9.2 Khi Q sinh nhiễu

Precision thấp:

D_{KL}(Q \| R_\lambda) \uparrow

⸻

10. So sánh với Jensen–Shannon Divergence

JSD:

JSD(P \| Q) =
\frac{1}{2} D_{KL}(P \| M)
+
\frac{1}{2} D_{KL}(Q \| M)

với:

M = \frac{1}{2}(P+Q)

MAUVE có thể xem như mở rộng hình học của JSD khi thay đổi \lambda.

⸻

11. Ý nghĩa trong đánh giá LLM

MAUVE đặc biệt hữu ích khi:
	•	So sánh hai mô hình sinh văn bản
	•	Đánh giá fine-tuning
	•	Đo hiệu quả RLHF

Trong pipeline huấn luyện tại OpenAI, MAUVE có thể bổ sung cho perplexity.

⸻

12. Hạn chế
	1.	Phụ thuộc embedding model
	2.	Phụ thuộc số cluster
	3.	Tốn chi phí tính toán

⸻

13. Kết luận

MAUVE là thước đo tiên tiến:
	•	Dựa trên hình học phân phối
	•	Cân bằng chất lượng và đa dạng
	•	Khắc phục hạn chế của perplexity

Nó kết nối lý thuyết phân kỳ KL với đánh giá mô hình sinh hiện đại.

⸻

Tài liệu tham khảo
	1.	Pillutla, K. et al. (2021). MAUVE: Measuring the Gap Between Neural Text and Human Text.
	2.	Kullback, S., Leibler, R. (1951). On Information and Sufficiency.
	3.	Shannon, C. (1948). A Mathematical Theory of Communication.
	4.	Vaswani, A. et al. (2017). Attention is All You Need.
	5.	Goodfellow, I. et al. (2016). Deep Learning.

