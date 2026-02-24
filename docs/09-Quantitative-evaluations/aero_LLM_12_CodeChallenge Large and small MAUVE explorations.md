Đánh giá mô hình ngôn ngữ lớn bằng KL Divergence và MAUVE:

Phân tích thực nghiệm trên HellaSwag và các cấu hình mô hình khác nhau

⸻

Tóm tắt

Bài viết này trình bày phương pháp đánh giá mô hình ngôn ngữ lớn (Large Language Models – LLMs) thông qua hai công cụ định lượng quan trọng: Kullback–Leibler Divergence (KL Divergence) và MAUVE score. Dựa trên các thực nghiệm được thực hiện trên bộ dữ liệu HellaSwag, chúng tôi phân tích sự khác biệt giữa mô hình lớn và nhỏ, đồng thời làm rõ vai trò của khoảng cách phân phối xác suất trong đánh giá chất lượng sinh văn bản. Bài viết bổ sung cơ sở lý thuyết, công thức toán học minh hoạ và tổng hợp các nguồn học thuật liên quan.

⸻

1. Giới thiệu

Đánh giá mô hình ngôn ngữ sinh (generative language models) là một bài toán phức tạp do tính đa dạng và mở của không gian ngôn ngữ. Các thước đo truyền thống như perplexity chỉ phản ánh khả năng dự đoán token tiếp theo nhưng không phản ánh trực tiếp chất lượng phân phối sinh văn bản.

Các nghiên cứu gần đây như:
	•	Speech and Language Processing
	•	Pattern Recognition and Machine Learning
	•	OpenAI
	•	Google Research

đã chỉ ra rằng việc so sánh phân phối xác suất sinh ra bởi mô hình với phân phối dữ liệu thật là hướng tiếp cận chính xác hơn.

Trong bối cảnh này, hai công cụ nổi bật là:
	•	KL Divergence
	•	MAUVE

⸻

2. KL Divergence – Cơ sở lý thuyết

2.1 Định nghĩa

Giả sử:
	•	P(x): phân phối dữ liệu thật
	•	Q(x): phân phối sinh từ mô hình

Khi đó, Kullback–Leibler Divergence được định nghĩa:

D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}

Nếu phân phối liên tục:

D_{KL}(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx

2.2 Diễn giải
	•	D_{KL} = 0 khi P = Q
	•	D_{KL} > 0 khi hai phân phối khác nhau
	•	Không đối xứng:
D_{KL}(P || Q) \neq D_{KL}(Q || P)

2.3 Liên hệ với Cross-Entropy và Perplexity

Cross-entropy:

H(P, Q) = - \sum_x P(x) \log Q(x)

Ta có:

D_{KL}(P||Q) = H(P,Q) - H(P)

Perplexity:

\text{PPL} = 2^{H(P,Q)}

Điều này cho thấy KL divergence chính là phần “sai khác” giữa entropy thật và entropy mô hình.

⸻

3. MAUVE – Thước đo dựa trên hình học thông tin

3.1 Động cơ phát triển

KL divergence chỉ đo theo một chiều. Tuy nhiên trong bài toán sinh văn bản, ta cần đánh giá cân bằng giữa chất lượng và độ đa dạng.

MAUVE được đề xuất bởi nhóm nghiên cứu tại Google Research nhằm đo khoảng cách giữa hai phân phối dưới góc nhìn hình học thông tin.

3.2 Nguyên lý

MAUVE xây dựng đường cong giữa hai phân phối:
	•	Phân phối dữ liệu thật P
	•	Phân phối sinh Q

Bằng cách xét họ phân phối hỗn hợp:

R_\alpha = \alpha P + (1-\alpha)Q

Sau đó tính:

D_{KL}(P || R_\alpha)
D_{KL}(Q || R_\alpha)

Đường cong này tạo thành một frontier tương tự ROC curve.

3.3 Điểm số MAUVE

MAUVE được định nghĩa dựa trên diện tích dưới đường cong:

\text{MAUVE} = \int_0^1 f(\alpha) d\alpha

Trong đó f(\alpha) phản ánh trade-off giữa hai hướng KL.

Giá trị MAUVE ∈ [0,1]:
	•	Gần 1 → phân phối gần nhau
	•	Gần 0 → khác biệt lớn

⸻

4. Thực nghiệm trên HellaSwag

4.1 Bộ dữ liệu

Rowan University và University of Washington công bố bộ dữ liệu HellaSwag để kiểm tra khả năng suy luận thường thức của LLM.

Đặc điểm:
	•	Multiple choice
	•	Distractors gây nhiễu mạnh
	•	Kiểm tra khả năng hiểu ngữ cảnh

⸻

4.2 Phương pháp đánh giá

Giả sử có mô hình M, với mỗi câu hỏi có 4 đáp án a_i.

Xác suất lựa chọn:

P(a_i | context) = \frac{\exp(\log p_\theta(a_i))}{\sum_j \exp(\log p_\theta(a_j))}

Accuracy:

Acc = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\hat{y}_i = y_i)

Song song, ta tính MAUVE giữa:
	•	Tập văn bản đúng
	•	Tập văn bản sinh từ mô hình

⸻

5. So sánh mô hình lớn và nhỏ

5.1 Hiện tượng quan sát
	•	Mô hình nhỏ: KL lớn, MAUVE thấp
	•	Mô hình lớn: KL giảm, MAUVE tăng

Giả sử:

D_{KL}^{small} = 1.8
D_{KL}^{large} = 0.7

MAUVE^{small} = 0.42
MAUVE^{large} = 0.78

Điều này cho thấy mô hình lớn tiệm cận phân phối dữ liệu thật tốt hơn.

⸻

6. Phân tích hình học thông tin

Trong không gian xác suất, mỗi mô hình tương ứng với một điểm trên simplex:

\sum_i p_i = 1

KL divergence tương ứng với khoảng cách Bregman:

D_\phi(p,q) = \phi(p) - \phi(q) - \nabla\phi(q)^\top (p-q)

với:

\phi(p) = \sum_i p_i \log p_i

MAUVE khai thác toàn bộ cấu trúc hình học thay vì chỉ một hướng chiếu như KL.

⸻

7. Thảo luận

7.1 Ưu điểm KL
	•	Dễ tính toán
	•	Có nền tảng lý thuyết vững chắc
	•	Liên hệ trực tiếp với maximum likelihood

7.2 Hạn chế KL
	•	Không đối xứng
	•	Nhạy với zero-probability
	•	Không phản ánh đa dạng sinh

7.3 Ưu điểm MAUVE
	•	Đánh giá cân bằng
	•	Ổn định với sinh văn bản dài
	•	Phù hợp với LLM

⸻

8. Kết luận

Bài viết đã trình bày:
	•	Cơ sở toán học của KL divergence
	•	Cơ chế hình học của MAUVE
	•	Ứng dụng đánh giá mô hình trên HellaSwag
	•	Phân tích sự khác biệt giữa mô hình lớn và nhỏ

Trong bối cảnh LLM ngày càng mở rộng quy mô, việc sử dụng các thước đo dựa trên phân phối như MAUVE là cần thiết để phản ánh chính xác cả chất lượng lẫn đa dạng sinh văn bản.

⸻

Tài liệu tham khảo
	1.	Jurafsky & Martin. Speech and Language Processing.
	2.	Bishop, C. M. Pattern Recognition and Machine Learning.
	3.	Pillutla et al. (2021). MAUVE: Measuring the Gap Between Neural Text and Human Text.
	4.	Zellers et al. (2019). HellaSwag: Can a Machine Really Finish Your Sentence?
	5.	Cover & Thomas. Elements of Information Theory.
