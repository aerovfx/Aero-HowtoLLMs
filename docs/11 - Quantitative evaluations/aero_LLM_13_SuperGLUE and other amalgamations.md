Đánh giá mô hình ngôn ngữ lớn qua SuperGLUE và các bộ benchmark tổng hợp

Phân tích lý thuyết, phương pháp và công thức toán học minh hoạ

⸻

Tóm tắt

Bài viết này trình bày cơ sở lý thuyết và thực nghiệm liên quan đến việc đánh giá mô hình ngôn ngữ lớn (Large Language Models – LLMs) thông qua các bộ benchmark tổng hợp, đặc biệt là SuperGLUE. Chúng tôi phân tích cấu trúc bài toán, cơ chế chấm điểm, các chỉ số thống kê thường dùng và mối liên hệ giữa tối ưu hoá hàm mất mát với hiệu năng tổng quát. Ngoài ra, bài viết mở rộng so sánh với các benchmark khác như GLUE, HellaSwag và các bộ đánh giá suy luận ngữ nghĩa hiện đại.

⸻

1. Giới thiệu

Đánh giá mô hình NLP truyền thống thường dựa trên các tập dữ liệu riêng lẻ cho từng nhiệm vụ: phân loại văn bản, suy luận ngôn ngữ tự nhiên, hỏi đáp, v.v. Tuy nhiên, sự phát triển nhanh chóng của LLM đòi hỏi các benchmark tổng hợp có độ khó cao hơn.

Năm 2018, nhóm nghiên cứu từ Stanford University và New York University công bố GLUE. Sau đó, để khắc phục hiện tượng mô hình đạt điểm cao nhưng chưa thực sự hiểu ngôn ngữ, nhóm tác giả giới thiệu:
	•	SuperGLUE (2019)

SuperGLUE được thiết kế nhằm:
	•	Tăng độ khó
	•	Giảm hiện tượng shortcut learning
	•	Đánh giá suy luận ngữ nghĩa sâu hơn

⸻

2. Cấu trúc của SuperGLUE

SuperGLUE bao gồm nhiều nhiệm vụ:

Nhiệm vụ	Mô tả	Loại bài toán
BoolQ	Trả lời Yes/No	Binary classification
CB	CommitmentBank	Entailment
COPA	Causal reasoning	Multiple choice
MultiRC	Multi-sentence reasoning	Multi-label
ReCoRD	Reading comprehension	Span prediction
WiC	Word sense disambiguation	Binary

Mỗi nhiệm vụ có hàm đánh giá riêng, nhưng điểm tổng hợp được chuẩn hoá và tính trung bình.

⸻

3. Cơ sở toán học của đánh giá phân loại

3.1 Xác suất dự đoán

Với một đầu vào x, mô hình tham số \theta sinh xác suất:

P_\theta(y|x) = \frac{\exp(z_y)}{\sum_{k=1}^K \exp(z_k)}

Trong đó:
	•	z_k là logit
	•	K là số lớp

Đây là hàm Softmax.

⸻

3.2 Hàm mất mát Cross-Entropy

Với nhãn thật y:

\mathcal{L}(\theta) = - \sum_{i=1}^N \log P_\theta(y_i | x_i)

Dưới dạng kỳ vọng:

\mathcal{L} = \mathbb{E}_{(x,y)\sim D}[-\log P_\theta(y|x)]

Tối thiểu hoá hàm này tương đương tối thiểu hoá KL divergence giữa phân phối thật và phân phối mô hình:

D_{KL}(P_{data} || P_\theta)

⸻

3.3 Accuracy

Acc = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\hat{y}_i = y_i)

⸻

3.4 F1-score (cho MultiRC)

Precision:

P = \frac{TP}{TP+FP}

Recall:

R = \frac{TP}{TP+FN}

F1 = \frac{2PR}{P+R}

⸻

4. SuperGLUE như một bài toán tổng hợp (Amalgamation Benchmark)

SuperGLUE không chỉ là một tập dữ liệu mà là một hệ thống hợp nhất (amalgamation) của nhiều dạng bài toán:

Score_{overall} = \frac{1}{M} \sum_{i=1}^M Score_i

Trong đó:
	•	M là số nhiệm vụ
	•	Score_i có thể là Accuracy, F1, EM (Exact Match)

Điều này tạo ra một không gian đánh giá đa chiều.

⸻

5. So sánh với GLUE và HellaSwag
	•	GLUE Benchmark
	•	HellaSwag

GLUE chủ yếu kiểm tra suy luận câu-ngắn.
HellaSwag tập trung vào hoàn thành câu thường thức.
SuperGLUE tăng độ phức tạp về:
	•	Lập luận nhân quả
	•	Ngữ nghĩa ngữ cảnh dài
	•	Giải tham chiếu

⸻

6. Phân tích thống kê hiệu năng mô hình

Giả sử mô hình A và B có điểm:

\mu_A = 89.2, \quad \mu_B = 91.5

Kiểm định t-test:

t = \frac{\mu_A - \mu_B}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}

Nếu:

p < 0.05

→ sự khác biệt có ý nghĩa thống kê.

⸻

7. Liên hệ với lý thuyết tổng quát hóa (Generalization)

Theo lý thuyết học thống kê trong:
	•	Pattern Recognition and Machine Learning
	•	Elements of Information Theory

Sai số tổng quát:

R(\theta) = \mathbb{E}_{(x,y)\sim P}[\ell(f_\theta(x), y)]

Sai số thực nghiệm:

\hat{R}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)

Bất đẳng thức tổng quát hóa:

R(\theta) \le \hat{R}(\theta) + O\left(\sqrt{\frac{VC}{N}}\right)

SuperGLUE có vai trò ước lượng gần đúng R(\theta).

⸻

8. Xu hướng hiện đại: Beyond SuperGLUE

Các tổ chức như:
	•	OpenAI
	•	Anthropic
	•	DeepMind

đang chuyển sang:
	•	Evaluation theo capability scaling
	•	Alignment benchmark
	•	Long-context reasoning
	•	Agentic evaluation

⸻

9. Thảo luận

Ưu điểm của SuperGLUE
	•	Chuẩn hoá cao
	•	Bao phủ nhiều dạng suy luận
	•	Phân biệt rõ mô hình mạnh/yếu

Hạn chế
	•	Dễ bị overfitting leaderboard
	•	Không đo creativity
	•	Không đo alignment hay an toàn

⸻

10. Kết luận

SuperGLUE là bước tiến quan trọng trong đánh giá LLM, cung cấp:
	•	Hệ thống benchmark tổng hợp
	•	Độ khó cao
	•	Đánh giá đa nhiệm vụ

Tuy nhiên, trong bối cảnh LLM hiện đại với hàng trăm tỷ tham số, việc đánh giá cần kết hợp:
	•	Benchmark tổng hợp
	•	Phân tích phân phối xác suất
	•	Thước đo hình học thông tin
	•	Đánh giá hành vi (behavioral evaluation)

⸻

Tài liệu tham khảo
	1.	Wang et al. (2018). GLUE: A Multi-Task Benchmark and Analysis Platform.
	2.	Wang et al. (2019). SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems.
	3.	Zellers et al. (2019). HellaSwag.
	4.	Bishop (2006). Pattern Recognition and Machine Learning.
	5.	Cover & Thomas (2006). Elements of Information Theory.
