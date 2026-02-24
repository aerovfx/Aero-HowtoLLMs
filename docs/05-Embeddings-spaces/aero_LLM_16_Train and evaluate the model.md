# Huấn luyện và Đánh giá Mô hình Học Máy: Cơ sở Lý thuyết và Thực tiễn

Tóm tắt

Huấn luyện và đánh giá mô hình là hai giai đoạn cốt lõi trong quy trình phát triển hệ thống học máy (Machine Learning – ML). Bài viết này trình bày cơ sở toán học của quá trình huấn luyện, các phương pháp tối ưu hóa, chiến lược chia dữ liệu, và các chỉ số đánh giá phổ biến. Đồng thời, bài viết bổ sung các công thức toán học nền tảng như hàm mất mát, gradient descent, entropy chéo và các thước đo hiệu suất. Nội dung được xây dựng theo chuẩn khoa học, kết hợp lý thuyết từ các tài liệu kinh điển trong lĩnh vực ML.

⸻

1. Giới thiệu

Trong học máy, mục tiêu của mô hình là xấp xỉ một hàm ánh xạ:

f_\theta: X \rightarrow Y

Trong đó:
	•	X là không gian đầu vào
	•	Y là không gian đầu ra
	•	\theta là tập tham số của mô hình

Quá trình huấn luyện nhằm tìm ra bộ tham số \theta^* sao cho hàm mất mát được tối thiểu hóa:

\theta^* = \arg\min_\theta \mathcal{L}(\theta)

⸻

2. Cơ sở Toán học của Huấn luyện Mô hình

2.1 Hàm mất mát (Loss Function)

Tùy theo loại bài toán, hàm mất mát được xác định khác nhau.

(a) Hồi quy – Mean Squared Error (MSE)

\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2

Trong đó:
	•	y_i là giá trị thực
	•	\hat{y}_i là giá trị dự đoán

⸻

(b) Phân loại – Cross Entropy Loss

\mathcal{L}_{CE} = - \sum_{i=1}^{n} y_i \log(\hat{y}_i)

Cross-entropy có nguồn gốc từ lý thuyết thông tin của Shannon (1948).

⸻

2.2 Tối ưu hóa bằng Gradient Descent

Thuật toán cập nhật tham số:

\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)

Trong đó:
	•	\eta là learning rate
	•	\nabla_\theta \mathcal{L} là gradient

Các biến thể:
	•	Batch Gradient Descent
	•	Stochastic Gradient Descent (SGD)
	•	Adam Optimizer:

m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2

Adam được đề xuất bởi Kingma & Ba (2015).

⸻

3. Quy trình Huấn luyện

3.1 Chia tập dữ liệu

Thông thường:
	•	Training set: 70–80%
	•	Validation set: 10–15%
	•	Test set: 10–15%

Mô hình được tối ưu trên training set, điều chỉnh siêu tham số trên validation set và đánh giá cuối cùng trên test set.

⸻

3.2 Overfitting và Underfitting

Overfitting

Mô hình học quá sát dữ liệu huấn luyện:

\mathcal{L}_{train} \ll \mathcal{L}_{test}

Giải pháp:
	•	Regularization:
\mathcal{L}_{reg} = \mathcal{L} + \lambda ||\theta||^2
	•	Dropout
	•	Early stopping

⸻

4. Đánh giá Mô hình

4.1 Bài toán Phân loại

(a) Accuracy

Accuracy = \frac{TP + TN}{TP + TN + FP + FN}

⸻

(b) Precision & Recall

Precision = \frac{TP}{TP + FP}

Recall = \frac{TP}{TP + FN}

⸻

(c) F1-score

F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}

⸻

(d) ROC-AUC

Diện tích dưới đường cong ROC đo khả năng phân biệt hai lớp.

⸻

4.2 Bài toán Hồi quy

(a) Mean Absolute Error (MAE)

MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|

(b) R² Score

R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}

⸻

5. Đánh giá Thực nghiệm

Trong quá trình huấn luyện:
	•	Theo dõi loss curve
	•	So sánh train vs validation
	•	Sử dụng confusion matrix
	•	Cross-validation:

CV = \frac{1}{k} \sum_{i=1}^{k} \mathcal{L}_i

⸻

6. Thảo luận

Huấn luyện và đánh giá mô hình không chỉ là quá trình kỹ thuật mà còn là bài toán tối ưu hóa thống kê. Sai lệch (bias) và phương sai (variance) đóng vai trò quan trọng:

\mathbb{E}[(y - \hat{f}(x))^2] = Bias^2 + Variance + \sigma^2

Cân bằng bias-variance là chìa khóa xây dựng mô hình tổng quát hóa tốt.

⸻

7. Kết luận

Quá trình huấn luyện và đánh giá mô hình dựa trên nền tảng toán học vững chắc của:
	•	Tối ưu hóa
	•	Xác suất thống kê
	•	Lý thuyết thông tin

Việc lựa chọn hàm mất mát, thuật toán tối ưu và chỉ số đánh giá phù hợp quyết định trực tiếp đến hiệu năng hệ thống. Trong bối cảnh AI hiện đại, đặc biệt với các mô hình lớn (Large Language Models), quy trình huấn luyện còn mở rộng sang:
	•	Fine-tuning
	•	Transfer learning
	•	Reinforcement Learning from Human Feedback (RLHF)

⸻

Tài liệu tham khảo
	1.	Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
	2.	Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
	3.	Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization.
	4.	Vapnik, V. (1998). Statistical Learning Theory. Wiley.
	5.	Shannon, C. E. (1948). A Mathematical Theory of Communication.
