# Hứa Hẹn và Thách Thức của Đánh Giá Định Lượng trong Mô Hình Học Máy

Tóm tắt

Đánh giá định lượng (quantitative evaluation) đóng vai trò trung tâm trong việc đo lường hiệu năng của các mô hình học máy, đặc biệt trong xử lý ngôn ngữ tự nhiên (NLP) và các hệ thống AI hiện đại. Tuy nhiên, các chỉ số định lượng không phải lúc nào cũng phản ánh chính xác năng lực thực tế của mô hình. Bài viết này phân tích cơ sở toán học của các thước đo phổ biến, đồng thời chỉ ra những giới hạn nội tại của đánh giá định lượng. Nội dung được xây dựng dựa trên bài giảng “Promises and Challenges of Quantitative Evaluations” và mở rộng từ các công trình của Christopher D. Manning, Colin Raffel và George Box.

⸻

1. Giới thiệu

Trong học máy, ta xây dựng một mô hình:

f_\theta : X \rightarrow Y

Mục tiêu là tìm tham số \theta tối ưu:

\theta^* = \arg\min_\theta \mathbb{E}_{(x,y)\sim D}
\left[
\mathcal{L}(f_\theta(x), y)
\right]

Đánh giá định lượng nhằm ước lượng kỳ vọng này thông qua tập kiểm tra hữu hạn.

⸻

2. Cơ sở Toán học của Đánh giá Định lượng

2.1 Ước lượng thực nghiệm (Empirical Risk)

Với tập test gồm n mẫu:

\hat{R}(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)

Theo luật số lớn:

\hat{R}(\theta) \rightarrow R(\theta)
\quad
\text{khi}
\quad
n \rightarrow \infty

⸻

2.2 Các chỉ số phổ biến

(a) Accuracy

Accuracy =
\frac{TP + TN}
{TP + TN + FP + FN}

Nhược điểm: mất cân bằng lớp (class imbalance).

⸻

(b) Cross-Entropy

\mathcal{L}_{CE}
=
- \sum_{i=1}^{n}
y_i \log(\hat{y}_i)

Liên hệ với entropy của Shannon:

H(p) =
- \sum p(x)\log p(x)

⸻

(c) BLEU Score (dịch máy)

BLEU =
BP \cdot
\exp
\left(
\sum_{n=1}^{N}
w_n \log p_n
\right)

Trong đó:
	•	p_n là precision n-gram
	•	BP là brevity penalty

⸻

(d) ROUGE Score (tóm tắt văn bản)

ROUGE-N =
\frac{
\sum_{gram_n \in Reference}
Count_{match}
}
{
\sum_{gram_n \in Reference}
Count
}

⸻

3. Hứa Hẹn của Đánh Giá Định Lượng

3.1 Tính tái lập (Reproducibility)

Khi hai mô hình được đánh giá trên cùng benchmark:

Score_A > Score_B
\Rightarrow
A \text{ tốt hơn } B

Giả định:
	•	Tập dữ liệu đại diện tốt cho phân phối thực tế.

⸻

3.2 So sánh khách quan

Đánh giá định lượng loại bỏ yếu tố chủ quan của con người.

Theo quan điểm của George Box:

“All models are wrong, but some are useful.”

Chỉ số giúp ta đo mức độ “useful”.

⸻

4. Thách Thức Cơ Bản

4.1 Sai lệch phân phối (Distribution Shift)

Nếu:

D_{train} \neq D_{test}

thì:

\hat{R}_{test} \not\approx R_{real}

⸻

4.2 Overfitting vào Benchmark

Giả sử có k mô hình thử nghiệm:

\max_{1 \le i \le k}
\hat{R}_i

Khi k lớn, xác suất chọn mô hình overfit vào test tăng theo bất đẳng thức Hoeffding.

⸻

4.3 Độ tin cậy thống kê

Sai số chuẩn:

SE =
\sqrt{
\frac{\hat{p}(1-\hat{p})}{n}
}

Khoảng tin cậy 95%:

\hat{p} \pm 1.96 \cdot SE

Nếu hai mô hình chênh lệch nhỏ hơn sai số chuẩn → khác biệt không có ý nghĩa thống kê.

⸻

5. Đánh giá Định lượng vs Đánh giá Con người

Giả sử:

Score_{auto}
=
g(f_\theta)

Score_{human}
=
h(f_\theta)

Ta quan tâm đến tương quan:

\rho =
Corr(Score_{auto}, Score_{human})

Nếu \rho thấp → chỉ số tự động không phản ánh đúng chất lượng thực tế.

⸻

6. Phân tích Bias–Variance trong Đánh giá

Sai số tổng quát:

\mathbb{E}
\left[
(y - \hat{f}(x))^2
\right]
=
Bias^2
+
Variance
+
\sigma^2

Benchmark nhỏ:
	•	Variance cao
	•	Không ổn định

Benchmark lớn:
	•	Giảm variance
	•	Tăng chi phí tính toán

⸻

7. Trường hợp Mô hình Ngôn ngữ Lớn (LLMs)

Trong các hệ thống hiện đại:
	•	Đánh giá zero-shot
	•	Few-shot
	•	In-context learning

Mô hình có thể tối ưu ngầm theo benchmark phổ biến.

Hiện tượng:

Performance_{public}
>
Performance_{real}

Do contamination dữ liệu huấn luyện.

⸻

8. Hướng Giải Quyết

8.1 Cross-validation

CV =
\frac{1}{k}
\sum_{i=1}^{k}
\hat{R}_i

⸻

8.2 Bootstrap

Lấy mẫu lại:

\hat{R}^{(b)} =
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i^{(b)}), y_i^{(b)})

⸻

8.3 Kết hợp đánh giá định tính

Tối ưu:

Score_{final}
=
\alpha Score_{auto}
+
(1-\alpha) Score_{human}

⸻

9. Thảo luận

Đánh giá định lượng mang lại:
	•	Tính hệ thống
	•	So sánh chuẩn hóa
	•	Tự động hóa

Nhưng cũng tồn tại:
	•	Lệ thuộc benchmark
	•	Sai lệch phân phối
	•	Hiệu ứng Goodhart:

\text{When a measure becomes a target, it ceases to be a good measure.}

⸻

10. Kết luận

Đánh giá định lượng là công cụ thiết yếu nhưng không toàn diện. Về bản chất, nó là ước lượng thống kê của rủi ro tổng quát hóa:

\hat{R}(\theta)
\approx
R(\theta)

Để đánh giá AI một cách đáng tin cậy, cần:
	•	Phân tích thống kê nghiêm ngặt
	•	Kiểm định ý nghĩa
	•	Kết hợp đánh giá con người
	•	Kiểm soát contamination dữ liệu

Hiểu đúng hứa hẹn và giới hạn của đánh giá định lượng là điều kiện tiên quyết để phát triển hệ thống AI đáng tin cậy.

⸻

Tài liệu tham khảo
	1.	Manning, C. D., & Schütze, H. (1999). Foundations of Statistical Natural Language Processing.
	2.	Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.
	3.	Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
	4.	Box, G. E. P. (1976). Science and Statistics.
	5.	Dror, R. et al. (2018). The Hitchhiker’s Guide to Statistical Significance in NLP.