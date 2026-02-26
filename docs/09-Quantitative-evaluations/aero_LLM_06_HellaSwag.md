
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [09 Quantitative evaluations](../index.md)

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
Đánh giá năng lực suy luận thường thức của mô hình ngôn ngữ lớn thông qua bộ dữ liệu HellaSwag

Phân tích phương pháp, cơ sở toán học và các thách thức định lượng

⸻

Tóm tắt

Bài viết này phân tích bộ dữ liệu HellaSwag – một chuẩn đánh giá năng lực suy luận thường thức (commonsense reasoning) của mô hình ngôn ngữ lớn (LLMs). Dựa trên nội dung tài liệu đính kèm, chúng tôi mở rộng phân tích bằng cách tham chiếu đến các nghiên cứu của Rowan Zellers et al. (2019), kiến trúc Transformer của Ashish Vaswani et al. (2017), cùng các phương pháp tiền huấn luyện trong OpenAI và Google.

Chúng tôi trình bày:
	•	Cấu trúc và nguyên lý của HellaSwag
	•	Cách tính xác suất lựa chọn đáp án
	•	Công thức toán học của accuracy và log-likelihood
	•	So sánh giữa con người và mô hình
	•	Các thách thức về adversarial filtering

⸻

1. Giới thiệu

HellaSwag được đề xuất nhằm kiểm tra khả năng:
	•	Suy luận tiếp diễn hành động (physical commonsense)
	•	Hiểu bối cảnh
	•	Phân biệt kết thúc hợp lý và vô lý

Ví dụ (rút gọn):

“A person is cooking in the kitchen. They pick up a knife and…”
A. start slicing vegetables
B. jump into a swimming pool
C. fly into space
D. dissolve into smoke

Con người dễ dàng chọn A.
Tuy nhiên mô hình ngôn ngữ phải tính xác suất cho từng lựa chọn.

⸻

2. Cấu trúc toán học của bài toán

Cho:
	•	Ngữ cảnh: c
	•	Tập 4 đáp án: \{a_1, a_2, a_3, a_4\}

Mô hình ước lượng:

P(a_i \mid c)

Đáp án được chọn:

\hat{a} = \arg\max_{a_i} P(a_i \mid c)

⸻

3. Tính xác suất trong mô hình tự hồi quy

Với mô hình kiểu GPT:

P(a_i \mid c) = \prod_{t=1}^{T_i} P(w_t \mid c, w_{<t})

Trong thực nghiệm, ta dùng log-likelihood:

\log P(a_i \mid c) = \sum_{t=1}^{T_i} \log P(w_t \mid c, w_{<t})

Để tránh thiên vị độ dài, thường dùng chuẩn hoá:

Score(a_i) = \frac{1}{T_i} \sum_{t=1}^{T_i} \log P(w_t \mid c, w_{<t})

⸻

4. Accuracy trong HellaSwag

Với N câu hỏi:

Accuracy = \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}(\hat{a}^{(j)} = a^{(j)}_{\text{true}})

Baseline ngẫu nhiên:

P_{\text{random}} = \frac{1}{4} = 25\%

Hiệu năng con người ≈ 95%
Các mô hình cũ (trước Transformer lớn) ≈ 30–40%

⸻

5. Adversarial Filtering

Theo Rowan Zellers, HellaSwag sử dụng Adversarial Filtering (AF):
	1.	Sinh nhiều kết thúc sai bằng mô hình ngôn ngữ.
	2.	Lọc bỏ những đáp án mà mô hình hiện tại dễ phân biệt.
	3.	Giữ lại những đáp án “đánh lừa” mô hình.

Mô hình lọc:

f_\theta(c, a_i)

Giữ lại các mẫu mà:

f_\theta(c, a_{\text{true}}) - f_\theta(c, a_{\text{false}}) \approx 0

Điều này làm bộ dữ liệu ngày càng khó.

⸻

6. Liên hệ với Self-Attention

Kiến trúc Transformer:

Attention(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

Self-attention cho phép mô hình:
	•	Hiểu quan hệ dài hạn
	•	Nắm bắt chuỗi hành động
	•	Phân biệt logic vật lý

⸻

7. So sánh HellaSwag và Perplexity

Perplexity đo:

PP = \exp\left(- \frac{1}{N} \sum \log P(w_i)\right)

Trong khi HellaSwag đo:
	•	Khả năng so sánh nhiều chuỗi hoàn chỉnh
	•	Suy luận cấp cao

Mô hình có perplexity thấp chưa chắc có accuracy cao trên HellaSwag.

⸻

8. Phân tích thống kê

Giả sử mô hình đạt accuracy \hat{p} trên N mẫu:

Sai số chuẩn:

SE = \sqrt{\frac{\hat{p}(1-\hat{p})}{N}}

Khoảng tin cậy 95%:

\hat{p} \pm 1.96 \cdot SE

Ví dụ:
	•	N = 10,000
	•	Accuracy = 0.80

SE = \sqrt{\frac{0.8(0.2)}{10000}} = 0.004

Khoảng tin cậy:

0.80 \pm 0.008

⸻

9. Những hạn chế của HellaSwag

9.1 Bias ngôn ngữ

Mô hình có thể học:
	•	Mẫu văn phong
	•	Cấu trúc câu hợp lý hơn

Chứ không thực sự hiểu vật lý.

⸻

9.2 Overfitting Benchmark

Nếu mô hình được fine-tune trực tiếp trên HellaSwag:

D_{train} \cap D_{test} \neq \varnothing

Kết quả không còn phản ánh khả năng tổng quát.

⸻

9.3 Scaling Law

Theo các nghiên cứu của OpenAI:

Loss(N) = A N^{-\alpha} + B

Khi số tham số tăng → accuracy trên HellaSwag tăng gần theo hàm lũy thừa.

⸻

10. Ý nghĩa đối với đánh giá LLM

HellaSwag:
	•	Không chỉ đo xác suất từ
	•	Mà đo khả năng suy luận hành động
	•	Giảm thiểu shortcut learning

Do đó nó là benchmark quan trọng bên cạnh:
	•	MMLU
	•	ARC
	•	Winogrande

⸻

11. Kết luận

HellaSwag là một bước tiến quan trọng trong đánh giá năng lực suy luận thường thức của mô hình ngôn ngữ lớn.

Các điểm chính:
	•	Dựa trên multiple choice completion
	•	Sử dụng adversarial filtering
	•	Đánh giá bằng log-likelihood và accuracy
	•	Phân biệt rõ giữa fluency và reasoning

Trong tương lai, cần kết hợp:
	•	Đánh giá động (interactive reasoning)
	•	Phân tích attention map
	•	Đo calibration và uncertainty

⸻

Tài liệu tham khảo
	1.	Zellers, R. et al. (2019). HellaSwag: Can a Machine Really Finish Your Sentence?
	2.	Vaswani, A. et al. (2017). Attention is All You Need.
	3.	Brown et al. (2020). Language Models are Few-Shot Learners.
	4.	Kaplan et al. (2020). Scaling Laws for Neural Language Models.
	5.	Jurafsky & Martin. Speech and Language Processing.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
