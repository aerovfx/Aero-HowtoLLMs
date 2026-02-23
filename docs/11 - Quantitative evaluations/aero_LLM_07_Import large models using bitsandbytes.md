Nhập và triển khai mô hình ngôn ngữ lớn bằng lượng tử hóa 8-bit/4-bit với BitsAndBytes

Phân tích kiến trúc, cơ sở toán học và hiệu năng thực nghiệm

⸻

Tóm tắt

Bài viết này phân tích phương pháp nhập và triển khai mô hình ngôn ngữ lớn (LLMs) bằng kỹ thuật lượng tử hóa (quantization) sử dụng thư viện bitsandbytes. Dựa trên nội dung tài liệu đính kèm, chúng tôi mở rộng với các nền tảng lý thuyết từ Transformer của Ashish Vaswani et al. (2017), nghiên cứu về scaling laws của OpenAI và hệ sinh thái triển khai của Hugging Face.

Bài viết trình bày:
	•	Bài toán giới hạn bộ nhớ khi tải LLM
	•	Nguyên lý lượng tử hóa trọng số 8-bit và 4-bit
	•	Công thức sai số lượng tử hóa
	•	Phân tích độ phức tạp bộ nhớ
	•	So sánh hiệu năng trước và sau lượng tử hóa

⸻

1. Giới thiệu

Mô hình ngôn ngữ lớn hiện đại có số tham số từ:

10^9 \rightarrow 10^{11}

Giả sử:
	•	Mô hình có N tham số
	•	Mỗi tham số ở dạng FP32 (4 bytes)

Dung lượng bộ nhớ:

Memory = 4N \text{ bytes}

Ví dụ:

N = 7 \times 10^9

Memory = 28GB

Điều này vượt quá khả năng của nhiều GPU phổ thông.

⸻

2. Nguyên lý lượng tử hóa (Quantization)

2.1 Định nghĩa

Lượng tử hóa là ánh xạ:

w \in \mathbb{R} \rightarrow \hat{w} \in \mathbb{Z}_k

Trong đó:
	•	k = 2^b
	•	b là số bit (8-bit, 4-bit,…)

⸻

2.2 Lượng tử hóa tuyến tính (Linear Quantization)

Cho trọng số w nằm trong khoảng:

[w_{min}, w_{max}]

Hệ số scale:

s = \frac{w_{max} - w_{min}}{2^b - 1}

Giá trị lượng tử hóa:

\hat{w} = \text{round}\left(\frac{w - w_{min}}{s}\right)

Giải lượng tử:

w \approx s \hat{w} + w_{min}

⸻

3. Sai số lượng tử hóa

Sai số:

\epsilon = w - \hat{w}

Giả sử phân phối đều:

Var(\epsilon) = \frac{s^2}{12}

Khi giảm số bit b:
	•	s tăng
	•	Sai số tăng
	•	Mất mát thông tin tăng

⸻

4. 8-bit vs 4-bit

4.1 Bộ nhớ

Với FP32:

Memory_{32} = 32N \text{ bits}

Với 8-bit:

Memory_{8} = 8N \text{ bits}

Giảm:

\frac{Memory_{8}}{Memory_{32}} = \frac{1}{4}

Với 4-bit:

Memory_{4} = 4N \text{ bits}

Giảm:

\frac{Memory_{4}}{Memory_{32}} = \frac{1}{8}

⸻

4.2 Ảnh hưởng đến forward pass

Transformer sử dụng:

Y = XW

Sau lượng tử hóa:

Y = X\hat{W}

Sai số lan truyền:

\Delta Y = X(W - \hat{W})

Nếu:

||W - \hat{W}||_2 \text{ nhỏ}

→ Ảnh hưởng tới output nhỏ.

⸻

5. Kỹ thuật của BitsAndBytes

Thư viện bitsandbytes triển khai:
	•	Lượng tử hóa động (dynamic quantization)
	•	Lượng tử hóa theo block
	•	NF4 (NormalFloat4)

NF4 giả định trọng số phân phối chuẩn:

w \sim \mathcal{N}(0, \sigma^2)

Mapping phi tuyến giúp giảm sai số so với lượng tử hóa tuyến tính.

⸻

6. Tích hợp với Hugging Face Transformers

Hệ sinh thái của Hugging Face hỗ trợ:
	•	load_in_8bit=True
	•	load_in_4bit=True

Giảm bộ nhớ GPU đáng kể mà không cần huấn luyện lại toàn bộ mô hình.

⸻

7. Ảnh hưởng đến Perplexity

Perplexity:

PP = \exp\left(- \frac{1}{N} \sum \log P(w_i)\right)

Sau lượng tử hóa:

PP_{quant} = PP_{fp32} + \delta

Trong thực nghiệm:
	•	8-bit: \delta \approx 1\% - 3\%
	•	4-bit: \delta \approx 3\% - 8\%

Phụ thuộc kích thước mô hình.

⸻

8. Phân tích độ phức tạp tính toán

Phép nhân ma trận:

O(n^3)

Nhưng khi dùng int8:
	•	Giảm băng thông bộ nhớ
	•	Tăng throughput
	•	Tối ưu Tensor Core

Tốc độ thực tế tăng 1.5–2x trên GPU hỗ trợ INT8.

⸻

9. Lượng tử hóa và Scaling Law

Theo nghiên cứu scaling law của OpenAI:

Loss(N) = A N^{-\alpha}

Nếu lượng tử hóa làm tăng loss một lượng nhỏ \delta,
thì có thể bù bằng tăng nhẹ số tham số N.

⸻

10. So sánh với Pruning

Kỹ thuật	Giảm bộ nhớ	Giảm FLOPs	Ảnh hưởng độ chính xác
Quantization	✔	✖	Thấp–Trung
Pruning	✔	✔	Trung
Distillation	✔	✔	Thấp

Quantization phù hợp cho triển khai inference.

⸻

11. Hạn chế
	•	Gradient không ổn định khi fine-tune trực tiếp 4-bit
	•	Một số layer nhạy cảm (LayerNorm, Embedding)
	•	Cần mixed-precision

⸻

12. Kết luận

Lượng tử hóa bằng bitsandbytes:
	•	Giảm 4–8 lần bộ nhớ
	•	Giữ chất lượng gần tương đương FP32
	•	Phù hợp triển khai LLM trên GPU tầm trung

Trong tương lai:
	•	QLoRA
	•	Post-training quantization nâng cao
	•	Mixed precision adaptive

⸻

Tài liệu tham khảo
	1.	Vaswani, A. et al. (2017). Attention is All You Need.
	2.	Dettmers, T. et al. (2022). 8-bit Optimizers via Block-wise Quantization.
	3.	Kaplan et al. (2020). Scaling Laws for Neural Language Models.
	4.	Goodfellow et al. (2016). Deep Learning.
	5.	Hugging Face Transformers Documentation.

