
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [08 instruction tuning](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
1. Giới thiệu

Mô hình GPT-2 được công bố bởi nhóm nghiên cứu tại OpenAI (Radford et al., 2019) dưới sự dẫn dắt của Alec Radford. GPT-2 dựa trên kiến trúc Transformer decoder-only và được huấn luyện theo mục tiêu mô hình hóa ngôn ngữ tự hồi quy:

$P(x)$ = $\prod$_{t=1}^{T} $P($x_t$ \mid x_{\lt t})$

Trong đó:

•	x = (x_1, x_2, ..., x_T) là chuỗi token

	•	x_{<t} là các token trước thời điểm t

Instruction tuning mở rộng cách tiếp cận này bằng cách huấn luyện mô hình trên dữ liệu gồm cặp (instruction, response), nhằm tối ưu khả năng tuân thủ yêu cầu người dùng.

⸻

2. Kiến trúc GPT-2 Large

GPT-2 Large có khoảng 1.5 tỷ tham số, với cấu hình điển hình:

$$
•	Số tầng Transformer: L = 36 •	Kích thước embedding: d_{model} = 1280 •	Số head attention: h = 20 •	Kích thước tầng MLP trung gian: d_{ff} = 4 \times d_{model} = 5120
$$

2.1. Cơ chế Self-Attention

Trong mỗi tầng Transformer, attention được tính theo công thức:

\text{Attention}(Q, K, V) = \text{softmax} \left\frac{QK^T}{\sqrt{d_k}} \rightV

Trong đó:

•	Q = XW_Q

$$
•	K = XW_K
$$

•	V = XW_V

Multi-head attention được định nghĩa:

\text{MultiHead}(Q,K,V) = \text{Concat}(head_1,...,head_h)W_O

2.2. Khối MLP

Sau attention là tầng feed-forward:

\text{MLP}x = \text{GELU}(xW_1 + b_1)W_2 + b_2

Việc mở rộng chiều không gian lên 4 \times d_{model} giúp tăng khả năng biểu diễn phi tuyến.

⸻

3. Phân tích dữ liệu Q&A

Trong bài toán instruction tuning, dữ liệu gồm:
	•	Câu hỏi (instruction)
	•	Câu trả lời (response)

3.1. Thống kê độ dài token

Giả sử:
	•	$Q_i$: độ dài câu hỏi thứ i
	•	$A_i$: độ dài câu trả lời thứ i

Tổng số token:

N_Q = \sum_{i=1}^{n} Q_i

$$
N_A = \sum_{i=1}^{n} A_i Kết quả quan sát thực nghiệm cho thấy: \mathbb{E}[A_i] \gg \mathbb{E}[Q_i] Điều này dẫn đến mất cân bằng trong gradient khi tối ưu hóa. ⸻ 4. Hàm mất mát và tối ưu hóa Mục tiêu huấn luyện là tối thiểu hóa cross-entropy:
$$

\mathcal${L} = - $\sum$_{t=1}^{T} $\log$ P_\theta $x_t \mid x_{\lt t}

$$
Trong instruction tuning, ta thường: •	Nối instruction và response thành một chuỗi •	Che (mask) loss phần instruction •	Chỉ tối ưu phần response Khi đó:
$$

\mathcal${L}_{response} = - $\sum$_{t \in R} $\log$ P_\theta $x_t \mid x_{\lt t}

$$
với R là tập token thuộc response. ⸻ 5. Tác động của phân bố token đến huấn luyện 5.1. Mất cân bằng gradient Vì response dài hơn nhiều so với instruction: |R| \gg |Q| Điều này dẫn đến: •	Gradient chủ yếu đến từ response •	Instruction ít ảnh hưởng nếu không masking hợp lý 5.2. Giới hạn chiều dài ngữ cảnh Nếu độ dài tối đa là T_{max}: |Q| + |A| \le T_{max} Với GPT-2:
$$

T_{max} = 1024

$$
Nếu câu trả lời quá dài, instruction có thể bị cắt ngắn → giảm khả năng hiểu ngữ cảnh. ⸻ 6. So sánh với các hướng tiếp cận hiện đại Instruction tuning sau này (ví dụ InstructGPT) bổ sung: 1.	Supervised fine-tuning (SFT) 2.	Reinforcement Learning from Human Feedback (RLHF) Hàm mục tiêu trong RLHF:
$$

\max_\theta \mathbb{E}_{x \sim \pi_\theta} [ rx ]

$$
Trong đó rx là reward model đánh giá chất lượng câu trả lời. ⸻ 7. Các cân nhắc thực tiễn khi huấn luyện GPT-2 Large 7.1. Bộ nhớ và batch size Với 1.5B tham số: \text{Memory} \approx 6 - 12 \text{ GB (FP16)} Gradient accumulation thường được sử dụng:
$$

\text{Effective Batch Size} = \text{Micro Batch} \times \text{Steps}

$$
7.2. Learning rate Thông thường: \eta \in [10^{-5}, 10^{-4}] Với warmup: \eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}}
$$

