# Phân tích nâng cao quá trình Instruction Tuning cho GPT-2 Large: Ổn định huấn luyện, động học gradient và tối ưu hoá tính toán

## Tóm tắt

Bài viết này tiếp tục phân tích quá trình instruction tuning cho GPT-2 Large (1.5B tham số), tập trung vào các vấn đề nâng cao gồm: động học gradient, ổn định huấn luyện (training stability), chiến lược tối ưu hoá bộ nhớ và ảnh hưởng của phân bố độ dài chuỗi. Phân tích được đặt trên nền tảng kiến trúc Transformer của Vaswani et al. (2017) và mô hình GPT-2 do OpenAI công bố (Radford et al., 2019). Đồng thời, bài viết liên hệ với hướng Instruction Tuning và RLHF sau này trong InstructGPT (Ouyang et al., 2022).

---

# 1. Bối cảnh lý thuyết

## 1.1. Mô hình ngôn ngữ tự hồi quy

GPT-2 tối ưu hoá xác suất chuỗi:

[
P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
]

Hàm mất mát cross-entropy:

[
\mathcal{L}(\theta) = - \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
]

Trong instruction tuning, chuỗi đầu vào có cấu trúc:

[
x = [\text{Instruction}; \text{Response}]
]

Và loss chỉ tính trên phần response:

[
\mathcal{L}*{SFT} = - \sum*{t \in R} \log P_\theta(x_t \mid x_{<t})
]

---

# 2. Động học Gradient trong Instruction Tuning

## 2.1. Phân bố độ dài token

Giả sử:

* ( L_Q = \mathbb{E}[|Q|] )
* ( L_A = \mathbb{E}[|A|] )

Thực nghiệm cho thấy:

[
L_A \gg L_Q
]

Gradient kỳ vọng:

[
\mathbb{E}[\nabla_\theta \mathcal{L}]
= - \mathbb{E} \left[ \sum_{t \in R} \nabla_\theta \log P_\theta(x_t \mid x_{<t}) \right]
]

Điều này dẫn tới hiện tượng:

* Phần response chi phối toàn bộ cập nhật tham số
* Instruction đóng vai trò điều kiện nhưng ít ảnh hưởng trực tiếp

---

## 2.2. Phương sai gradient

Phương sai gradient tỉ lệ với độ dài chuỗi:

[
Var(\nabla_\theta \mathcal{L}) \propto T
]

Khi câu trả lời dài, ta có:

[
Var \uparrow \Rightarrow \text{training instability}
]

Biện pháp:

* Gradient clipping:

[
g \leftarrow \frac{g}{\max(1, \frac{|g|}{c})}
]

* Mixed precision (FP16/BF16)
* Gradient accumulation

---

# 3. Phân tích độ phức tạp tính toán

Self-attention có độ phức tạp:

[
\mathcal{O}(T^2 d)
]

Với:

* (T): chiều dài chuỗi
* (d): embedding dimension

Tổng chi phí cho toàn mô hình:

[
\mathcal{O}(L \cdot T^2 \cdot d)
]

Trong đó:

* (L = 36) (số layer GPT-2 Large)
* (d = 1280)

Nếu tăng chiều dài chuỗi từ 512 lên 1024:

[
\text{Compute} \approx 4 \times
]

Do phụ thuộc bậc hai theo (T).

---

# 4. Ổn định huấn luyện (Training Stability)

## 4.1. Learning rate schedule

Warmup tuyến tính:

[
\eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}}
]

Sau warmup, thường dùng cosine decay:

[
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min}) \left(1 + \cos \frac{t\pi}{T}\right)
]

---

## 4.2. Adam Optimizer

GPT-2 thường dùng Adam:

[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
]

[
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
]

Cập nhật tham số:

[
\theta_t = \theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
]

Adam giúp ổn định khi gradient dao động mạnh do chuỗi dài.

---

# 5. Ảnh hưởng của Masking Loss

Nếu không mask instruction:

[
\mathcal{L}*{total} = \mathcal{L}*{instruction} + \mathcal{L}_{response}
]

Khi đó mô hình sẽ học:

* Sao chép instruction
* Tối ưu phân phối token không mong muốn

Masking đảm bảo:

[
\mathcal{L}_{instruction} = 0
]

Giúp mô hình tập trung vào sinh response.

---

# 6. So sánh với RLHF

Trong InstructGPT (Ouyang et al., 2022), quá trình gồm:

1. Supervised Fine-Tuning
2. Reward Model
3. Proximal Policy Optimization (PPO)

Mục tiêu PPO:

[
\max_\theta \mathbb{E}*{x \sim \pi*\theta}
\left[
r(x) - \beta D_{KL}(\pi_\theta | \pi_{ref})
\right]
]

Trong đó:

* ( r(x) ): reward từ mô hình đánh giá
* ( D_{KL} ): KL divergence

[
D_{KL}(P|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}
]

KL giúp giữ mô hình không lệch quá xa mô hình gốc.

---

# 7. Vấn đề bộ nhớ GPU

Bộ nhớ cần thiết:

[
Memory \approx
\text{Parameters} +
\text{Gradients} +
\text{Optimizer States}
]

Với 1.5B tham số:

* FP16: ~6GB
* Adam states: ~12GB

Tổng có thể vượt 20GB.

Giải pháp:

* ZeRO optimization
* Gradient checkpointing
* Offloading

---

# 8. Động học tổng quát hóa (Generalization Dynamics)

Theo lý thuyết bias-variance:

[
\mathbb{E}[(y - \hat y)^2] = Bias^2 + Variance + Noise
]

Instruction tuning làm:

* Giảm bias với tác vụ hỏi-đáp
* Có thể tăng variance nếu dataset nhỏ

Do đó cần:

[
n \gg \frac{d}{\epsilon}
]

Trong đó:

* (n): số mẫu
* (d): số tham số hiệu dụng
* (\epsilon): sai số mong muốn

---

# 9. Thảo luận

Phần 2 của quá trình instruction tuning cho thấy:

* Độ dài response chi phối gradient
* Attention tạo chi phí bậc hai theo chiều dài
* Masking là quyết định thiết kế quan trọng
* Ổn định huấn luyện phụ thuộc mạnh vào LR schedule và optimizer

GPT-2 Large, dù không được thiết kế ban đầu cho chatbot, vẫn có thể đạt hiệu quả cao sau instruction tuning nhờ khả năng biểu diễn lớn.

---

# 10. Kết luận

Instruction tuning cho GPT-2 Large minh họa:

1. Mối quan hệ giữa kiến trúc Transformer và động học gradient
2. Ảnh hưởng của phân bố token đến tối ưu hóa
3. Vai trò của các kỹ thuật ổn định huấn luyện
4. Giới hạn tính toán do attention bậc hai

Những phân tích này là nền tảng cho các mô hình lớn hơn và các phương pháp huấn luyện nâng cao như RLHF.

---

# Tài liệu tham khảo

1. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI.
2. Vaswani, A. et al. (2017). *Attention Is All You Need*.
3. Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback*.
4. Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
5. Kingma, D., Ba (2014). *Adam: A Method for Stochastic Optimization*.

---