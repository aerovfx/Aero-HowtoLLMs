
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
# Reinforcement Learning from Human Feedback (RLHF): Cơ sở lý thuyết, mô hình toán học và ứng dụng trong huấn luyện mô hình ngôn ngữ lớn

---

## Tóm tắt

Reinforcement Learning from Human Feedback (RLHF) là phương pháp huấn luyện mô hình ngôn ngữ lớn (LLMs) nhằm tối ưu hóa đầu ra theo đánh giá của con người. Bài viết này trình bày cơ sở toán học của RLHF, phân tích từng giai đoạn huấn luyện (Supervised Fine-Tuning, Reward Modeling, Policy Optimization), và thảo luận vai trò của PPO cùng regularization KL-divergence. Phân tích được đặt trong bối cảnh các mô hình GPT do OpenAI phát triển, đặc biệt là InstructGPT (Ouyang et al., 2022).

---

# 1. Giới thiệu

Các mô hình ngôn ngữ như GPT-2 hay GPT-3 được huấn luyện theo mục tiêu dự đoán token kế tiếp:

$$
P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t \mid x_{\lt t})
$$

Tuy nhiên, mục tiêu tối đa hóa likelihood không đảm bảo mô hình:

* Tuân thủ chỉ thị (instruction-following)
* Trả lời an toàn
* Phù hợp với giá trị con người

RLHF được đề xuất để giải quyết khoảng cách giữa tối ưu hóa xác suất và tối ưu hóa sự hài lòng của con người.

---

# 2. Khung lý thuyết Reinforcement Learning

Trong RL cổ điển, ta có:

* Trạng thái: $s$
* Hành động: $a$
* Chính sách: $\pi_\theta(a\mids$ )
* Phần thưởng: ( r(s,a) )

Mục tiêu tối ưu:

$$
\max_\theta \mathbb{E}*{\tau \sim \pi*\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

Trong RLHF:

* Trạng thái $s$: prompt (instruction)
* Hành động $a$: chuỗi phản hồi
* Reward: điểm đánh giá từ con người hoặc reward model

---

# 3. Quy trình RLHF

## 3.1. Bước 1 – Supervised Fine-Tuning (SFT)

Huấn luyện trên dữ liệu cặp (instruction, response):

$$
\mathcal{L}*{SFT} = - \sum*{t \in R} \log P_\theta(x_t \mid x_{\lt t})
$$

Mục tiêu: đưa mô hình về phân phối gần với hành vi mong muốn.

---

## 3.2. Bước 2 – Huấn luyện Reward Model

Cho hai phản hồi ( y_1, y_2 ) với cùng prompt $x$, con người chọn phản hồi tốt hơn.

Reward model $r_\phi(x,y$ ) được huấn luyện bằng loss Bradley-Terry:

$$
P(y_1 \succ y_2) = \frac{e^{r_\phi(x,y_1)}}{e^{r_\phi(x,y_1)} + e^{r_\phi(x,y_2)}}
$$

Loss:

$$
\mathcal{L}*{RM} = - \log \sigma(r*\phi(x,y_{chosen}) - r_\phi(x,y_{rejected}))
$$

Trong đó $\sigma$ là sigmoid:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

---

## 3.3. Bước 3 – Policy Optimization (PPO)

Sau khi có reward model, ta tối ưu policy:

$$
\max_\theta \mathbb{E}*{x \sim \pi*\theta} \left[ r_\phi(x) - \beta D_{KL}(\pi_\theta | \pi_{ref}) \right]
$$

Trong đó:

* $\pi_{ref}$: mô hình SFT ban đầu
* $D_{KL}$: KL divergence

$$
D_{KL}(P|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

---

# 4. Proximal Policy Optimization (PPO)

PPO tối ưu hàm mục tiêu:

$$
L^{CLIP}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

Trong đó:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)} {\pi_{\theta_{old}}(a_t|s_t)}
$$

$A_t$: advantage estimate.

Clipping giúp:

* Tránh cập nhật quá lớn
* Ổn định huấn luyện

---

# 5. KL Regularization

Nếu không có KL penalty:

$$
\pi_\theta \to \text{mode collapse}
$$

Với KL:

$$
\mathcal{L} = \mathbb{E}[r(x)] * \beta D_{KL}(\pi_\theta | \pi_{ref})
$$

KL đóng vai trò như regularizer:

* Giữ mô hình gần phân phối gốc
* Tránh hành vi bất thường

---

# 6. Phân tích độ phức tạp tính toán

Giả sử:

* $L$: số layer
* $T$: chiều dài chuỗi
* $d$: embedding dimension

Self-attention:

$$
\mathcal{O}(L \cdot T^2 \cdot d)
$$

Trong RLHF:

* Mỗi bước cần forward nhiều mẫu
* Tính thêm reward model
* Tính KL divergence

Chi phí tăng gấp 2–3 lần so với SFT.

---

# 7. Các vấn đề lý thuyết

## 7.1. Reward Hacking

Mô hình có thể tối đa hóa reward model nhưng không thực sự tốt.

Giả sử reward model xấp xỉ:

$$
r_\phi(x) = r_{true}(x) + \epsilon(x)
$$

Khi tối ưu:

$$
\max_\theta \mathbb{E}[r_\phi(x)]
$$

Sai số $\epsilon(x$ ) có thể bị khai thác.

---

## 7.2. Alignment Problem

Ta muốn:

$$
\pi_\theta \approx \pi_{human}
$$

Nhưng reward chỉ là xấp xỉ.

Đây là trung tâm của nghiên cứu alignment hiện đại.

---

# 8. So sánh với các hướng tiếp cận khác

| Phương pháp | Ưu điểm                  | Nhược điểm         |
| ----------- | ------------------------ | ------------------ |
| SFT         | Đơn giản                 | Phụ thuộc dữ liệu  |
| RLHF        | Linh hoạt, alignment tốt | Tốn chi phí        |
| DPO         | Không cần PPO            | Giới hạn lý thuyết |

---

# 9. Thảo luận

RLHF là cầu nối giữa:

* Học có giám sát
* Học tăng cường
* Học theo giá trị con người

Cách tiếp cận này đã được áp dụng trong các mô hình GPT của OpenAI và mở ra hướng phát triển LLM an toàn hơn.

---

# 10. Kết luận

RLHF cho phép:

1. Tối ưu hóa hành vi thay vì chỉ tối ưu xác suất
2. Kết hợp đánh giá con người vào vòng lặp huấn luyện
3. Kiểm soát mô hình thông qua KL-regularization

Về mặt toán học, RLHF là sự kết hợp giữa:

* Maximum Likelihood Estimation
* Policy Gradient
* Regularized Optimization

Phương pháp này hiện là nền tảng của các hệ thống LLM hiện đại.

---

# Tài liệu tham khảo

1. Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback*.
2. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*.
3. Sutton, R., Barto, A. (2018). *Reinforcement Learning: An Introduction*.
4. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Instruction Tuning (Tinh Chỉnh Bằng Chỉ Thị) Trong Các Mô Hình Ngôn Ngữ Lớn (LLMs)](aero_llm_01_what_is_instruction_tuning.md) | [Xem bài viết →](aero_llm_01_what_is_instruction_tuning.md) |
| [Instruction Tuning trong Mô hình Ngôn ngữ Lớn](aero_llm_02_some_datasets_for_instruction_tuning.md) | [Xem bài viết →](aero_llm_02_some_datasets_for_instruction_tuning.md) |
| [Huấn luyện Chatbot theo Instruction Tuning và Mô hình System–User–Assistant](aero_llm_03_training_a_chatbot_with_system_user_assistant.md) | [Xem bài viết →](aero_llm_03_training_a_chatbot_with_system_user_assistant.md) |
| [Instruction Tuning với GPT-2 trong Huấn luyện Mô hình Ngôn ngữ](aero_llm_04_instruction_tuning_with_gpt2.md) | [Xem bài viết →](aero_llm_04_instruction_tuning_with_gpt2.md) |
| [aero llm 05 codechallenge instruction tuning gpt2 large part 1](aero_llm_05_codechallenge_instruction_tuning_gpt2_large_part_1_.md) | [Xem bài viết →](aero_llm_05_codechallenge_instruction_tuning_gpt2_large_part_1_.md) |
| [Phân tích nâng cao quá trình Instruction Tuning cho GPT-2 Large: Ổn định huấn luyện, động học gradient và tối ưu hoá tính toán](aero_llm_06_codechallenge_instruction_tuning_gpt2_large_part_2_.md) | [Xem bài viết →](aero_llm_06_codechallenge_instruction_tuning_gpt2_large_part_2_.md) |
| 📌 **[Reinforcement Learning from Human Feedback (RLHF): Cơ sở lý thuyết, mô hình toán học và ứng dụng trong huấn luyện mô hình ngôn ngữ lớn](aero_llm_07_reinforcement_learning_from_human_feedback_rlhf_.md)** | [Xem bài viết →](aero_llm_07_reinforcement_learning_from_human_feedback_rlhf_.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
