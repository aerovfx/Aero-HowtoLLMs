
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [07 fine tune pretrained models](index.md)

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
# 📘 Phân Tích Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu Quy Mô Lớn

## Tóm tắt (Abstract)

Trong huấn luyện mô hình học sâu hiện đại, đặc biệt là các mô hình ngôn ngữ lớn, việc kiểm soát tốc độ học và độ ổn định số học đóng vai trò then chốt. Learning Rate Scheduler là một kỹ thuật giúp điều chỉnh learning rate theo thời gian nhằm cải thiện khả năng hội tụ và hạn chế dao động. Bài viết trình bày cơ sở lý thuyết, mô hình toán học và kết quả thực nghiệm về các bộ điều chỉnh learning rate phổ biến như Cosine Scheduler và Linear Scheduler dựa trên tài liệu thực hành.

---

## 1. Giới thiệu

Tối ưu hóa trong học sâu chủ yếu dựa trên các thuật toán gradient-based. Tuy nhiên, việc sử dụng learning rate cố định thường gây ra các vấn đề như:

* Hội tụ chậm
* Dao động mạnh
* Dễ mắc kẹt tại điểm tối ưu cục bộ

Theo tài liệu thực nghiệm , Learning Rate Scheduler giúp khắc phục các hạn chế trên thông qua điều chỉnh learning rate động.

Mục tiêu nghiên cứu:

* Phân tích cơ chế hoạt động của scheduler
* Xây dựng mô hình toán học
* Đánh giá tác động đến quá trình học
* So sánh các phương pháp điều chỉnh

---

## 2. Cơ sở lý thuyết

### 2.1 Cập nhật tham số trong học sâu

Quy trình cập nhật tham số:
$$
\theta_{t+1}=\theta_t-\eta_t \nabla_\theta L(\theta_t)
$$


Trong đó:

* $\eta_t$: learning rate tại thời điểm $t$
* $\nabla_\theta L$: gradient hàm mất mát

Learning rate biến thiên theo thời gian giúp điều chỉnh độ lớn bước học.

---

### 2.2 Vai trò của Learning Rate

Learning rate ảnh hưởng trực tiếp tới:

* Tốc độ hội tụ
* Độ ổn định
* Khả năng tối ưu toàn cục

Khi:
$$
\eta_t \to 0 \Rightarrow \theta_{t+1}\approx \theta_t
$$


⇒ quá trình học gần như dừng lại.

---

## 3. Phương pháp nghiên cứu

### 3.1 Warm-up Phase

#### 3.1.1 Khái niệm

Warm-up giúp tránh cập nhật quá mạnh ở giai đoạn đầu huấn luyện.

Theo , learning rate tăng dần trong giai đoạn đầu.

---

#### 3.1.2 Mô hình toán học

Warm-up tuyến tính:
$$
\eta_t=\eta_{max}\cdot\frac{t}{T_{warm}},\quad t\le T_{warm}
$$


Trong đó:

* $T_{warm}$: số bước warm-up

---

### 3.2 Cosine Learning Rate Scheduler

#### 3.2.1 Nguyên lý

Cosine scheduler làm giảm learning rate theo hàm cosin.

---

#### 3.2.2 Công thức

Với $C$ chu kỳ:
$$
\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})
\left(1+\cos\frac{2\pi Ct}{T}\right)
$$


Trường hợp $C=\frac{1}{2}$:
$$
\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})
\left(1+\cos\frac{\pi t}{T}\right)
$$


---

#### 3.2.3 Đặc điểm

* Giảm learning rate mượt
* Tránh giảm đột ngột
* Phù hợp Transformer, LLM

---

### 3.3 Linear Learning Rate Scheduler

#### 3.3.1 Nguyên lý

Giảm learning rate tuyến tính sau warm-up.

---

#### 3.3.2 Công thức
$$
\eta_t=
\begin{cases}
\eta_{max}\frac{t}{T_{warm}} & t\le T_{warm}\
\eta_{max}\left(1-\frac{t-T_{warm}}{T-T_{warm}}\right) & t>T_{warm}
\end{cases}
$$


---

#### 3.3.3 Điều chỉnh số bước huấn luyện

Theo , việc khai báo số bước khác với thực tế giúp:
$$
T_{sched}>T_{train}
\Rightarrow \eta_t>0
$$


trong suốt quá trình huấn luyện.

---

### 3.4 Kết hợp với Gradient Clipping

Cập nhật tham số tổng quát:
$$
\theta_{t+1}=\theta_t-\eta_t\cdot
\frac{c}{\max(|\mathbf{g}|,c)}\mathbf{g}
$$


Trong đó:

* $c$: ngưỡng clipping

---

## 4. Thực nghiệm

### 4.1 Mô hình minh họa

Theo tài liệu , mô hình gồm:

* Vector trọng số (w=(w_1,w_2))
* Mục tiêu: (w_1>w_2)
* SGD + Scheduler

Hàm mất mát:
$$
L=-\log\frac{e^{w_1}}{e^{w_1}+e^{w_2}}
$$


---

### 4.2 Cosine Scheduler

Quan sát thực nghiệm:

* Học theo từng pha
* Xuất hiện giai đoạn "đóng băng"
* Học mạnh khi $\eta_t$ lớn

Đồ thị:
$$
w(t)\propto \int_0^t \eta_s ds
$$


---

### 4.3 Linear Scheduler

Đặc điểm:

* Học đều
* Ít dao động
* Dễ kiểm soát

Trường hợp $\eta_t=0$:
$$
\theta_{t+1}=\theta_t
$$


⇒ không học.

---

### 4.4 So sánh thực nghiệm

| Phương pháp      | Độ mượt    | Hội tụ  | Ổn định |
| ---------------- | ---------- | ------- | ------- |
| Không scheduler  | Thấp       | Kém     | Thấp    |
| Cosine           | Cao        | Tốt     | Tốt     |
| Linear           | Trung bình | Tốt     | Cao     |
| Warm-up + Cosine | Rất cao    | Rất tốt | Rất tốt |

---

## 5. Thảo luận

### 5.1 Kiểm soát phạm vi giá trị

Theo , hệ thống học sâu cần giữ giá trị trong miền ổn định:
$$
|\theta_i|<M,\quad |g_i|<K
$$


Các kỹ thuật hỗ trợ:

* Weight initialization
* LayerNorm
* Weight decay
* Clipping
* Scheduler

---

### 5.2 Ứng dụng trong LLM

Scheduler giúp:

* Ổn định huấn luyện Transformer
* Giảm gradient noise
* Hạn chế overfitting

Đặc biệt quan trọng với mô hình trên 1B tham số.

---

### 5.3 Hạn chế

* Phụ thuộc siêu tham số
* Khó tối ưu thủ công
* Tăng độ phức tạp huấn luyện

Cần thử nghiệm nhiều cấu hình.

---

## 6. Kết luận

Bài viết đã trình bày Learning Rate Scheduler trong huấn luyện mô hình học sâu, tập trung vào Cosine và Linear Scheduler.

Kết quả cho thấy:

* Scheduler cải thiện hội tụ
* Warm-up tăng ổn định
* Kết hợp clipping cho hiệu quả cao

Các phương pháp này là thành phần không thể thiếu trong huấn luyện mô hình AI hiện đại.

---

## Tài liệu tham khảo

1. Learning Rate Scheduler Tutorial (Part 2) 
2. Loshchilov, I., Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts.
3. Kingma, D., Ba, J. (2015). Adam: A Method for Stochastic Optimization.
4. Vaswani, A. et al. (2017). Attention Is All You Need.

---
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 07_fine_tune_pretrained_models](README.md) | [Xem bài viết →](README.md) |
| [Fine-tuning Có Mục Tiêu và Đóng Băng Chính Xác Trọng Số Trong Mô Hình Ngôn Ngữ Lớn](aero_llm_010_codechallenge_fine_tuning_and_targeted_freezing_part_1_.md) | [Xem bài viết →](aero_llm_010_codechallenge_fine_tuning_and_targeted_freezing_part_1_.md) |
| [Phân Tích Hiệu Quả Fine-tuning và Targeted Freezing (Phần 2): Đánh Giá Bằng Trực Quan Hóa và Chuẩn Ma Trận](aero_llm_011_codechallenge_fine_tuning_and_targeted_freezing_part_2_.md) | [Xem bài viết →](aero_llm_011_codechallenge_fine_tuning_and_targeted_freezing_part_2_.md) |
| [Fine-tuning Hiệu Quả Tham Số (Parameter-Efficient Fine-Tuning – PEFT) Trong Mô Hình Ngôn Ngữ Lớn](aero_llm_012_parameter_efficient_fine_tuning_peft_.md) | [Xem bài viết →](aero_llm_012_parameter_efficient_fine_tuning_peft_.md) |
| [Mô Hình CodeGen Cho Bài Toán Hoàn Thành Mã Nguồn: Kiến Trúc, Huấn Luyện và Ứng Dụng](aero_llm_013_codegen_for_code_completion.md) | [Xem bài viết →](aero_llm_013_codegen_for_code_completion.md) |
| [Fine-tuning Mô Hình CodeGen Cho Bài Toán Giải Tích: Phương Pháp, Đánh Giá và Ứng Dụng](aero_llm_014_codechallenge_fine_tune_codegen_for_calculus.md) | [Xem bài viết →](aero_llm_014_codechallenge_fine_tune_codegen_for_calculus.md) |
| [Tinh Chỉnh Mô Hình BERT Cho Bài Toán Phân Loại Cảm Xúc Văn Bản IMDb](aero_llm_015_fine_tuning_bert_for_classification.md) | [Xem bài viết →](aero_llm_015_fine_tuning_bert_for_classification.md) |
| [📘 Ứng Dụng Mô Hình BERT Trong Phân Tích Cảm Xúc Đánh Giá Phim IMDB](aero_llm_016_codechallenge_imdb_sentiment_analysis_using_bert_en_us.md) | [Xem bài viết →](aero_llm_016_codechallenge_imdb_sentiment_analysis_using_bert_en_us.md) |
| [📘 Ứng Dụng Gradient Clipping và Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu](aero_llm_017_gradient_clipping_and_learning_rate_scheduler_part_1_en_us.md) | [Xem bài viết →](aero_llm_017_gradient_clipping_and_learning_rate_scheduler_part_1_en_us.md) |
| 📌 **[📘 Phân Tích Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu Quy Mô Lớn](aero_llm_018_gradient_clipping_and_learning_rate_scheduler_part_2_.md)** | [Xem bài viết →](aero_llm_018_gradient_clipping_and_learning_rate_scheduler_part_2_.md) |
| [📘 Kết Hợp Gradient Clipping, Freezing và Learning Rate Scheduler Trong Fine-Tuning Mô Hình BERT](aero_llm_019_codechallenge_clip_freeze_and_schedule_bert.md) | [Xem bài viết →](aero_llm_019_codechallenge_clip_freeze_and_schedule_bert.md) |
| [Tối Ưu Hóa Quá Trình Tiền Huấn Luyện Mô Hình Ngôn Ngữ Lớn: Phân Tích Các Chiến Lược Tính Toán và Học Tập](aero_llm_01_what_does_fine_tuning_mean.md) | [Xem bài viết →](aero_llm_01_what_does_fine_tuning_mean.md) |
| [Lưu Trữ và Tải Lại Mô Hình Học Sâu Trong PyTorch và Hugging Face: Phương Pháp, Cấu Trúc và Đánh Giá](aero_llm_020_saving_and_loading_trained_models.md) | [Xem bài viết →](aero_llm_020_saving_and_loading_trained_models.md) |
| [Ứng Dụng Mô Hình BERT Trong Phân Loại Văn Bản Văn Học: Trường Hợp Alice và Edgar](aero_llm_021_bert_decides_alice_or_edgar.md) | [Xem bài viết →](aero_llm_021_bert_decides_alice_or_edgar.md) |
| [Đồng Tiến Hóa Mô Hình Sinh Văn Bản và Mô Hình Phân Loại: Trường Hợp Alice và Edgar](aero_llm_022_codechallenge_evolution_of_alice_and_edgar_part_1_.md) | [Xem bài viết →](aero_llm_022_codechallenge_evolution_of_alice_and_edgar_part_1_.md) |
| [📘 Đánh Giá Mô Hình Sinh Văn Bản Thông Qua Phân Loại BERT: Nghiên Cứu Trường Hợp Alice và Edgar](aero_llm_023_codechallenge_evolution_of_alice_and_edgar_part_2_.md) | [Xem bài viết →](aero_llm_023_codechallenge_evolution_of_alice_and_edgar_part_2_.md) |
| [Fine-tuning Mô hình GPT-2 trên Tác phẩm *Gulliver’s Travels*: Phân tích Thực nghiệm và Đánh giá Hiệu quả](aero_llm_02_fine_tune_a_pretrained_gpt2.md) | [Xem bài viết →](aero_llm_02_fine_tune_a_pretrained_gpt2.md) |
| [Đánh giá Ảnh hưởng của Learning Rate trong Fine-tuning GPT-2 trên *Gulliver’s Travels*](aero_llm_03codechallenge_gulliver_s_learning_rates.md) | [Xem bài viết →](aero_llm_03codechallenge_gulliver_s_learning_rates.md) |
| [Nghiên cứu Quy trình Sinh Văn bản từ Mô hình Ngôn ngữ Tiền Huấn luyện GPT-2](aero_llm_04_on_generating_text_from_pretrained_models.md) | [Xem bài viết →](aero_llm_04_on_generating_text_from_pretrained_models.md) |
| [Tinh Chỉnh Mô Hình GPT-2 Bằng Hàm Mất Mát KL Divergence Để Tối Ưu Hóa Việc Sinh Token Chứa Ký Tự “X”](aero_llm_05_codechallenge_maximize_the_x_factor_.md) | [Xem bài viết →](aero_llm_05_codechallenge_maximize_the_x_factor_.md) |
| [Tinh Chỉnh Mô Hình GPT-Neo Để Mô Phỏng Phong Cách Văn Học Alice in Wonderland và Edgar Allan Poe](aero_llm_06_alice_in_wonderland_and_edgar_allen_poe_with_gpt_neo_.md) | [Xem bài viết →](aero_llm_06_alice_in_wonderland_and_edgar_allen_poe_with_gpt_neo_.md) |
| [Đánh Giá Định Lượng và Định Tính Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp Văn Phong *Alice* và *Edgar Allan Poe*](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tunin.md) | [Xem bài viết →](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tunin.md) |
| [Định Lượng Hiệu Quả Tinh Chỉnh Phong Cách Văn Học: Thử Thách Alice và Edgar](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tuning.md) | [Xem bài viết →](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tuning.md) |
| [Mô Phỏng Hội Thoại Giữa Hai Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp *Alice* và *Edgar*](aero_llm_08_codechallenge_a_chat_between_alice_and_edgar.md) | [Xem bài viết →](aero_llm_08_codechallenge_a_chat_between_alice_and_edgar.md) |
| [Tinh Chỉnh Từng Phần Bằng Cách Đóng Băng Trọng Số Attention: Chiến Lược Tối Ưu Hóa Tham Số Cho LLM](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) | [Xem bài viết →](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
