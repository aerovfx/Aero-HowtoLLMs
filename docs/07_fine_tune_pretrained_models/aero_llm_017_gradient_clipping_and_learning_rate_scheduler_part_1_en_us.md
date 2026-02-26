
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
# 📘 Ứng Dụng Gradient Clipping và Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu

## Tóm tắt (Abstract)

Trong quá trình huấn luyện các mô hình học sâu quy mô lớn, hiện tượng mất ổn định số học và hội tụ kém thường xuyên xảy ra. Hai kỹ thuật phổ biến nhằm khắc phục vấn đề này là Gradient Clipping và Learning Rate Scheduler. Bài viết trình bày nguyên lý, cơ sở toán học và ứng dụng thực nghiệm của hai phương pháp trên, dựa trên tài liệu huấn luyện thực tế. Kết quả cho thấy việc áp dụng hợp lý các kỹ thuật này giúp tăng tính ổn định và độ tin cậy của quá trình tối ưu.

---

## 1. Giới thiệu

Huấn luyện mạng nơ-ron sâu thường dựa trên phương pháp tối ưu gradient descent. Tuy nhiên, với các mô hình lớn, gradient có thể trở nên rất lớn (gradient explosion), dẫn đến:

* Mất ổn định số học
* Sai lệch quá trình cập nhật
* Mô hình không hội tụ

Theo tài liệu hướng dẫn , hai kỹ thuật thường được sử dụng để giải quyết vấn đề này là:

* Gradient Clipping
* Learning Rate Scheduler

Mục tiêu nghiên cứu gồm:

* Phân tích cơ chế hoạt động của hai kỹ thuật
* Trình bày công thức toán học liên quan
* Đánh giá ảnh hưởng tới quá trình học
* Đề xuất hướng áp dụng thực tế

---

## 2. Cơ sở lý thuyết

### 2.1 Gradient Descent

Quá trình cập nhật tham số trong học sâu được mô tả bởi:

$$

\theta_{t+1}=\theta_t-\eta \nabla_\theta L(\theta_t)

$$


Trong đó:

* $\theta_t$: tham số tại bước $t$
* $\eta$: learning rate
* $L$: hàm mất mát
* $\nabla_\theta L$: gradient

Khi $|\nabla_\theta L|$ quá lớn, cập nhật tham số trở nên không ổn định.

---

### 2.2 Chuẩn của Gradient

Chuẩn Euclid của gradient:

$$

|\mathbf{g}|*2=\sqrt{\sum*{i=1}^{n}g_i^2}

$$


Trong đó:

* $\mathbf{g}$: vector gradient
* $g_i$: phần tử thứ $i$

Gradient explosion xảy ra khi:

$$

|\mathbf{g}|_2 \gg 1

$$


---

## 3. Phương pháp nghiên cứu

### 3.1 Gradient Clipping

#### 3.1.1 Khái niệm

Gradient clipping là kỹ thuật giới hạn độ lớn của gradient nhằm tránh cập nhật quá mức.

Theo tài liệu , thay vì cắt từng phần tử riêng lẻ, toàn bộ vector gradient được chuẩn hóa.

---

#### 3.1.2 Công thức toán học

Với ngưỡng $c$, gradient sau clipping:

$$

\mathbf{g}_{clip}=
\begin{cases}
\mathbf{g} & \text{nếu } |\mathbf{g}|\le c\
\frac{c}{|\mathbf{g}|}\mathbf{g} & \text{nếu } |\mathbf{g}|>c
\end{cases}

$$


Điều này đảm bảo:

$$

|\mathbf{g}_{clip}|\le c

$$


---

#### 3.1.3 Cập nhật tham số

Sau clipping:

$$

\theta_{t+1}=\theta_t-\eta \mathbf{g}_{clip}

$$


Việc này giúp giới hạn bước nhảy của tham số.

---

### 3.2 Learning Rate Scheduler

#### 3.2.1 Khái niệm

Learning rate scheduler là kỹ thuật thay đổi learning rate theo thời gian huấn luyện.

Theo , việc duy trì learning rate cố định có thể làm giảm hiệu quả học với mô hình lớn.

---

#### 3.2.2 Warm-up

Trong giai đoạn khởi động:

$$

\eta_t=\eta_{max}\cdot\frac{t}{T_{warm}}

$$


Trong đó:

* $T_{warm}$: số epoch warm-up
* $\eta_{max}$: learning rate cực đại

---

#### 3.2.3 Cosine Scheduler

Hàm cosine decay:

$$

\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})\left(1+\cos\frac{\pi t}{T}\right)

$$


Trong đó:

* $T$: tổng số epoch
* $\eta_{min}$: learning rate tối thiểu

---

#### 3.2.4 Linear Scheduler

Giảm tuyến tính:

$$

\eta_t=\eta_{max}\left(1-\frac{t}{T}\right)

$$


---

### 3.3 Kết hợp Clipping và Scheduler

Quy trình huấn luyện:

1. Tính gradient
2. Áp dụng clipping
3. Cập nhật learning rate
4. Cập nhật tham số

$$

\theta_{t+1}=\theta_t-\eta_t\cdot \mathbf{g}_{clip}

$$


---

## 4. Thực nghiệm

### 4.1 Mô hình minh họa

Theo mô tả trong tài liệu , mô hình gồm:

* Hai tham số trọng số
* Hàm mất mát L2
* SGD optimizer

Loss function:

$$

L=\sum_{i=1}^{n}w_i^2

$$


---

### 4.2 Ảnh hưởng của Gradient Clipping

| Trạng thái     | Chuẩn Gradient | Tốc độ học                |
| -------------- | -------------- | ------------------------- |
| Không clipping | > 10           | Nhanh nhưng không ổn định |
| Có clipping    | = 1            | Chậm, ổn định             |

Clipping giúp giảm hiện tượng gradient explosion nhưng làm chậm tốc độ hội tụ.

---

### 4.3 Ảnh hưởng của Scheduler

Kết quả cho thấy:

* Giai đoạn đầu: học ổn định
* Giai đoạn sau: giảm dao động
* Tránh overfitting

Learning curve mượt hơn khi dùng scheduler.

---

### 4.4 So sánh tổng hợp

| Phương pháp   | Ổn định    | Hội tụ     | Hiệu quả   |
| ------------- | ---------- | ---------- | ---------- |
| Không dùng    | Thấp       | Kém        | Trung bình |
| Chỉ clipping  | Trung bình | Trung bình | Tốt        |
| Chỉ scheduler | Tốt        | Tốt        | Tốt        |
| Kết hợp       | Rất tốt    | Cao        | Rất tốt    |

---

## 5. Thảo luận

### 5.1 Lợi ích của Gradient Clipping

Theo phân tích từ :

* Ngăn gradient explosion
* Ổn định số học
* Phù hợp mô hình lớn

Tuy nhiên, làm mất thông tin về độ lớn gradient.

---

### 5.2 Vai trò của Learning Rate Scheduler

Scheduler giúp:

* Tránh cập nhật quá mạnh ban đầu
* Tinh chỉnh ở giai đoạn cuối
* Cải thiện khả năng hội tụ

Đặc biệt hiệu quả với Transformer và LLM.

---

### 5.3 Hạn chế

* Cần tinh chỉnh siêu tham số
* Không phù hợp mô hình nhỏ
* Có thể làm chậm huấn luyện

Do đó cần lựa chọn phù hợp với bài toán.

---

## 6. Kết luận

Bài viết đã trình bày cơ sở lý thuyết và thực nghiệm của Gradient Clipping và Learning Rate Scheduler trong huấn luyện học sâu.

Kết quả cho thấy:

* Gradient Clipping giúp ổn định quá trình tối ưu
* Scheduler cải thiện hội tụ
* Kết hợp hai phương pháp cho hiệu quả cao nhất

Các kỹ thuật này đặc biệt quan trọng trong huấn luyện mô hình lớn và hệ thống AI hiện đại.

---

## Tài liệu tham khảo

1. Gradient Clipping and Learning Rate Scheduler Tutorial 
2. Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep Learning. MIT Press.
3. Loshchilov, I., Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts.
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
| 📌 **[📘 Ứng Dụng Gradient Clipping và Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu](aero_llm_017_gradient_clipping_and_learning_rate_scheduler_part_1_en_us.md)** | [Xem bài viết →](aero_llm_017_gradient_clipping_and_learning_rate_scheduler_part_1_en_us.md) |
| [📘 Phân Tích Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu Quy Mô Lớn](aero_llm_018_gradient_clipping_and_learning_rate_scheduler_part_2_.md) | [Xem bài viết →](aero_llm_018_gradient_clipping_and_learning_rate_scheduler_part_2_.md) |
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
