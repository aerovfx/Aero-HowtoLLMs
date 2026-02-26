
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
# 📘 Kết Hợp Gradient Clipping, Freezing và Learning Rate Scheduler Trong Fine-Tuning Mô Hình BERT

## Tóm tắt (Abstract)

Fine-tuning các mô hình ngôn ngữ lớn như BERT cho bài toán phân loại văn bản thường gặp các vấn đề về tính ổn định và hội tụ. Ba kỹ thuật quan trọng gồm đóng băng tham số (freezing), cắt gradient (gradient clipping) và điều chỉnh tốc độ học (learning rate scheduler) được đề xuất nhằm cải thiện hiệu suất huấn luyện. Bài viết phân tích cơ sở lý thuyết, mô hình toán học và kết quả thực nghiệm của việc kết hợp ba phương pháp này trong bài toán phân tích cảm xúc đánh giá phim.

---

## 1. Giới thiệu

Các mô hình Transformer tiền huấn luyện như BERT đã trở thành nền tảng trong xử lý ngôn ngữ tự nhiên. Tuy nhiên, quá trình fine-tuning đòi hỏi:

* Kiểm soát số lượng tham số học
* Ổn định gradient
* Điều chỉnh tốc độ hội tụ

Theo tài liệu thực hành , việc kết hợp freezing, clipping và scheduling giúp tăng tính ổn định và hiệu quả huấn luyện.

Mục tiêu nghiên cứu:

* Phân tích vai trò từng kỹ thuật
* Xây dựng mô hình toán học tổng hợp
* Đánh giá tác động lên BERT
* Đề xuất hướng tối ưu

---

## 2. Cơ sở lý thuyết

### 2.1 Fine-tuning mô hình tiền huấn luyện

Cho mô hình tiền huấn luyện với tham số $\theta_0$. Fine-tuning nhằm tìm:

$$

\theta^*=\arg\min_{\theta}L(\theta;D_{task})

$$


Trong đó $D_{task}$ là tập dữ liệu mục tiêu.

---

### 2.2 Freezing tham số

Giả sử tập tham số được huấn luyện là $T\subset\theta$:

$$

\theta=\theta_{freeze}\cup\theta_{train},\quad
\theta_{freeze}\cap\theta_{train}=\emptyset

$$


Với:

$$

\nabla_{\theta_{freeze}}L=0

$$


⇒ các tham số bị đóng băng không cập nhật.

---

### 2.3 Gradient Descent

Quy trình cập nhật:

$$

\theta_{t+1}=\theta_t-\eta_t\mathbf{g}_t

$$


$$

\mathbf{g}*t=\nabla*\theta L(\theta_t)

$$


---

## 3. Phương pháp nghiên cứu

### 3.1 Chiến lược Freezing trong BERT

Theo , mô hình được cấu hình:

* Đóng băng: Embedding + Attention
* Huấn luyện: MLP + Pooler + Classifier

Tỷ lệ tham số:

$$

R=\frac{|\theta_{train}|}{|\theta_{total}|}\approx 0.5

$$


---

### 3.2 Gradient Clipping

#### 3.2.1 Chuẩn hóa gradient

Với ngưỡng $c=1$:

$$

\mathbf{g}'=
\frac{c}{\max(|\mathbf{g}|,c)}\mathbf{g}

$$


Đảm bảo:

$$

|\mathbf{g}'|\le c

$$


---

#### 3.2.2 Ảnh hưởng tới cập nhật

$$

\theta_{t+1}=\theta_t-\eta_t\mathbf{g}'

$$


Giúp hạn chế gradient explosion.

---

### 3.3 Learning Rate Scheduler

#### 3.3.1 Warm-up

$$

\eta_t=\eta_{max}\frac{t}{T_{warm}},\quad t\le T_{warm}

$$


---

#### 3.3.2 Linear Decay

$$

\eta_t=\eta_{max}\left(1-\frac{t}{T_{sched}}\right)

$$


Trong đó:

$$

T_{sched}>T_{train}

$$


để tránh $\eta_t=0$.

---

### 3.4 Quy trình tổng hợp

Quy trình huấn luyện:

1. Forward
2. Backprop
3. Ghi nhận gradient norm
4. Clipping
5. Scheduler
6. Update

Phương trình tổng quát:

$$

\theta_{t+1}=
\theta_t-
\eta_t
\frac{c}{\max(|\mathbf{g}_t|,c)}\mathbf{g}_t

$$


---

## 4. Thực nghiệm

### 4.1 Thiết lập

Theo :

* 300 batch huấn luyện
* Warm-up 5%
* Linear scheduler (450 steps)
* Clipping: $c=1$

Theo dõi:

* Loss
* Accuracy
* Gradient norm

---

### 4.2 Phân tích hàm mất mát

Cross-Entropy:

$$

L=-\sum_{i=1}^{N}y_i\log(p_i)

$$


Quan sát:

$$

Var(L_{clip+sch})<Var(L_{baseline})

$$


⇒ học ổn định hơn.

---

### 4.3 Độ chính xác

Accuracy:

$$

Acc=\frac{TP+TN}{TP+TN+FP+FN}

$$


Kết quả:

| Giai đoạn    | Accuracy |
| ------------ | -------- |
| Trước tối ưu | ~85%     |
| Sau tối ưu   | ~90%     |

---

### 4.4 Phân tích Gradient Norm

Hai lớp được theo dõi:

* MLP layer (pre-trained)
* Classifier layer (random)

Chuẩn gradient:

$$

G_t=|\nabla W_t|

$$


Quan sát:

$$

G_{MLP}<1 \quad (\text{đa số})

$$


$$

G_{CLS}>1 \quad (\text{nhiều giai đoạn đầu})

$$


⇒ Clipping ảnh hưởng mạnh đến classifier.

---

### 4.5 Hiện tượng mất thông tin Gradient

Lượng thông tin bị mất:

$$

\Delta g=
|\mathbf{g}|-|\mathbf{g}'|

$$


Với:

$$

|\mathbf{g}|>1

$$


⇒ $\Delta g>0$

Đặc biệt lớn ở giai đoạn đầu.

---

## 5. Thảo luận

### 5.1 Đánh giá tính phù hợp của Clipping

Theo , clipping sớm có thể:

* Giảm tốc độ học
* Làm chậm classifier

Giải pháp:

$$

c(t)=
\begin{cases}
\infty & t<T_0\
1 & t\ge T_0
\end{cases}

$$


(Delayed clipping)

---

### 5.2 Tương tác giữa các kỹ thuật

Ba kỹ thuật phối hợp:

| Kỹ thuật  | Vai trò      |
| --------- | ------------ |
| Freezing  | Giảm tham số |
| Clipping  | Ổn định      |
| Scheduler | Hội tụ       |

Tác động tổng hợp:

$$

Stability\propto f(F,C,S)

$$


---

### 5.3 Ứng dụng cho LLM

Kết quả cho thấy:

* Cần thiết cho mô hình >1B tham số
* Giảm rủi ro divergence
* Tăng khả năng tái lập

---

## 6. Kết luận

Nghiên cứu đã phân tích việc kết hợp freezing, gradient clipping và learning rate scheduler trong fine-tuning BERT.

Kết quả chính:

* Loss ổn định hơn
* Accuracy tăng
* Gradient được kiểm soát
* Hội tụ nhanh hơn

Phương pháp phù hợp cho huấn luyện mô hình ngôn ngữ lớn trong điều kiện tài nguyên hạn chế.

---

## Tài liệu tham khảo

1. BERT Fine-Tuning Code Challenge 
2. Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
3. Goodfellow, I. et al. (2016). Deep Learning. MIT Press.
4. Loshchilov, I., Hutter, F. (2017). SGDR.

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
| [📘 Phân Tích Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu Quy Mô Lớn](aero_llm_018_gradient_clipping_and_learning_rate_scheduler_part_2_.md) | [Xem bài viết →](aero_llm_018_gradient_clipping_and_learning_rate_scheduler_part_2_.md) |
| 📌 **[📘 Kết Hợp Gradient Clipping, Freezing và Learning Rate Scheduler Trong Fine-Tuning Mô Hình BERT](aero_llm_019_codechallenge_clip_freeze_and_schedule_bert.md)** | [Xem bài viết →](aero_llm_019_codechallenge_clip_freeze_and_schedule_bert.md) |
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
