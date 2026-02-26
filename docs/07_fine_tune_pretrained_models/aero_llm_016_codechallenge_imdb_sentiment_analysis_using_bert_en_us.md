
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
# 📘 Ứng Dụng Mô Hình BERT Trong Phân Tích Cảm Xúc Đánh Giá Phim IMDB

## Tóm tắt (Abstract)

Bài viết này trình bày phương pháp ứng dụng mô hình BERT trong bài toán phân tích cảm xúc (sentiment analysis) đối với tập dữ liệu đánh giá phim IMDB. Dựa trên kỹ thuật fine-tuning có 강조 freeze một phần tham số, nghiên cứu tập trung vào việc tối ưu hiệu suất phân loại với chi phí tính toán thấp. Kết quả thực nghiệm cho thấy mô hình đạt độ chính xác xấp xỉ 90%.

---

## 1. Giới thiệu

Phân tích cảm xúc là một bài toán quan trọng trong xử lý ngôn ngữ tự nhiên (NLP), nhằm xác định thái độ tích cực hoặc tiêu cực của văn bản. Với sự phát triển của mô hình Transformer, đặc biệt là BERT, hiệu quả của các hệ thống phân loại văn bản đã được cải thiện đáng kể.

Theo tài liệu huấn luyện , bài toán được xây dựng dựa trên việc huấn luyện bộ phân loại nhằm dự đoán cảm xúc người xem thông qua nội dung đánh giá phim.

Mục tiêu nghiên cứu gồm:

* Xây dựng mô hình phân loại dựa trên BERT.
* Áp dụng kỹ thuật đóng băng (freeze) một phần tham số.
* Đánh giá hiệu quả huấn luyện.
* Phân tích sự ổn định của mô hình.

---

## 2. Cơ sở lý thuyết

### 2.1 Kiến trúc BERT

BERT (Bidirectional Encoder Representations from Transformers) sử dụng kiến trúc Transformer Encoder nhiều tầng.

Mỗi tầng gồm:

* Self-Attention
* Feedforward Neural Network (MLP)
* Layer Normalization
* Residual Connection

Công thức Attention:

$$

\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

$$


Trong đó:

* $Q$: Query
* $K$: Key
* $V$: Value
* $d_k$: kích thước vector

---

### 2.2 Lớp MLP trong BERT

Mỗi encoder layer chứa mạng MLP hai tầng:

$$

\text{MLP}(x)=W_2 \cdot \sigma(W_1 x + b_1)+b_2

$$


Trong đó:

* (W_1,W_2): ma trận trọng số
* (b_1,b_2): bias
* $\sigma$: hàm kích hoạt (GELU)

MLP giúp ánh xạ dữ liệu sang không gian đặc trưng phi tuyến.

---

### 2.3 Hàm mất mát phân loại

Bài toán phân loại nhị phân sử dụng hàm Cross-Entropy:

$$

L=-\frac{1}{N}\sum_{i=1}^{N} \left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]

$$


Trong đó:

* $y_i$: nhãn thật
* $p_i$: xác suất dự đoán
* $N$: số mẫu

---

## 3. Phương pháp nghiên cứu

### 3.1 Tập dữ liệu

Dữ liệu gồm 15.000 đánh giá phim, được chia đều:

* 50% tích cực
* 50% tiêu cực

Theo mô tả trong tài liệu gốc , dữ liệu được tiền xử lý bằng tokenizer của BERT và đưa vào DataLoader.

---

### 3.2 Đóng băng tham số (Freezing)

Chiến lược huấn luyện:

* Đóng băng:

  * Embedding layer
  * Attention layers
* Huấn luyện:

  * MLP layers
  * Pooler layer
  * Classifier head

Điều kiện đóng băng:

$$

\text{requires_grad}=False

$$


Việc này giúp:

* Giảm số tham số cần cập nhật
* Giảm overfitting
* Tăng tốc huấn luyện

---

### 3.3 Tỷ lệ tham số huấn luyện

Số tham số được tính:

$$

P_{total}=\sum_i |W_i|

$$


$$

P_{trainable}=\sum_{j \in T}|W_j|

$$


$$

R=\frac{P_{trainable}}{P_{total}}

$$


Trong đó:

* $T$: tập tham số được huấn luyện
* $R$: tỷ lệ trainable

Kết quả cho thấy:

$$

R \approx 0.5

$$


Tức khoảng 50% tham số được cập nhật.

---

### 3.4 Quy trình huấn luyện

Mô hình được huấn luyện trong 300 batch:

$$

\theta_{t+1}=\theta_t-\eta \nabla_\theta L(\theta)

$$


Trong đó:

* $\theta$: tham số mô hình
* $\eta$: learning rate
* $L$: hàm mất mát

Sau mỗi 10 batch, tiến hành đánh giá tập kiểm tra.

---

## 4. Kết quả thực nghiệm

### 4.1 Độ chính xác

Độ chính xác được tính:

$$

Accuracy=\frac{TP+TN}{TP+TN+FP+FN}

$$


Kết quả trung bình:

| Giai đoạn     | Accuracy |
| ------------- | -------- |
| Ban đầu       | ~50%     |
| Sau 100 batch | ~80%     |
| Sau 300 batch | ~90%     |

Theo báo cáo trong tài liệu , độ chính xác dao động mạnh trong giai đoạn đầu.

---

### 4.2 Hàm mất mát

Loss giảm theo thời gian:

$$

L_t \downarrow \quad \text{khi } t \uparrow

$$


Tuy nhiên xuất hiện dao động do:

* Batch nhỏ
* Learning rate cố định
* Dữ liệu ngẫu nhiên

---

### 4.3 Độ ổn định

Mô hình có hiện tượng:

* Accuracy dao động
* Loss không hội tụ mượt

Nguyên nhân:

* Gradient nhiễu
* Không dùng scheduler
* Không clipping gradient

---

## 5. Thảo luận

### 5.1 Lý do freeze Attention

Theo phân tích từ tài liệu :

* Attention học quan hệ token
* Đã được huấn luyện tốt
* Không cần tinh chỉnh nhiều

Ngược lại, MLP thích hợp cho:

* Phân tách tuyến tính
* Điều chỉnh theo nhiệm vụ cụ thể

---

### 5.2 So sánh chiến lược huấn luyện

| Phương pháp       | Hiệu quả | Ổn định    |
| ----------------- | -------- | ---------- |
| Fine-tune toàn bộ | Cao      | Thấp       |
| Freeze Attention  | Tốt      | Trung bình |
| Freeze toàn bộ    | Kém      | Cao        |

Việc chỉ huấn luyện classifier dẫn đến suy giảm nghiêm trọng hiệu suất.

---

### 5.3 Hướng cải tiến

Có thể cải thiện bằng:

* Learning rate scheduler
* Batch size lớn hơn
* Data augmentation
* Gradient clipping
* Regularization

---

## 6. Kết luận

Nghiên cứu đã trình bày phương pháp ứng dụng BERT cho phân tích cảm xúc IMDB thông qua chiến lược đóng băng tham số.

Kết quả cho thấy:

* Độ chính xác đạt ~90%
* Thời gian huấn luyện giảm
* Hiệu quả ổn định

Phương pháp này phù hợp với các hệ thống có tài nguyên hạn chế và dữ liệu vừa phải.

---

## Tài liệu tham khảo

1. Tài liệu huấn luyện BERT IMDB Sentiment Analysis 
2. Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
3. Vaswani, A. et al. (2017). Attention Is All You Need.

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
| 📌 **[📘 Ứng Dụng Mô Hình BERT Trong Phân Tích Cảm Xúc Đánh Giá Phim IMDB](aero_llm_016_codechallenge_imdb_sentiment_analysis_using_bert_en_us.md)** | [Xem bài viết →](aero_llm_016_codechallenge_imdb_sentiment_analysis_using_bert_en_us.md) |
| [📘 Ứng Dụng Gradient Clipping và Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu](aero_llm_017_gradient_clipping_and_learning_rate_scheduler_part_1_en_us.md) | [Xem bài viết →](aero_llm_017_gradient_clipping_and_learning_rate_scheduler_part_1_en_us.md) |
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
