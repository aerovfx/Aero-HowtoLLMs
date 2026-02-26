
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [06 pretraining](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
Dưới đây là **bài viết khoa học dưới dạng Markdown**, được xây dựng dựa trên tài liệu **“CodeChallenge Add a Test Set”**, có bổ sung phân tích học thuật và trích dẫn nguồn.

---

# 📘 Thiết Lập Tập Kiểm Thử Trong Huấn Luyện Mô Hình Ngôn Ngữ: Phân Tích Phương Pháp Train–Test Split và Đánh Giá Hiệu Suất

---

## Tóm tắt (Abstract)

Trong huấn luyện mô hình học sâu, việc đánh giá hiệu suất thông qua tập kiểm thử độc lập là yếu tố then chốt để đảm bảo khả năng tổng quát hóa. Bài viết này phân tích phương pháp chia tập dữ liệu huấn luyện và kiểm thử (train–test split) trong bối cảnh huấn luyện mô hình ngôn ngữ đơn giản. Dựa trên tài liệu *CodeChallenge Add a Test Set*, nghiên cứu trình bày cách xây dựng tập dữ liệu, thiết kế vòng lặp đánh giá, và phân tích mối quan hệ giữa train loss và test loss. Kết quả cho thấy việc sử dụng tập kiểm thử giúp phát hiện hiện tượng overfitting và cung cấp thước đo khách quan về chất lượng mô hình. 

---

## 1. Giới thiệu (Introduction)

Trong các nghiên cứu học máy, mô hình thường đạt hiệu suất cao trên dữ liệu huấn luyện do có xu hướng ghi nhớ (memorization). Tuy nhiên, hiệu suất này không phản ánh chính xác khả năng tổng quát hóa. Để giải quyết vấn đề này, tập kiểm thử được sử dụng như một cơ chế đánh giá độc lập.

Tài liệu *CodeChallenge Add a Test Set* chỉ ra rằng trong các bài thực hành ban đầu, mô hình được huấn luyện trên toàn bộ dữ liệu mà không có bước đánh giá riêng biệt, dẫn đến nguy cơ overfitting. Do đó, việc bổ sung tập test là cần thiết nhằm nâng cao tính khoa học của quá trình huấn luyện. 

Bài viết này tập trung phân tích:

* Khái niệm train–test split,
* Quy trình xây dựng tập dữ liệu,
* Phương pháp đánh giá mô hình,
* Hiện tượng khác biệt giữa train loss và test loss.

---

## 2. Cơ Sở Lý Thuyết (Theoretical Background)

### 2.1. Overfitting và Khả Năng Tổng Quát Hóa

Overfitting xảy ra khi mô hình học quá kỹ dữ liệu huấn luyện và mất khả năng dự đoán dữ liệu mới. Hiện tượng này thường xuất hiện khi:

* Tập dữ liệu nhỏ,
* Số epoch lớn,
* Mô hình có độ phức tạp cao.

Theo tài liệu, mô hình có thể ghi nhớ chuỗi token thay vì học đặc trưng thống kê tổng quát của ngôn ngữ. 

---

### 2.2. Train–Test Split

Train–test split là phương pháp chia dữ liệu thành hai phần:

* **Training set**: dùng để huấn luyện,
* **Test set**: dùng để đánh giá.

Trong nghiên cứu này, dữ liệu được chia theo tỷ lệ:

[
90% \text{ training} \quad + \quad 10% \text{ testing}
]

Cách tiếp cận này giúp đảm bảo tập test chưa từng được mô hình quan sát trong quá trình huấn luyện. 

---

## 3. Phương Pháp Thực Nghiệm (Methodology)

### 3.1. Dữ Liệu và Tokenization

Nguồn dữ liệu là tác phẩm *The Time Machine* từ Gutenberg Project, được mã hóa bằng tokenizer GPT-2. Sau khi token hóa, dữ liệu được chuyển thành tensor PyTorch để phục vụ huấn luyện. 

---

### 3.2. Xây Dựng Dataset

Mỗi mẫu dữ liệu gồm một chuỗi 8 token đầu vào và token mục tiêu tương ứng. Cấu trúc dữ liệu có dạng:

[
(x_1, x_2, \dots, x_8) \rightarrow (x_2, x_3, \dots, x_9)
]

Cách xây dựng này phù hợp với bài toán dự đoán token tiếp theo. 

---

### 3.3. Phân Chia Dữ Liệu

Việc phân chia được thực hiện bằng hàm `random_split` trong PyTorch:

* 21.000 chuỗi cho training,
* 2.500 chuỗi cho testing.

Việc chia ngẫu nhiên giúp giảm rủi ro thiên lệch dữ liệu. 

---

### 3.4. DataLoader và Shuffle

Hai DataLoader được sử dụng:

| Tập dữ liệu | Shuffle |
| ----------- | ------- |
| Training    | True    |
| Testing     | False   |

Shuffle được áp dụng cho tập training nhằm tránh học theo thứ tự văn bản. Tập test không shuffle để đảm bảo tính ổn định trong đánh giá. 

---

## 4. Đánh Giá Đầu Ra Mô Hình (Output Evaluation)

### 4.1. Điều Kiện Phân Phối Xác Suất

Một phân phối xác suất phải thỏa mãn:

1. Không âm,
2. Tổng bằng 1.

Kết quả mô hình ban đầu là log-softmax, nên chưa phải phân phối xác suất trực tiếp. 

---

### 4.2. Chuyển Đổi Log-Probability

Phân phối xác suất được khôi phục bằng:

[
P = e^{\log p}
]

Sau chuyển đổi, mỗi hàng của ma trận đầu ra có tổng bằng 1, xác nhận tính hợp lệ. 

---

## 5. Quy Trình Huấn Luyện và Đánh Giá (Training and Evaluation)

### 5.1. Huấn Luyện Trên GPU

Toàn bộ mô hình và dữ liệu được chuyển sang GPU nhằm tăng tốc tính toán. Optimizer tự động kế thừa thiết bị từ tham số mô hình. 

---

### 5.2. Vòng Lặp Huấn Luyện

Mỗi epoch gồm:

1. Forward pass,
2. Tính loss,
3. Backpropagation,
4. Cập nhật tham số.

Mô hình được huấn luyện trong 10 epochs. 

---

### 5.3. Đánh Giá Với `torch.no_grad()`

Trong quá trình đánh giá, gradient được vô hiệu hóa bằng:

```python
with torch.no_grad():
```

Cơ chế này giúp:

* Giảm chi phí tính toán,
* Tiết kiệm bộ nhớ,
* Tăng tốc độ suy luận.



---

## 6. Kết Quả Thực Nghiệm (Results)

### 6.1. Diễn Biến Loss

Sau 10 epochs, loss giảm dần và hội tụ quanh giá trị 3.

| Epoch | Train Loss | Test Loss |
| ----- | ---------- | --------- |
| 1     | Cao        | Thấp hơn  |
| 10    | ≈ 3        | ≈ 3       |

Kết quả này cho thấy mô hình đã học được các đặc trưng cơ bản của dữ liệu. 

---

### 6.2. Hiện Tượng Test Loss < Train Loss

Quan sát cho thấy ở giai đoạn đầu:

[
Loss_{test} < Loss_{train}
]

Nguyên nhân là do train loss trung bình bao gồm giai đoạn đầu khi mô hình chưa học được gì, trong khi test loss được tính sau khi mô hình đã cải thiện. 

---

## 7. Thảo Luận (Discussion)

### 7.1. Vai Trò Của Tập Kiểm Thử

Tập test giúp:

* Phát hiện overfitting,
* Đánh giá khả năng tổng quát hóa,
* So sánh các cấu hình mô hình.

Mặc dù LLM ít bị overfitting hơn do dữ liệu lớn, tập test vẫn đóng vai trò quan trọng trong nghiên cứu thực nghiệm. 

---

### 7.2. Ý Nghĩa Thực Tiễn

Phương pháp được trình bày phù hợp cho:

* Sinh viên học deep learning,
* Nghiên cứu NLP cơ bản,
* Xây dựng pipeline huấn luyện chuẩn hóa.

Nó cung cấp nền tảng cho việc phát triển mô hình Transformer sau này.

---

### 7.3. Hạn Chế

Một số hạn chế chính:

* Dataset nhỏ,
* Context ngắn (8 token),
* Không có attention mechanism.

Do đó, kết quả chỉ mang tính minh họa.

---

## 8. Kết Luận (Conclusion)

Bài viết đã phân tích phương pháp bổ sung tập kiểm thử trong huấn luyện mô hình ngôn ngữ. Các kết luận chính bao gồm:

1. Train–test split giúp đánh giá khách quan hiệu suất mô hình.
2. `torch.no_grad()` là công cụ quan trọng trong đánh giá.
3. Test loss phản ánh tốt khả năng tổng quát hóa.
4. Hiện tượng test loss thấp hơn train loss ở giai đoạn đầu là hợp lý.
5. Mô hình phù hợp cho mục đích giảng dạy và nghiên cứu cơ bản.

Nghiên cứu này đặt nền móng cho việc xây dựng hệ thống huấn luyện LLM chuẩn hóa trong các giai đoạn tiếp theo.

---

## Tài Liệu Tham Khảo (References)

[1] CodeChallenge Add a Test Set, Lecture Transcript.


---
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📘 Huấn Luyện Mô Hình Ngôn Ngữ Với Thiên Lệch Có Chủ Đích Bằng KL-Divergence: Một Nghiên Cứu Thực Nghiệm](aero_llm_010_codechallenge_train_a_model_to_like_x.md) | [Xem bài viết →](aero_llm_010_codechallenge_train_a_model_to_like_x.md) |
| [📘 Các Vấn Đề Tỷ Lệ Số Học Trong Mô Hình Học Sâu: Phân Tích Vai Trò Của Scaling và Normalization Trong Cơ Chế Attention](aero_llm_011_codechallenge_numerical_scaling_issues_in_dl_models_copy_2.md) | [Xem bài viết →](aero_llm_011_codechallenge_numerical_scaling_issues_in_dl_models_copy_2.md) |
| [Weight Initialization and Numerical Stability in Large Language Models](aero_llm_012_weight_initializations.md) | [Xem bài viết →](aero_llm_012_weight_initializations.md) |
| [Phân Tích Ảnh Hưởng Của Khởi Tạo Trọng Số Và Sự Tiến Hóa Phân Phối Tham Số Trong Quá Trình Huấn Luyện Mô Hình Transformer](aero_llm_013_codechallenge_train_model_5_with_weight_inits.md) | [Xem bài viết →](aero_llm_013_codechallenge_train_model_5_with_weight_inits.md) |
| [Dropout as a Regularization Mechanism in Large Language Models: Theory, Implementation, and Practical Implications](aero_llm_014_dropout_in_theory_and_in_pytorch.md) | [Xem bài viết →](aero_llm_014_dropout_in_theory_and_in_pytorch.md) |
| [So Sánh Đầu Ra Logits và Log-Softmax Trong Mô Hình Ngôn Ngữ: Tác Động Đến Huấn Luyện và Sinh Văn Bản](aero_llm_015_should_you_output_logits_or_log_softmax_logits_.md) | [Xem bài viết →](aero_llm_015_should_you_output_logits_or_log_softmax_logits_.md) |
| [aero llm 016 the fineweb dataset](aero_llm_016_the_fineweb_dataset.md) | [Xem bài viết →](aero_llm_016_the_fineweb_dataset.md) |
| [Tích Hợp Dropout Trong Mô Hình Ngôn Ngữ Transformer: Phân Tích Trường Hợp Model 5](aero_llm_017_codechallenge_fine_dropout_in_model_5_part_1.md) | [Xem bài viết →](aero_llm_017_codechallenge_fine_dropout_in_model_5_part_1.md) |
| [Chiến Lược Huấn Luyện Dựa Trên Final-Token Loss Trong Mô Hình Transformer: Phân Tích Trường Hợp Model 5 Với Dropout](aero_llm_018_codechallenge_fine_dropout_in_model_5_part_2_.md) | [Xem bài viết →](aero_llm_018_codechallenge_fine_dropout_in_model_5_part_2_.md) |
| [Phân Tích Hành Vi Học Biểu Diễn Token Trong Mô Hình Ngôn Ngữ Lớn](aero_llm_019_codechallenge_what_happens_to_unused_tokens_.md) | [Xem bài viết →](aero_llm_019_codechallenge_what_happens_to_unused_tokens_.md) |
| [📘 Vai Trò Của Pre-training Trong Mô Hình Ngôn Ngữ Lớn: Phân Tích Chi Phí, Hiệu Quả và Tính Ứng Dụng](aero_llm_01_what_is_pretraining.md) | [Xem bài viết →](aero_llm_01_what_is_pretraining.md) |
| [Tối Ưu Hóa Quá Trình Tiền Huấn Luyện Mô Hình Ngôn Ngữ Lớn: Phân Tích Các Chiến Lược Tính Toán và Học Tập](aero_llm_020_optimization_options.md) | [Xem bài viết →](aero_llm_020_optimization_options.md) |
| [📘 Nền Tảng Hugging Face Trong Hệ Sinh Thái Trí Tuệ Nhân Tạo: Vai Trò, Cấu Trúc và Ứng Dụng Trong Nghiên Cứu Mô Hình Ngôn Ngữ](aero_llm_02_huggingface.md) | [Xem bài viết →](aero_llm_02_huggingface.md) |
| [📘 Thuật Toán Tối Ưu AdamW Trong Huấn Luyện Mô Hình Học Sâu: Cơ Sở Lý Thuyết, Cải Tiến và Ứng Dụng](aero_llm_03_the_adamw_optimizer.md) | [Xem bài viết →](aero_llm_03_the_adamw_optimizer.md) |
| [📘 So Sánh SGD, Adam và AdamW Trong Huấn Luyện Mô Hình Học Sâu: Phân Tích Thực Nghiệm và Ứng Dụng](aero_llm_04_codechallenge_sgd_vs_adam_vs_adamw_.md) | [Xem bài viết →](aero_llm_04_codechallenge_sgd_vs_adam_vs_adamw_.md) |
| [📘 Huấn Luyện Mô Hình Ngôn Ngữ Đơn Giản Bằng PyTorch: Phân Tích Quy Trình, Động Lực Học và Hiệu Suất Thực Nghiệm](aero_llm_05_train_model.md) | [Xem bài viết →](aero_llm_05_train_model.md) |
| 📌 **[📘 Thiết Lập Tập Kiểm Thử Trong Huấn Luyện Mô Hình Ngôn Ngữ: Phân Tích Phương Pháp Train–Test Split và Đánh Giá Hiệu Suất](aero_llm_06_codechallenge_add_a_test_set.md)** | [Xem bài viết →](aero_llm_06_codechallenge_add_a_test_set.md) |
| [📘 Chuyển Giao Trọng Số và Đóng Băng Tham Số Trong Huấn Luyện Mô Hình Ngôn Ngữ: Phân Tích Thực Nghiệm Với Embedding GPT-2](aero_llm_07_codechallenge_train_model_1_with_gpt2_s_embeddings.md) | [Xem bài viết →](aero_llm_07_codechallenge_train_model_1_with_gpt2_s_embeddings.md) |
| [📘 Phương Pháp Lấy Mẫu Ngẫu Nhiên và Huấn Luyện Mô Hình GPT-2 Thu Gọn: Phân Tích Thực Nghiệm Với Dữ Liệu Văn Bản Cổ Điển](aero_llm_08_codechallenge_train_model_5_with_modifications.md) | [Xem bài viết →](aero_llm_08_codechallenge_train_model_5_with_modifications.md) |
| [Thiết Kế Hàm Mất Mát Tùy Biến Trong Huấn Luyện Mô Hình Ngôn Ngữ Lớn](aero_llm_09_create_a_custom_loss_function.md) | [Xem bài viết →](aero_llm_09_create_a_custom_loss_function.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
