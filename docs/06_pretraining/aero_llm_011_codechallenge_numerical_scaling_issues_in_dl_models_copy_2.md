
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
# 📘 Các Vấn Đề Tỷ Lệ Số Học Trong Mô Hình Học Sâu: Phân Tích Vai Trò Của Scaling và Normalization Trong Cơ Chế Attention

---

## **Abstract**

Trong các mô hình học sâu hiện đại, đặc biệt là các mô hình dựa trên Transformer, việc kiểm soát độ lớn của giá trị số học đóng vai trò quan trọng trong đảm bảo tính ổn định và hiệu quả huấn luyện. Bài viết này phân tích các vấn đề liên quan đến việc nhân ma trận, sự khuếch đại phương sai, và ảnh hưởng của chúng đến hàm Softmax trong cơ chế attention. Dựa trên tài liệu *CodeChallenge: Numerical Scaling Issues in DL Models*, nghiên cứu làm rõ lý do cần chuẩn hóa tích QKᵀ bằng căn bậc hai của chiều không gian, đồng thời khảo sát phân phối tham số Layer Normalization trong GPT-2. Kết quả cho thấy scaling và normalization là các thành phần thiết yếu nhằm duy trì “vùng Goldilocks” cho logits trong quá trình học. 

---

## **1. Introduction**

Các mô hình ngôn ngữ lớn (Large Language Models – LLMs) dựa trên kiến trúc Transformer sử dụng hàng triệu phép nhân ma trận trong mỗi bước suy luận. Mặc dù các phép toán này giúp mô hình học được biểu diễn phức tạp, chúng cũng gây ra hiện tượng khuếch đại giá trị số học.

Theo tài liệu, Softmax là một phép biến đổi mạnh nhưng rất nhạy cảm với độ lớn của đầu vào. Khi logits có giá trị quá lớn, phân phối xác suất trở nên cực đoan, làm suy giảm khả năng học của mô hình. Do đó, việc nghiên cứu các vấn đề scaling là cần thiết để hiểu rõ cơ chế hoạt động của attention. 

Bài viết tập trung phân tích:

* Ảnh hưởng của nhân ma trận đến phương sai,
* Lý do cần scaling trong attention,
* Tác động đến Softmax,
* Vai trò của Layer Normalization.

---

## **2. Theoretical Background**

### **2.1. Dot Product trong Attention**

Trong cơ chế self-attention, điểm tương đồng giữa Query và Key được tính bằng:

[
A = QK^T
]

Mỗi phần tử của (A) là tích vô hướng của hai vector có chiều (d).

Nếu các phần tử của (Q) và (K) có phân phối chuẩn với phương sai bằng 1, thì phương sai của tích vô hướng xấp xỉ:

[
Var(QK^T) \approx d
]

Do đó, độ lệch chuẩn xấp xỉ:

[
\sigma \approx \sqrt{d}
]



---

### **2.2. Softmax và Độ Nhạy Số Học**

Hàm Softmax được định nghĩa:

[
Softmax(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
]

Khi (z_i) lớn, hàm mũ làm cho một số phần tử chiếm ưu thế tuyệt đối, dẫn đến:

* Hiện tượng bão hòa,
* Gradient gần bằng 0,
* Giảm khả năng học.

Theo tài liệu, đây là nguyên nhân chính khiến logits cần được kiểm soát về mặt số học. 

---

### **2.3. Scaling trong Attention**

Để giảm phương sai của (QK^T), Transformer áp dụng phép chia:

[
A_{scaled} = \frac{QK^T}{\sqrt{d}}
]

Phép scaling này đưa độ lệch chuẩn của ma trận attention về xấp xỉ 1, giúp Softmax hoạt động trong vùng ổn định. 

---

## **3. Methodology**

### **3.1. Thí Nghiệm 1: Ma Trận Ngẫu Nhiên**

Hai ma trận (Q, K \in \mathbb{R}^{50 \times 50}) được sinh từ phân phối Gaussian chuẩn.

Các đại lượng được tính:

* (\sigma(Q)),
* (\sigma(K)),
* (\sigma(QK^T)),
* (\sqrt{50}).

Kết quả cho thấy:

[
\sigma(QK^T) \approx \sqrt{50} \approx 7
]



---

### **3.2. Thí Nghiệm 2: Thay Đổi Chiều Không Gian**

Ma trận có kích thước (50 \times n), với (n) từ 2 đến 100.

Mỗi lần lặp, tính:

* Độ lệch chuẩn của (QK^T),
* Giá trị (\sqrt{n}).

Hai đại lượng này được so sánh bằng biểu đồ.

Kết quả cho thấy sự trùng khớp gần như hoàn hảo giữa lý thuyết và thực nghiệm. 

---

### **3.3. Thí Nghiệm 3: Softmax Trước và Sau Scaling**

Thí nghiệm này so sánh:

1. Softmax của (QK^T),
2. Softmax của (\frac{QK^T}{\sqrt{d}}),
3. Negative log-softmax tương ứng.

Các giá trị được trực quan hóa bằng scatter plot.

Mục tiêu là đánh giá ảnh hưởng của scaling đến phân phối xác suất. 

---

### **3.4. Thí Nghiệm 4: Phân Tích Layer Norm Trong GPT-2**

Tất cả tham số Layer Normalization của GPT-2 được trích xuất:

* Weight (γ – stretching),
* Bias (β – shifting).

Các giá trị này được biểu diễn bằng histogram với trục y ở dạng log-scale. 

---

## **4. Experimental Results**

### **4.1. Khuếch Đại Phương Sai Khi Nhân Ma Trận**

Kết quả cho thấy:

* (\sigma(Q) \approx 1),
* (\sigma(K) \approx 1),
* (\sigma(QK^T) \approx \sqrt{d}).

Điều này chứng minh rằng nhân ma trận làm tăng phương sai theo chiều không gian. 

---

### **4.2. Ảnh Hưởng Đến Softmax**

Trước scaling:

* Chỉ một vài token có xác suất lớn,
* Phần lớn xác suất ≈ 0.

Sau scaling:

* Phân phối trải đều hơn,
* Nhiều token có cơ hội được chọn.

Hiện tượng này giúp mô hình học đa dạng hơn ở giai đoạn đầu. 

---

### **4.3. Phân Phối Tham Số Layer Norm**

Phân tích GPT-2 cho thấy:

* Tham số γ chủ yếu nằm trong khoảng 0.2–0.4,
* Tham số β tập trung quanh 0.

Điều này cho thấy Layer Norm chủ yếu có tác dụng thu nhỏ (shrink) activation. 

---

## **5. Discussion**

### **5.1. Vùng “Goldilocks” Của Logits**

Theo tài liệu, logits cần nằm trong một vùng trung gian:

* Không quá lớn → tránh bão hòa,
* Không quá nhỏ → tránh mất phân biệt.

Scaling và normalization giúp duy trì vùng này. 

---

### **5.2. Vai Trò Của Normalization**

Layer Normalization giúp:

* Ổn định gradient,
* Giảm drift của activation,
* Cân bằng giữa các tầng.

Nó là thành phần không thể thiếu trong Transformer.

---

### **5.3. Liên Hệ Với Temperature Sampling**

Scaling trong attention có vai trò tương tự tham số temperature (T):

[
P_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
]

Cả hai đều điều chỉnh độ “sắc nét” của phân phối. 

---

## **6. Limitations**

Nghiên cứu còn tồn tại một số hạn chế:

* Chủ yếu dựa trên dữ liệu ngẫu nhiên,
* Chưa đánh giá ảnh hưởng đến downstream tasks,
* Chỉ khảo sát GPT-2,
* Không so sánh với các kiến trúc khác.

Do đó, kết quả mang tính minh họa nhiều hơn tổng quát.

---

## **7. Conclusion**

Bài viết đã phân tích các vấn đề scaling số học trong mô hình học sâu và cơ chế attention. Các kết luận chính gồm:

1. Nhân ma trận làm tăng phương sai theo (\sqrt{d}).
2. Scaling là cần thiết để ổn định Softmax.
3. Không scaling dẫn đến phân phối xác suất cực đoan.
4. Layer Norm giúp kiểm soát biên độ activation.
5. Các cơ chế này phối hợp để đảm bảo tính ổn định số học.

Nghiên cứu khẳng định rằng kiểm soát tỷ lệ số học là nền tảng cho việc huấn luyện thành công các mô hình Transformer quy mô lớn.

---

## **References**

1. CodeChallenge: Numerical Scaling Issues in DL Models. Lecture Transcript.

2. Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
3. Ba et al. (2016). Layer Normalization. *arXiv*.
4. Goodfellow et al. (2016). *Deep Learning*. MIT Press.

---
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📘 Huấn Luyện Mô Hình Ngôn Ngữ Với Thiên Lệch Có Chủ Đích Bằng KL-Divergence: Một Nghiên Cứu Thực Nghiệm](aero_llm_010_codechallenge_train_a_model_to_like_x.md) | [Xem bài viết →](aero_llm_010_codechallenge_train_a_model_to_like_x.md) |
| 📌 **[📘 Các Vấn Đề Tỷ Lệ Số Học Trong Mô Hình Học Sâu: Phân Tích Vai Trò Của Scaling và Normalization Trong Cơ Chế Attention](aero_llm_011_codechallenge_numerical_scaling_issues_in_dl_models_copy_2.md)** | [Xem bài viết →](aero_llm_011_codechallenge_numerical_scaling_issues_in_dl_models_copy_2.md) |
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
| [📘 Thiết Lập Tập Kiểm Thử Trong Huấn Luyện Mô Hình Ngôn Ngữ: Phân Tích Phương Pháp Train–Test Split và Đánh Giá Hiệu Suất](aero_llm_06_codechallenge_add_a_test_set.md) | [Xem bài viết →](aero_llm_06_codechallenge_add_a_test_set.md) |
| [📘 Chuyển Giao Trọng Số và Đóng Băng Tham Số Trong Huấn Luyện Mô Hình Ngôn Ngữ: Phân Tích Thực Nghiệm Với Embedding GPT-2](aero_llm_07_codechallenge_train_model_1_with_gpt2_s_embeddings.md) | [Xem bài viết →](aero_llm_07_codechallenge_train_model_1_with_gpt2_s_embeddings.md) |
| [📘 Phương Pháp Lấy Mẫu Ngẫu Nhiên và Huấn Luyện Mô Hình GPT-2 Thu Gọn: Phân Tích Thực Nghiệm Với Dữ Liệu Văn Bản Cổ Điển](aero_llm_08_codechallenge_train_model_5_with_modifications.md) | [Xem bài viết →](aero_llm_08_codechallenge_train_model_5_with_modifications.md) |
| [Thiết Kế Hàm Mất Mát Tùy Biến Trong Huấn Luyện Mô Hình Ngôn Ngữ Lớn](aero_llm_09_create_a_custom_loss_function.md) | [Xem bài viết →](aero_llm_09_create_a_custom_loss_function.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
