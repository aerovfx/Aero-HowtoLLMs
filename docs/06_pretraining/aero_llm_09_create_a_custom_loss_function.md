
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [06 pretraining](index.md)

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
Dưới đây là **bài viết khoa học** được biên soạn dựa trên tài liệu đính kèm, có bổ sung trích dẫn và trình bày theo định dạng **Markdown**.

---

# **Thiết Kế Hàm Mất Mát Tùy Biến Trong Huấn Luyện Mô Hình Ngôn Ngữ Lớn**

## **Abstract**

Hàm mất mát (loss function) đóng vai trò trung tâm trong quá trình huấn luyện các mô hình học sâu, đặc biệt là các mô hình ngôn ngữ lớn (Large Language Models – LLMs). Bài viết này trình bày vai trò của hàm mất mát trong huấn luyện mô hình, phân tích hàm Cross-Entropy, Negative Log-Likelihood, và đề xuất phương pháp xây dựng hàm mất mát tùy biến trong PyTorch. Ngoài ra, bài viết so sánh đặc tính của hàm L1 và L2, đồng thời đánh giá tác động của chúng đến quá trình hội tụ mô hình.

---

## **1. Introduction**

Trong học sâu, quá trình huấn luyện mô hình được xem như một bài toán tối ưu hóa, trong đó mục tiêu là điều chỉnh tham số để giảm thiểu giá trị hàm mất mát. Hàm mất mát cung cấp một thước đo định lượng về mức độ sai lệch giữa đầu ra của mô hình và nhãn mục tiêu.

Đối với các mô hình ngôn ngữ, nhiệm vụ chính là dự đoán token tiếp theo trong chuỗi văn bản. Do token mang tính rời rạc và phân loại, việc lựa chọn hàm mất mát phù hợp là điều kiện tiên quyết để mô hình học hiệu quả .

---

## **2. Loss Functions in Language Model Training**

### **2.1. Categorical Cross-Entropy và Negative Log-Likelihood**

Trong huấn luyện LLMs, hàm Cross-Entropy (CE) và Negative Log-Likelihood (NLL) thường được sử dụng:

$\mathcal${L} = - $\sum$_{i=1}^{N} $y_i$ $\log$(\hat{y}_i)

Trong đó:

* $y_i$: nhãn thật (one-hot encoding),
* $\hat{y}_i$: xác suất dự đoán,
* $N$: số lớp (token trong từ điển).

Vì các token là các lớp rời rạc và loại trừ lẫn nhau, nên trong thực tế chỉ có một giá trị ($y_i$ = 1), các giá trị còn lại bằng 0. Do đó, hàm mất mát có thể rút gọn thành:

$\mathcal${L} = -$\log$(\hat{y}_{target})

Theo tài liệu, PyTorch triển khai Cross-Entropy Loss và NLL Loss theo cách gần tương đương, trong đó NLL yêu cầu đầu vào ở dạng log-softmax .

---

### **2.2. Vai Trò Của Loss Function**

Hàm mất mát quyết định:

* Hướng cập nhật gradient,
* Tốc độ hội tụ,
* Độ ổn định huấn luyện,
* Khả năng tổng quát hóa.

Một hàm mất mát không phù hợp có thể khiến mô hình không hội tụ hoặc học sai nhiệm vụ.

---

## **3. Custom Loss Functions in PyTorch**

### **3.1. Cấu Trúc Cơ Bản**

Trong PyTorch, hàm mất mát tùy biến được xây dựng bằng cách kế thừa `nn.Module` và định nghĩa phương thức `forward`:

```python
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, y_hat, y):

$$
loss = ...
$$

        return loss

Cấu trúc này tương tự như cách xây dựng một mô hình neural network, giúp dễ dàng tích hợp vào pipeline huấn luyện .

---

### **3.2. Ví Dụ: L1 và L2 Loss**

#### **L1 Loss (Mean Absolute Error)**

$\mathcal${L}_{L1} = |y - \hat{y}|

#### **L2 Loss (Mean Squared Error)**

$\mathcal${L}_{L2} = (y - \hat{y})^2

Cài đặt trong PyTorch:

```python
class MyLossL1(nn.Module):
    def forward(self, y_hat, y):
        return torch.abs(y_hat - y)

class MyLossL2(nn.Module):
    def forward(self, y_hat, y):
        return (y_hat - y) ** 2

$$
Theo tài liệu, khi giá trị mục tiêu là 5 và dự đoán là 2, L1 = 3 và L2 = 9 .
$$

---

## **4. Experimental Analysis**

### **4.1. Thiết Lập Thí Nghiệm**

* Mô hình: một tham số đơn.
* Optimizer: Stochastic Gradient Descent (SGD).
* Learning rate: 0.05.
* Loss: L1 và L2.

Mỗi tham số được huấn luyện độc lập với hàm mất mát tương ứng.

---

### **4.2. Kết Quả**

Kết quả thực nghiệm cho thấy:

* L1 Loss dao động quanh giá trị mục tiêu.
* L2 Loss hội tụ mượt và nhanh hơn.
* L2 giảm gradient dần khi tiến gần nghiệm tối ưu.

Hiện tượng dao động của L1 chủ yếu do đạo hàm không liên tục tại điểm 0 và learning rate lớn .

---

### **4.3. Phân Tích So Sánh**

| Đặc tính         | L1 Loss    | L2 Loss |
| ---------------- | ---------- | ------- |
| Độ mượt          | Thấp       | Cao     |
| Nhạy với outlier | Thấp       | Cao     |
| Tốc độ hội tụ    | Chậm       | Nhanh   |
| Ổn định          | Trung bình | Cao     |

L2 thường phù hợp cho các bài toán cần hội tụ mượt, trong khi L1 thích hợp khi dữ liệu có nhiều nhiễu.

---

## **5. Applications of Custom Loss Functions**

Hàm mất mát tùy biến cho phép:

* Điều chỉnh hành vi học của mô hình,
* Giảm thiên lệch (bias),
* Tối ưu mục tiêu đặc thù,
* Áp dụng regularization riêng.

Ví dụ:

* Tối ưu tương quan,
* Phạt mô hình thiên vị,
* Hạn chế overfitting,
* Cân bằng dữ liệu không đồng đều.

Theo tài liệu, việc thiết kế loss có thể được sử dụng để “hack” quá trình học nhằm tạo ra các đặc tính mong muốn .

---

## **6. Discussion**

Mặc dù hàm mất mát tùy biến mang lại tính linh hoạt cao, nhưng việc thiết kế không phù hợp có thể gây:

* Gradient exploding/vanishing,
* Mất ổn định số học,
* Khó hội tụ,
* Overfitting.

Do đó, các hàm loss nên:

* Đơn giản,
* Có đạo hàm liên tục,
* Dễ tối ưu,
* Có ý nghĩa vật lý/thống kê.

Trong thực tế, các hàm loss tiêu chuẩn vẫn là lựa chọn ưu tiên, và loss tùy biến chỉ nên dùng khi có nhu cầu đặc biệt.

---

## **7. Conclusion**

Bài viết đã trình bày:

* Vai trò của hàm mất mát trong huấn luyện LLMs,
* Cơ chế hoạt động của Cross-Entropy và NLL,
* Phương pháp xây dựng loss tùy biến trong PyTorch,
* So sánh L1 và L2 Loss,
* Ảnh hưởng của loss đến hội tụ mô hình.

Kết quả cho thấy, việc lựa chọn và thiết kế hàm mất mát phù hợp là yếu tố quyết định hiệu suất và độ ổn định của mô hình học sâu.

Trong tương lai, nghiên cứu có thể tập trung vào việc kết hợp nhiều hàm loss (hybrid loss) và tự động tối ưu cấu trúc loss bằng meta-learning.

---

## **References**

1. Tài liệu hướng dẫn về xây dựng custom loss function trong PyTorch và huấn luyện LLMs. 
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. Paszke et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.

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
| [📘 Thiết Lập Tập Kiểm Thử Trong Huấn Luyện Mô Hình Ngôn Ngữ: Phân Tích Phương Pháp Train–Test Split và Đánh Giá Hiệu Suất](aero_llm_06_codechallenge_add_a_test_set.md) | [Xem bài viết →](aero_llm_06_codechallenge_add_a_test_set.md) |
| [📘 Chuyển Giao Trọng Số và Đóng Băng Tham Số Trong Huấn Luyện Mô Hình Ngôn Ngữ: Phân Tích Thực Nghiệm Với Embedding GPT-2](aero_llm_07_codechallenge_train_model_1_with_gpt2_s_embeddings.md) | [Xem bài viết →](aero_llm_07_codechallenge_train_model_1_with_gpt2_s_embeddings.md) |
| [📘 Phương Pháp Lấy Mẫu Ngẫu Nhiên và Huấn Luyện Mô Hình GPT-2 Thu Gọn: Phân Tích Thực Nghiệm Với Dữ Liệu Văn Bản Cổ Điển](aero_llm_08_codechallenge_train_model_5_with_modifications.md) | [Xem bài viết →](aero_llm_08_codechallenge_train_model_5_with_modifications.md) |
| 📌 **[Thiết Kế Hàm Mất Mát Tùy Biến Trong Huấn Luyện Mô Hình Ngôn Ngữ Lớn](aero_llm_09_create_a_custom_loss_function.md)** | [Xem bài viết →](aero_llm_09_create_a_custom_loss_function.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
