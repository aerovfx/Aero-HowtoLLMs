
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
Dưới đây là **bài viết khoa học** được xây dựng dựa trên tài liệu **“CodeChallenge: Fine Dropout in Model 5 (Part 2)”**, có bổ sung phân tích học thuật và nguồn trích dẫn, trình bày dưới dạng **Markdown**.

---

# **Chiến Lược Huấn Luyện Dựa Trên Final-Token Loss Trong Mô Hình Transformer: Phân Tích Trường Hợp Model 5 Với Dropout**

---

## Abstract

Trong huấn luyện mô hình ngôn ngữ dựa trên Transformer, việc lựa chọn chiến lược tính hàm mất mát (loss function) ảnh hưởng trực tiếp đến tốc độ hội tụ, độ ổn định và chất lượng sinh văn bản. Bài viết này phân tích phương pháp huấn luyện Model 5 với dropout, trong đó loss chỉ được tính trên token cuối cùng của mỗi chuỗi đầu vào. Dựa trên tài liệu CodeChallenge “Fine Dropout in Model 5 (Part 2)”, nghiên cứu làm rõ sự khác biệt giữa huấn luyện toàn bộ token và huấn luyện final-token, vai trò của log-softmax, cơ chế bật/tắt dropout trong PyTorch, cũng như tác động của dữ liệu FineWeb đến độ biến thiên của loss. Kết quả cho thấy phương pháp final-token training giúp mô hình tập trung vào nhiệm vụ sinh token kế tiếp nhưng làm tăng độ nhiễu và độ khó trong tối ưu hóa. 

---

## 1. Introduction

Huấn luyện mô hình ngôn ngữ lớn (LLMs) thường dựa trên bài toán dự đoán token tiếp theo (next-token prediction). Trong hầu hết các thiết lập tiêu chuẩn, hàm loss được tính trên toàn bộ chuỗi đầu ra. Tuy nhiên, trong một số trường hợp, chỉ token cuối cùng mới trực tiếp tương ứng với hành vi sinh văn bản trong quá trình suy luận.

Trong CodeChallenge “Fine Dropout in Model 5 (Part 2)”, tác giả yêu cầu điều chỉnh pipeline huấn luyện để:

* Chỉ tính loss trên token cuối,
* Áp dụng lại log-softmax,
* Quản lý chế độ train/eval khi dùng dropout,
* So sánh train/test loss,
* Phân tích nguyên nhân khiến loss cao và nhiễu hơn.

Tài liệu cung cấp một góc nhìn thực tiễn về đánh đổi giữa hiệu quả huấn luyện và chất lượng sinh văn bản. 

---

## 2. Background

### 2.1. Next-Token Prediction in Language Models

Trong huấn luyện LLMs, mục tiêu tiêu chuẩn là:

$$

$$

$\mathcal${L} = -$\sum$_{t=1}^{T} $\log$ P($x_t$ \mid x_{\lt t})

$$

$$

với $T$ là độ dài chuỗi.

Cách tiếp cận này cho phép mô hình học từ mọi vị trí trong chuỗi.

---

### 2.2. Dropout và Regularization

Dropout giúp giảm overfitting bằng cách ngẫu nhiên loại bỏ neuron trong quá trình huấn luyện. Trong Transformer, dropout thường được áp dụng tại:

* Attention,
* MLP,
* Residual connections.

Việc kết hợp dropout với chiến lược loss ảnh hưởng đáng kể đến động học huấn luyện.

---

## 3. Methodology

### 3.1. Final-Token Loss Strategy

Trong bài tập, hàm loss chỉ được tính tại token cuối:

$$

$$

$\mathcal${L} = - $\log$ P($x_T$ \mid x_{\lt T})

$$

$$

Thay vì flatten toàn bộ chuỗi, tác giả chỉ sử dụng:

* Logits của token cuối,
* Target của token cuối.

Cách tiếp cận này được mô tả rõ trong tài liệu. 

---

### 3.2. Log-Softmax Integration

Do forward pass chỉ trả về logits, cần áp dụng log-softmax trước khi đưa vào loss:

$$

$$

$\ell(_i = $z_i$ - )$\log$$ $\sum$_j e^{$z_j$}

$$

$$

Trong PyTorch:

```python

$$

$$

log_probs = F.log_softmax(logits, dim=-1)

$$

$$

$$
loss = NLLLoss(log_probs, targets)
$$

$$
Việc thiếu bước này dẫn đến lỗi huấn luyện nghiêm trọng. --- ### 3.3. Device Consistency Tài liệu nhấn mạnh lỗi phổ biến: > Expected all tensors to be on the same device Lỗi xảy ra khi dữ liệu và mô hình nằm trên các thiết bị khác nhau CPU/GPU. Việc đồng bộ thiết bị là điều kiện bắt buộc trong pipeline huấn luyện. --- ### 3.4. Training and Evaluation Mode Switching Để đảm bảo dropout chỉ hoạt động khi huấn luyện, tác giả sử dụng: ```python model.eval() ... model.train() Tuy nhiên, một số hàm như `scaled_dot_product_attention` không tự động tắt dropout. Do đó, trạng thái `self.training` được sử dụng để kiểm soát thủ công. --- ## 4. Experimental Design ### 4.1. Training Procedure Quy trình huấn luyện gồm: 1. Sampling batch, 2. Forward pass, 3. Log-softmax, 4. Final-token loss, 5. Backpropagation, 6. Optimization, 7. Periodic evaluation. Đánh giá được thực hiện mỗi 80 iteration. --- ### 4.2. Dataset Mô hình được huấn luyện trên FineWeb dataset, có đặc điểm: * Đa dạng chủ đề, * Phong cách không đồng nhất, * Độ biến thiên cao. Khác với huấn luyện trên một cuốn sách đơn lẻ, FineWeb tạo ra môi trường học phức tạp hơn. --- ### 4.3. Visualization Train loss và test loss được vẽ theo epoch để quan sát: * Độ ổn định, * Xu hướng hội tụ, * Mức độ nhiễu. Kết quả cho thấy loss dao động mạnh. --- ## 5. Results ### 5.1. Loss Magnitude So với huấn luyện toàn bộ token, final-token training cho thấy: * Loss cao hơn, * Dao động lớn hơn, * Hội tụ chậm hơn. Trong các bài tập trước, loss thường giảm về 3–4, trong khi ở đây duy trì ở mức cao hơn. --- ### 5.2. Loss Variability Loss biến động mạnh giữa các epoch, phản ánh: * Tín hiệu huấn luyện ít hơn, * Độ nhiễu gradient cao, * Khó tối ưu hóa. Hiện tượng này được ghi nhận rõ trong tài liệu. --- ### 5.3. Text Quality Văn bản sinh ra có đặc điểm: * Thiếu khoảng trắng, * Cấu trúc kém mạch lạc, * Token bị dính liền. So với mô hình huấn luyện trên một cuốn sách, chất lượng thấp hơn đáng kể. --- ## 6. Discussion ### 6.1. Nguyên Nhân Loss Cao Tài liệu xác định hai nguyên nhân chính: #### (1) Reduced Training Signal Trước đây, mô hình học từ 256 token/chuỗi. Hiện tại, chỉ học từ 1 token:
$$

$$
\text{Signal reduction factor} \approx 256
$$

