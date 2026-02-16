
# Tinh Chỉnh Mô Hình GPT-Neo Trên Tác Phẩm Văn Học: Phân Tích Thực Nghiệm Alice và Edgar Allan Poe

## Tóm tắt

Bài viết này trình bày phương pháp tinh chỉnh mô hình ngôn ngữ **GPT-Neo 125M** trên hai miền dữ liệu văn học mang phong cách đối lập: *Alice's Adventures in Wonderland* của Lewis Carroll và các tác phẩm kinh dị, trinh thám của Edgar Allan Poe. Dựa trên tài liệu thực nghiệm , nghiên cứu phân tích quá trình chuẩn bị dữ liệu, cấu hình tokenizer, và các tham số huấn luyện nhằm tối ưu hóa khả năng mô phỏng phong cách tác giả. Kết quả cho thấy GPT-Neo có khả năng thích nghi nhanh chóng với từ vựng và cấu trúc câu đặc thù của từng miền dữ liệu chỉ sau một số lượng epoch hạn chế.

---

## 1. Giới thiệu

Việc tinh chỉnh (fine-tuning) cho phép các mô hình ngôn ngữ lớn (LLMs) chuyển đổi từ các kiến thức tổng quát sang các khả năng chuyên biệt. Trong nghiên cứu này, mục tiêu là tinh chỉnh mô hình GPT-Neo để sinh văn bản mang phong cách của các tác giả cổ điển.

Theo tài liệu , việc lựa chọn Alice và Edgar Allan Poe mang lại sự so sánh thú vị:
* **Alice:** Ngôn ngữ tưởng tượng, trong sáng, cấu trúc câu logic nhưng phi lý.
* **Edgar Allan Poe:** Ngôn ngữ tối tăm, u uất, nhiều từ vựng cổ và hình ảnh mang tính biểu tượng.

Mục tiêu nghiên cứu:
* Thực hiện fine-tuning trên hai tập dữ liệu độc lập.
* Phân tích sự thay đổi trong xác suất sinh token.
* Đánh giá tính ổn định của mô hình trong quá trình học.

---

## 2. Cơ sở lý thuyết

### 2.1. Kiến trúc GPT-Neo

GPT-Neo là kiến trúc Transformer dựa trên thiết kế của GPT-3 nhưng mã nguồn mở. Mô hình sử dụng cơ chế **Local Attention** kết hợp với **Sparse Attention** để xử lý các chuỗi dài hiệu quả hơn.

Hàm mục tiêu (Causal Language Modeling):
[
\mathcal{L} = -\sum_{i=1}^{n} \log P(x_i \mid x_{<i}; \theta)
]

---

### 2.2. Tokenizer và Padding

Mô hình sử dụng tokenizer của GPT-2 dựa trên thuật toán Byte-Pair Encoding (BPE). Một vấn đề quan trọng trong fine-tuning là xử lý độ dài chuỗi cố định:
[
X \in \mathbb{R}^{B \times L}
]
Trong đó các chuỗi ngắn hơn $L$ sẽ được thêm token `<|endoftext|>` làm padding.

---

## 3. Phương pháp nghiên cứu

### 3.1. Chuẩn bị dữ liệu

Dữ liệu được trích xuất từ các file văn bản thô:
1. `alice29.txt` (Dữ liệu Alice).
2. `poe.txt` (Dữ liệu Edgar Allan Poe).

Quy trình xử lý:
* Loại bỏ các ký tự rác.
* Chia văn bản thành các đoạn có độ dài 128 token.
* Tải dữ liệu vào `DataLoader` với batch size 64.

---

### 3.2. Cấu hình huấn luyện

Theo , các tham số chính bao gồm:
* **Optimizer:** AdamW.
* **Learning Rate:** $10^{-5}$ (nhằm tránh phá vỡ các tri thức tiền huấn luyện).
* **Số mẫu:** 200 đoạn văn mỗi miền.

---

## 4. Kết quả thực nghiệm

### 4.1. Sự sụt giảm của Loss

Quá trình huấn luyện cho thấy hàm mất mát giảm nhanh trong 50 batch đầu tiên:
[
\mathcal{L}(t) = a \cdot e^{-bt} + c
]
Điều này chứng tỏ mô hình bắt đầu "thuộc" các đặc trưng ngôn ngữ của văn bản mục tiêu.

---

### 4.2. Quan sát kết quả sinh

Văn bản sinh ra từ mô hình sau fine-tuning:
* **Alice models:** Xuất hiện nhiều từ như "Queen", "Hatter", "Rabbbit" và phong cách hội thoại dí dỏm.
* **Poe models:** Xuất hiện các từ u ám như "ghastly", "dreary", "sorrow".

---

## 5. Thảo luận

### 5.1. Ảnh hưởng của quy mô dữ liệu

Với chỉ 200 mẫu dữ liệu, mô hình có xu hướng học thuộc lòng (overfitting) nếu huấn luyện quá nhiều epoch. Tuy nhiên, đối với bài toán mô phỏng phong cách (style mimicry), mức độ quá khớp nhẹ thực sự giúp văn bản sinh ra mang đậm nét đặc trưng của tác giả hơn.

---

### 5.2. Chênh lệch Tokens BERT và GPT

Nghiên cứu cũng lưu ý rằng khi sử dụng bộ phân loại BERT để đánh giá mô hình GPT, sự khác biệt trong cách token hóa có thể gây ra sai lệch nhỏ trong việc định lượng nội dung.

---

## 6. Kết luận

Thử nghiệm tinh chỉnh GPT-Neo trên Alice và Poe đã minh chứng cho sức mạnh của việc tinh chỉnh từng phần trên các mô hình ngôn ngữ 125M tham số. Chỉ với một tập dữ liệu nhỏ và tài nguyên tính toán vừa phải, chúng ta có thể tạo ra các biến thể mô hình mang phong cách sáng tạo độc đáo.

---

## Tài liệu tham khảo

1. Tài liệu thực hành: Alice in Wonderland and Edgar Allan Poe (with GPT-neo).
2. Black et al. (2021). *GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow*.
3. Carroll, L. (1865). *Alice's Adventures in Wonderland*.
4. Poe, E. A. (1845). *The Raven and Other Poems*.

---
