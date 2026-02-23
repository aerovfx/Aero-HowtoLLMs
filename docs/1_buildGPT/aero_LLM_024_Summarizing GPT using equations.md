

# Kiến Trúc Transformer và Triển Khai GPT-2 trên GPU: Phân Tích Toán Học và Hiệu Năng Tính Toán

## Tóm tắt

Các mô hình ngôn ngữ lớn (Large Language Models – LLMs) dựa trên kiến trúc Transformer đã đạt được nhiều thành tựu trong xử lý ngôn ngữ tự nhiên. Bài viết này trình bày tổng quan về cấu trúc toán học của GPT-2, cơ chế multi-head attention, quy trình huấn luyện và suy luận, cũng như đánh giá hiệu năng khi triển khai trên CPU và GPU. Thông qua phân tích lý thuyết và thực nghiệm, nghiên cứu cho thấy GPU đóng vai trò thiết yếu trong việc vận hành các mô hình ngôn ngữ hiện đại.

---

## 1. Giới thiệu

Transformer là nền tảng của hầu hết các mô hình ngôn ngữ hiện đại. Kiến trúc này cho phép mô hình hóa mối quan hệ dài hạn giữa các token thông qua cơ chế attention. GPT-2 là một trong những mô hình tiêu biểu sử dụng Transformer để sinh ngôn ngữ tự nhiên.

Việc triển khai hiệu quả các mô hình này đòi hỏi sự kết hợp giữa hiểu biết toán học, thiết kế kiến trúc và tối ưu phần cứng.

---

## 2. Biểu diễn Embedding và Dữ liệu Đầu vào

Trong GPT-2, mỗi token được ánh xạ sang một vector embedding thông qua ma trận từ vựng (E \in \mathbb{R}^{V \times D}), kết hợp với embedding vị trí (P \in \mathbb{R}^{L \times D}). Quá trình này được mô tả bằng one-hot encoding và phép nhân ma trận.

Phép biến đổi từ token sang embedding được thực hiện thông qua:

[
X = \Delta E + P
]

trong đó (X \in \mathbb{R}^{T \times D}) là ma trận biểu diễn chuỗi đầu vào.

Quá trình này được trình bày chi tiết trong tài liệu tổng hợp toán học về GPT. 

---

## 3. Cơ Chế Multi-Head Attention

### 3.1. Nguyên lý toán học

Multi-head attention chia không gian embedding thành nhiều phần (heads) song song. Với mỗi head (h), ta có:

[
Q_h = XW_Q^h, \quad K_h = XW_K^h, \quad V_h = XW_V^h
]

Sau đó, attention được tính:

[
A_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{D/H}} + M \right)V_h
]

Các đầu ra được nối lại và chiếu tuyến tính:

[
A = \text{Concat}(A_1, \dots, A_H)W_0
]

Việc chia nhỏ attention giúp mô hình học được nhiều kiểu quan hệ ngữ nghĩa khác nhau. 

---

### 3.2. Triển khai trong PyTorch

Trong thực tế, các ma trận (W_Q, W_K, W_V) thường được gộp thành một ma trận duy nhất để tăng hiệu suất. Quá trình reshape và transpose được sử dụng để tách các head trong forward pass. 

Việc sử dụng hàm attention tích hợp giúp tối ưu tính toán song song trên GPU. 

---

## 4. Khối Transformer và Mạng MLP

### 4.1. Attention Sub-layer

Mỗi khối Transformer bắt đầu bằng layer normalization, sau đó là multi-head attention và residual connection:

[
X' = X + \text{Attention}(\text{LN}(X))
]

### 4.2. Feed-Forward Network (MLP)

Sau attention, dữ liệu được đưa qua mạng MLP gồm hai lớp tuyến tính:

[
Y = X' + W_2(\text{GELU}(W_1(\text{LN}(X'))))
]

Mạng MLP giúp mô hình trích xuất đặc trưng phi tuyến trong không gian chiều cao. 

---

## 5. Unembedding và Sinh Token

Đầu ra cuối cùng được chuẩn hóa và nhân với ma trận embedding ban đầu để tạo logits:

[
L = \text{LN}(X_{out})E^T
]

Sau đó, softmax được sử dụng để sinh phân phối xác suất cho token tiếp theo. 

Chiến lược sampling (temperature, top-k, top-p) ảnh hưởng mạnh đến chất lượng văn bản sinh ra. 

---

## 6. Kiến Trúc GPT-2 và Số Lượng Tham Số

GPT-2 Small có:

* 12 Transformer blocks
* 12 attention heads mỗi block
* Embedding dimension: 768
* Context length: 1024

Tổng số tham số huấn luyện khoảng 124 triệu, sau khi chia sẻ embedding và unembedding. 

Phân tích cấu trúc và tham số có thể được thực hiện thông qua torchinfo. 

---

## 7. Hiệu Năng Tính Toán: CPU và GPU

### 7.1. So sánh thời gian khởi tạo

Việc khởi tạo mô hình trên CPU và GPU có chênh lệch nhỏ (~300ms), không đáng kể trong thực tế. 

---

### 7.2. Forward Pass và Huấn luyện

Trong các thử nghiệm, forward pass trên GPU nhanh hơn CPU nhiều bậc độ lớn:

* CPU: ~20 giây
* GPU: ~30 ms



Điều này cho thấy GPU là bắt buộc đối với các LLM. 

---

### 7.3. Chi phí Truyền Dữ liệu

Việc chuyển dữ liệu giữa CPU và GPU gây độ trễ đáng kể. Tối ưu hiệu năng đòi hỏi duy trì dữ liệu trên cùng một thiết bị. 

---

## 8. Quản Lý Thiết Bị và Lỗi Thường Gặp

Một lỗi phổ biến là tensor nằm trên các thiết bị khác nhau (CPU/GPU), dẫn đến runtime error. Việc truyền tham số `device` đồng bộ là bắt buộc. 

Ví dụ, vector vị trí tạo trên CPU sẽ gây lỗi nếu mô hình chạy trên GPU. 

---

## 9. Thảo Luận

### 9.1. Vai trò của Multi-Head Attention

Multi-head attention giúp mô hình học được nhiều kiểu phụ thuộc ngữ cảnh khác nhau. Tuy nhiên, lý do thành công chủ yếu vẫn mang tính thực nghiệm. 

### 9.2. Tính Mở Rộng của Mô Hình

Cấu trúc GPT-2 có thể mở rộng bằng cách tăng:

* Số layer
* Số head
* Kích thước embedding
* Dữ liệu huấn luyện

Các mô hình thương mại hiện nay chủ yếu mở rộng theo hướng này. 

---

## 10. Kết luận

Bài viết đã trình bày một cách hệ thống kiến trúc GPT-2 từ góc độ toán học, lập trình và phần cứng. Các kết quả cho thấy:

1. Transformer duy trì embedding thông qua residual learning.
2. Multi-head attention giúp học đặc trưng đa chiều.
3. GPU là thành phần không thể thiếu cho LLM.
4. Việc quản lý thiết bị ảnh hưởng lớn đến độ ổn định và hiệu năng.

Hiểu rõ các yếu tố này giúp tối ưu việc phát triển và triển khai mô hình ngôn ngữ lớn trong thực tế.
