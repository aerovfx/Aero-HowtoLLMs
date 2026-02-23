Dưới đây là bài viết khoa học được tổng hợp từ tài liệu bạn cung cấp, có bổ sung trích dẫn và trình bày dưới dạng **Markdown**.

---

# Multi-Head Attention: Cơ Sở Lý Thuyết và Triển Khai Thực Tiễn

## Tóm tắt (Abstract)

Multi-Head Attention (MHA) là một thành phần cốt lõi trong kiến trúc Transformer, cho phép mô hình học đồng thời nhiều dạng quan hệ ngữ cảnh khác nhau trong chuỗi dữ liệu. Bài viết này trình bày cơ sở toán học, cơ chế hoạt động, cách triển khai và ý nghĩa thực nghiệm của multi-head attention dựa trên tài liệu học tập đi kèm. Qua đó, bài viết giúp làm rõ vai trò của việc phân tách không gian biểu diễn thành nhiều "đầu chú ý" (attention heads) nhằm nâng cao khả năng biểu diễn của mô hình.

---

## 1. Giới thiệu

Cơ chế Attention đã trở thành nền tảng của các mô hình xử lý ngôn ngữ tự nhiên hiện đại. Trong đó, multi-head attention mở rộng mô hình single-head attention bằng cách cho phép xử lý song song nhiều không gian đặc trưng.

Theo tài liệu tham khảo, multi-head attention được xây dựng bằng cách chia các ma trận attention thành nhiều ma trận con, giúp xử lý song song các vector token

Mục tiêu của bài viết là:

* Trình bày cách xây dựng multi-head attention.
* Phân tích cơ sở toán học.
* Giải thích lý do sử dụng nhiều head.
* Mô tả quy trình triển khai trong thực tế.

---

## 2. Cơ sở toán học của Attention

### 2.1. Ma trận Query, Key và Value

Trong attention, ba ma trận chính được xây dựng:

* Query (Q)
* Key (K)
* Value (V)

Chúng được tính như sau:

[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
]

Trong đó:

* (X): Ma trận embedding.
* (W_Q, W_K, W_V): Ma trận trọng số huấn luyện.

Các chiều embedding được trộn lẫn thông qua phép nhân ma trận, không được giữ nguyên theo từng chiều ban đầu

---

### 2.2. Single-Head Attention

Với một head, attention được tính theo công thức:

[
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

Trong đó (d_k) là số chiều của vector key.

---

## 3. Cơ chế Multi-Head Attention

### 3.1. Phân tách thành nhiều Head

Multi-head attention chia các ma trận Q, K, V thành (H) phần không chồng lấn:

[
Q = [Q_1, Q_2, ..., Q_H]
]

Mỗi head có kích thước:

[
d_h = \frac{D}{H}
]

với (D) là số chiều embedding.

Việc chia này yêu cầu (D) chia hết cho (H)

---

### 3.2. Attention trên từng Head

Với mỗi head (i):

[
\text{head}_i =
\text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_h}}\right)V_i
]

Hệ số chuẩn hóa được điều chỉnh theo số chiều mới (D/H)

---

### 3.3. Kết hợp các Head

Sau khi tính attention cho từng head, kết quả được ghép nối:

[
A = \text{Concat}(\text{head}_1,...,\text{head}_H)W_0
]

Trong đó (W_0) là ma trận tuyến tính dùng để trộn thông tin giữa các head.

Không sử dụng hàm phi tuyến tại bước này nhằm tránh làm mất thông tin học được từ từng head

---

## 4. Phân tích số lượng tham số

Một điểm quan trọng là multi-head attention **không làm tăng số lượng tham số huấn luyện** so với single-head attention.

Mặc dù số phép tính tăng lên, tổng số tham số vẫn giữ nguyên vì các ma trận trọng số không bị chia nhỏ từ đầu

---

## 5. Lý do sử dụng Multi-Head Attention

### 5.1. Học nhiều đặc trưng song song

Mỗi head có thể tập trung vào một dạng quan hệ khác nhau:

* Quan hệ cục bộ.
* Quan hệ dài hạn.
* Tương đồng ngữ nghĩa.
* Cấu trúc cú pháp.

Nhờ đó, mô hình có khả năng biểu diễn phong phú hơn

---

### 5.2. Góc nhìn thực nghiệm

Hiện nay, chưa có lý thuyết toán học hoàn chỉnh giải thích vì sao multi-head attention hiệu quả.

Theo tài liệu, lý do chính là:

> Các nhà phát triển thử nghiệm và nhận thấy mô hình hoạt động tốt hơn.
> Deep learning mang tính thực nghiệm cao.

---

## 6. Triển khai Multi-Head Attention

### 6.1. Cấu trúc lớp

Một lớp multi-head attention thường bao gồm:

* Số head: (H)
* Kích thước mỗi head: (d_h)
* Các ma trận: (W_Q, W_K, W_V, W_0)

Các ma trận này ban đầu có kích thước (D \times D) và chỉ được chia trong quá trình forward pass

---

### 6.2. Quy trình Forward Pass

Quy trình cơ bản:

1. Tính Q, K, V từ embedding.
2. Reshape thành dạng:
   [
   (B, T, H, d_h)
   ]
3. Hoán vị chiều để phù hợp với hàm attention.
4. Tính attention song song.
5. Ghép các head.
6. Nhân với (W_0).

Việc hoán vị chiều giúp tối ưu cho GPU, dù gây thêm chi phí xử lý

---

### 6.3. Theo dõi kích thước Tensor

Một số triển khai cho phép bật chế độ theo dõi kích thước tensor trong quá trình tính toán nhằm hỗ trợ debug và học tập

---

## 7. Ví dụ kích thước

Ví dụ với:

* Embedding: 128
* Số head: 4

Ta có:

[
128 \rightarrow 4 \times 32 \rightarrow 128
]

Trong quá trình tính toán, embedding được chia thành 4 head, mỗi head 32 chiều, sau đó ghép lại

---

## 8. Thảo luận

Multi-head attention mang lại các lợi ích chính:

* Tăng khả năng biểu diễn.
* Học đa dạng quan hệ.
* Cải thiện hiệu suất mô hình.
* Không làm tăng số tham số.

Tuy nhiên, chi phí tính toán và bộ nhớ cao hơn vẫn là một thách thức trong các mô hình quy mô lớn.

Ngoài ra, việc hiểu sâu cơ chế này hỗ trợ:

* Thiết kế kiến trúc mới.
* Tối ưu mô hình.
* Phân tích hành vi của LLM.

---

## 9. Kết luận

Bài viết đã trình bày:

* Cơ sở toán học của multi-head attention.
* Cách phân tách và kết hợp các head.
* Cơ chế triển khai trong thực tế.
* Lý do sử dụng nhiều head.

Multi-head attention là nền tảng quan trọng của các mô hình Transformer hiện đại, đóng vai trò quyết định trong sự thành công của các hệ thống ngôn ngữ lớn ngày nay.

---
