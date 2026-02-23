
Như vậy, từ “lowest” có thể biểu diễn thành:

\[
\text{lowest} = \text{low} + \text{est}
\]

---

## 5. Biểu diễn Embedding và Kích thước Tính toán

Giả sử:

- Kích thước từ vựng: \( V \)
- Kích thước embedding: \( d \)

Ma trận embedding:

\[
E \in \mathbb{R}^{V \times d}
\]

Số tham số của embedding:

\[
\text{Params} = V \times d
\]

Nếu dùng word-level tokenization:
\[
V \approx 500,000
\]

Nếu dùng BPE:
\[
V \approx 30,000 - 50,000
\]

Giảm số tham số đáng kể:

\[
\Delta = (V_{word} - V_{BPE}) \times d
\]

Điều này giúp:
- Giảm bộ nhớ
- Tăng tốc huấn luyện
- Cải thiện khả năng tổng quát hóa

---

## 6. BPE trong Mô hình Transformer

Trong kiến trúc Transformer, chuỗi token được ánh xạ sang embedding:

\[
x_i = E(t_i)
\]

Sau đó được đưa vào cơ chế Attention:

\[
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
\]

Việc sử dụng BPE giúp:

- Giảm chiều dài chuỗi so với character-level.
- Giữ thông tin hình thái tốt hơn word-level.
- Tối ưu hóa hiệu suất attention.

---

## 7. So sánh với các phương pháp khác

| Phương pháp | Đơn vị | Ưu điểm | Nhược điểm |
|------------|--------|----------|------------|
| Word-level | Từ | Dễ hiểu | OOV cao |
| Character-level | Ký tự | Không OOV | Chuỗi dài |
| BPE | Subword | Cân bằng tốt | Phụ thuộc số vòng gộp |

---

## 8. Ứng dụng trong Mô hình Ngôn ngữ Lớn

Các mô hình như GPT sử dụng biến thể của BPE để xây dựng tokenizer. Với dữ liệu huấn luyện hàng trăm tỷ token, BPE cho phép:

- Nén biểu diễn từ vựng.
- Tăng khả năng học cấu trúc ngôn ngữ.
- Xử lý tốt từ hiếm và từ mới.

Giả sử tổng số token huấn luyện:

\[
T = 10^{11}
\]

Thời gian huấn luyện phụ thuộc vào:

\[
\mathcal{O}(T \cdot L \cdot d^2)
\]

Trong đó:
- \( L \): chiều dài chuỗi
- \( d \): kích thước mô hình

BPE giúp giảm \( L \) so với character-level → giảm chi phí tính toán.

---

## 9. Hạn chế của BPE

- Không xét ngữ nghĩa khi gộp token.
- Có thể tạo token không trực quan.
- Phụ thuộc mạnh vào dữ liệu huấn luyện ban đầu.

---

## 10. Kết luận

Byte Pair Encoding là một phương pháp phân tách từ hiệu quả, đóng vai trò nền tảng trong các mô hình ngôn ngữ hiện đại. Nhờ khả năng cân bằng giữa kích thước từ vựng và chiều dài chuỗi, BPE giúp tối ưu hóa cả bộ nhớ và hiệu suất tính toán.

Trong bối cảnh các mô hình ngày càng lớn (hàng trăm tỷ tham số), việc tối ưu tokenizer như BPE không chỉ là bước tiền xử lý, mà còn ảnh hưởng trực tiếp đến hiệu quả huấn luyện và suy luận.

---

## Tài liệu tham khảo

1. Gage, P. (1994). *A New Algorithm for Data Compression.*
2. Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.*
3. Vaswani, A. et al. (2017). *Attention Is All You Need.*
4. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners.*

---
