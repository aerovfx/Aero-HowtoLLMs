# Thuật toán Byte Pair Encoding (BPE) và Bài toán Tối ưu Kích thước Từ vựng trong Mô hình Ngôn ngữ

## Tóm tắt

Trong các mô hình ngôn ngữ hiện đại, đặc biệt là các hệ thống dựa trên kiến trúc Transformer, việc xây dựng tokenizer đóng vai trò nền tảng quyết định hiệu suất và chi phí tính toán. Tài liệu đính kèm trình bày một bài toán thực hành: **triển khai thuật toán Byte Pair Encoding (BPE) để đạt kích thước từ vựng mong muốn**. Bài viết này phân tích cơ sở lý thuyết của BPE, mô hình hóa toán học quá trình gộp cặp, bài toán tối ưu kích thước từ vựng, độ phức tạp tính toán và mối liên hệ với huấn luyện mô hình ngôn ngữ lớn (LLM).

---

## 1. Giới thiệu

Các mô hình học sâu xử lý văn bản thông qua ánh xạ:

\[
\text{text} \rightarrow \text{tokens} \rightarrow \text{embedding vectors}
\]

Một tokenizer hiệu quả cần:

- Giảm kích thước từ vựng \(V\)
- Hạn chế token ngoài tập huấn luyện (OOV)
- Giữ độ dài chuỗi \(L\) ở mức hợp lý

Byte Pair Encoding (BPE) được đề xuất ban đầu cho nén dữ liệu (Gage, 1994) và được áp dụng cho NLP bởi Sennrich et al. (2016). Hiện nay, nhiều mô hình ngôn ngữ lớn sử dụng biến thể của BPE.

---

## 2. Mô hình hóa bài toán BPE

### 2.1 Biểu diễn dữ liệu

Giả sử tập dữ liệu huấn luyện:

\[
\mathcal{D} = \{w_1, w_2, \dots, w_N\}
\]

Mỗi từ được biểu diễn thành chuỗi ký tự:

\[
w_i = (c_1, c_2, \dots, c_m)
\]

Tập token ban đầu:

\[
V_0 = \{ \text{tất cả ký tự xuất hiện} \}
\]

---

### 2.2 Hàm đếm tần suất cặp token

Tại bước \(k\), tập token là \(V_k\).

Tập các cặp token liền kề:

\[
P_k = \{(t_i, t_{i+1})\}
\]

Hàm tần suất:

\[
f_k(p) = \sum_{w \in \mathcal{D}} \text{count}(p, w)
\]

Chọn cặp tối ưu:

\[
p_k^* = \arg\max_{p \in P_k} f_k(p)
\]

Sau đó cập nhật:

\[
V_{k+1} = V_k \cup \{ t_{new} \}
\]

Quá trình dừng khi:

\[
|V_k| = V_{target}
\]

---

## 3. Bài toán đạt kích thước từ vựng mong muốn

Giả sử:

- Từ vựng ban đầu: \( |V_0| = C \)
- Số vòng gộp: \( M \)

Khi đó:

\[
|V_M| = C + M
\]

Nếu muốn:

\[
|V_M| = V_{target}
\]

Ta cần:

\[
M = V_{target} - C
\]

Như vậy, bài toán trở thành:

> Thực hiện chính xác \( M \) phép gộp có tần suất cao nhất.

---

## 4. Phân tích độ phức tạp tính toán

### 4.1 Mỗi vòng lặp

- Đếm tần suất tất cả cặp:  
  \[
  \mathcal{O}(T)
  \]
  với \(T\) là tổng số token trong tập dữ liệu.

- Chọn cặp lớn nhất:  
  \[
  \mathcal{O}(|P_k|)
  \]

### 4.2 Tổng thể

Với \(M\) vòng lặp:

\[
\mathcal{O}(M \cdot T)
\]

Trong thực tế:

\[
T \approx 10^9 - 10^{12}
\]

Do đó cần:
- Cấu trúc heap
- Cập nhật tần suất cục bộ
- Phân mảnh dữ liệu (sharding)

---

## 5. Ảnh hưởng đến Mô hình Ngôn ngữ

### 5.1 Số tham số embedding

Ma trận embedding:

\[
E \in \mathbb{R}^{V \times d}
\]

Số tham số:

\[
\text{Params} = V \times d
\]

Ví dụ:

- \(V = 50,000\)
- \(d = 4096\)

\[
\text{Params} = 204,800,000
\]

Nếu tăng \(V\) lên 100,000:

\[
\text{Params} = 409,600,000
\]

Chi phí tăng gấp đôi.

---

### 5.2 Ảnh hưởng đến Attention

Attention có độ phức tạp:

\[
\mathcal{O}(L^2 \cdot d)
\]

Trong đó:
- \(L\) là chiều dài chuỗi token.

Nếu token quá nhỏ (character-level):

\[
L \uparrow \Rightarrow \text{Chi phí tăng}
\]

Nếu token quá lớn (word-level):

- OOV tăng
- Mất khả năng phân tích hình thái

BPE cân bằng hai yếu tố này.

---

## 6. So sánh BPE với WordPiece và Unigram LM

| Thuật toán | Tiêu chí tối ưu | Cơ chế |
|------------|----------------|---------|
| BPE | Tần suất cặp | Gộp lặp |
| WordPiece | Likelihood | Chọn cặp tối đa hóa xác suất |
| Unigram LM | Xác suất mô hình | Loại bỏ token kém |

BPE là phương pháp tham lam (greedy):

\[
\max f_k(p)
\]

Trong khi WordPiece tối ưu:

\[
\max \log P(\mathcal{D} | V_k)
\]

---

## 7. Mối liên hệ với Huấn luyện LLM

Giả sử:

- Tổng token huấn luyện: \(T\)
- Kích thước mô hình: \(d\)
- Số lớp: \(L\)

Chi phí huấn luyện xấp xỉ:

\[
\mathcal{O}(T \cdot L \cdot d^2)
\]

Việc chọn tokenizer ảnh hưởng trực tiếp đến:

- \(T\) (số token sau phân tách)
- Hiệu quả tổng quát hóa
- Khả năng biểu diễn từ hiếm

---

## 8. Hạn chế

- Không xét ngữ nghĩa khi gộp
- Phụ thuộc dữ liệu huấn luyện ban đầu
- Có thể tạo token không trực quan

---

## 9. Kết luận

Thuật toán Byte Pair Encoding cung cấp một cơ chế phân tách từ hiệu quả, đặc biệt trong bối cảnh mô hình ngôn ngữ lớn. Bài toán đạt kích thước từ vựng mong muốn có thể được mô hình hóa thành việc thực hiện chính xác số vòng gộp cần thiết:

\[
M = V_{target} - |V_0|
\]

Việc tối ưu hóa BPE không chỉ là bước tiền xử lý, mà còn ảnh hưởng trực tiếp đến:

- Bộ nhớ
- Thời gian huấn luyện
- Chất lượng mô hình

Trong tương lai, các phương pháp tokenizer thích nghi động (adaptive tokenization) có thể thay thế BPE truyền thống nhằm tối ưu hóa tốt hơn theo mục tiêu huấn luyện.

---

## Tài liệu tham khảo

1. Gage, P. (1994). *A New Algorithm for Data Compression.*
2. Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.*
3. Vaswani, A. et al. (2017). *Attention Is All You Need.*
4. Kudo, T. (2018). *Subword Regularization.*
5. Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers.*

---
