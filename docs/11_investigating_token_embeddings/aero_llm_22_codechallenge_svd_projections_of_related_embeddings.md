
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [11 investigating token embeddings](index.md)

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
# Ánh Xạ SVD Các Dải Điểm Nhúng Có Quan Hệ Chéo

## Tóm tắt

Một trong những giới hạn khi phân tích toàn bộ Không gian Embeddings bằng SVD (Singular Value Decomposition) là sự bão hòa nhiễu - những thành phần chính (Principal Components) thường đánh mất độ chi tiết do phải gánh đỡ một tỷ lệ phương sai khổng lồ của cả trăm ngàn cụm từ dị biệt. Giải pháp đối trọng là khoanh vùng tọa độ vi mô: Thay vì nén toàn thể mảng BERT, ta tạo ra các "Ma trận Con" (Submatrices) chứa duy nhất tổ hợp token mang đặc trưng nhóm (VD: Tên quốc gia, chữ số). Bài phân tích dưới đây minh chứng năng lực của SVD trong việc tìm ra những trục liên kết đồng dạng ẩn dật dưới các nhóm từ vựng cụ thể.

---

## 1. Kiến Tạo Ma Trận Con (Submatrices Embeddings)

Cấu hình thử nghiệm được thiết kế dựa trên 2 tập dữ liệu mẫu trích từ mô hình BERT:
1. **Tập 10 Chữ số đơn:** `["0", "1", "2", ..., "9"]`
2. **Tập 10 Quốc gia Liên Minh Châu Âu (EU):** `["France", "Germany", "Italy", "Spain", ...]` (Chọn lọc ưu tiên các quốc gia không bị băm ngang bởi tokenizer để đảm bảo luật *1 word = 1 token*).

Hai ma trận con nhận được (Matrix $M_{\text{digits}}$ và $M_{\text{EU}}$) có chung kích thước $10 \times 768$.
Tính độc lập phân phối (Orthogonality mapping) được khẳng định ngay từ bước thử nghiệm khi Ma trận Vectơ Trung Bình (Mean vectors) của tập Chữ số và tập EU trả về mức tương quan cực thấp ($r $\approx$ 0.01$). Điều này chứng minh 2 đám mây tọa độ này bay xa nhau hoàn toàn trong cấu trúc dải ngân hà 768 chiều.

---

## 2. Loại Bỏ Đường Tiệm Cận Bằng Kỹ Thuật Dịch Tâm (Mean-Centering)

Trước khi tiến hành phân rã nhân ma trận $M$, mọi cấu trúc dữ liệu hình học tuyến tính đều phải tiến hành lùi tâm (Mean Centering).
Tính tịnh tiến này cưa bỏ khoảng cách dư thừa từ điểm $0$ đến lõi đám mây dữ liệu:

\hat{M}_{i} = M_{i} - \mu

*(Với \mu là vector trung bình cực đại có độ dài bằng số cột kích thước D=768).*

$$
Khi Mean-centering được thực thi chặt chẽ, đường quang phổ giá trị suy biến (Singular value spectrum / Scree plot) từ SVD sẽ có đặc tính rỗng dư tại giá trị cuối cùng. Nói cách khác, thuật toán cưa đi một *bậc tự do* (Rank minus 1), biểu diễn bằng việc singular value cuối cùng sẽ đâm thẳng về 0. Nếu không lùi tâm, trục phân phối SVD sẽ dồn toàn bộ sự khác biệt vào Component-1 (Trục thứ 1), làm sai lệch khả năng đọc hiểu Component-2. --- ## 3. Khám Phá Ý Nghĩa SVD Bằng Phép Chiếu Nghịch Tập Hợp (Over-Projections) ### Khái Niệm Phép Chiếu Rộng Rãi: Sau khi SVD thành công \hat{M}_{\text{EU}} = U \Sigma V^T, chúng ta thu được chùm Vector riêng biệt đặc tả tính "*Châu Âu*" nắm giữ tại hàng thứ tự đầu tiên của đa giác V^T (Kí hiệu V_{\text{top}}). Phép màu giải thích nằm ở bước sau: Thay vì giới hạn khảo sát trên 10 nước Châu Âu, ta lấy **toàn bộ 30.000 tokenizer còn lại của hệ BERT**, trừ đi \mu_{\text{EU}}, rồi nhân tích vô hướng đổ bóng toàn bộ 30.000 từ này lên trục V_{\text{top}}: \text{Projections} = (E_{\text{all\_tokens}} - \mu_{\text{EU}}) \cdot V_{\text{top}}
$$

