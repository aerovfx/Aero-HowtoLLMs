# Ánh Xạ SVD Các Dải Điểm Nhúng Có Quan Hệ Chéo

## Tóm tắt

Một trong những giới hạn khi phân tích toàn bộ Không gian Embeddings bằng SVD (Singular Value Decomposition) là sự bão hòa nhiễu - những thành phần chính (Principal Components) thường đánh mất độ chi tiết do phải gánh đỡ một tỷ lệ phương sai khổng lồ của cả trăm ngàn cụm từ dị biệt. Giải pháp đối trọng là khoanh vùng tọa độ vi mô: Thay vì nén toàn thể mảng BERT, ta tạo ra các "Ma trận Con" (Submatrices) chứa duy nhất tổ hợp token mang đặc trưng nhóm (VD: Tên quốc gia, chữ số). Bài phân tích dưới đây minh chứng năng lực của SVD trong việc tìm ra những trục liên kết đồng dạng ẩn dật dưới các nhóm từ vựng cụ thể.

---

## 1. Kiến Tạo Ma Trận Con (Submatrices Embeddings)

Cấu hình thử nghiệm được thiết kế dựa trên 2 tập dữ liệu mẫu trích từ mô hình BERT:
1. **Tập 10 Chữ số đơn:** `["0", "1", "2", ..., "9"]`
2. **Tập 10 Quốc gia Liên Minh Châu Âu (EU):** `["France", "Germany", "Italy", "Spain", ...]` (Chọn lọc ưu tiên các quốc gia không bị băm ngang bởi tokenizer để đảm bảo luật *1 word = 1 token*).

Hai ma trận con nhận được (Matrix $M_{\text{digits}}$ và $M_{\text{EU}}$) có chung kích thước $10 \times 768$.
Tính độc lập phân phối (Orthogonality mapping) được khẳng định ngay từ bước thử nghiệm khi Ma trận Vectơ Trung Bình (Mean vectors) của tập Chữ số và tập EU trả về mức tương quan cực thấp ($r \approx 0.01$). Điều này chứng minh 2 đám mây tọa độ này bay xa nhau hoàn toàn trong cấu trúc dải ngân hà 768 chiều.

---

## 2. Loại Bỏ Đường Tiệm Cận Bằng Kỹ Thuật Dịch Tâm (Mean-Centering)

Trước khi tiến hành phân rã nhân ma trận $M$, mọi cấu trúc dữ liệu hình học tuyến tính đều phải tiến hành lùi tâm (Mean Centering).
Tính tịnh tiến này cưa bỏ khoảng cách dư thừa từ điểm $0$ đến lõi đám mây dữ liệu:
$$ 
\hat{M}_{i} = M_{i} - \mu 
$$
*(Với $\mu$ là vector trung bình cực đại có độ dài bằng số cột kích thước D=768).*

Khi Mean-centering được thực thi chặt chẽ, đường quang phổ giá trị suy biến (Singular value spectrum / Scree plot) từ SVD sẽ có đặc tính rỗng dư tại giá trị cuối cùng. Nói cách khác, thuật toán cưa đi một *bậc tự do* (Rank minus 1), biểu diễn bằng việc singular value cuối cùng sẽ đâm thẳng về $0$. Nếu không lùi tâm, trục phân phối SVD sẽ dồn toàn bộ sự khác biệt vào Component-1 (Trục thứ 1), làm sai lệch khả năng đọc hiểu Component-2.

---

## 3. Khám Phá Ý Nghĩa SVD Bằng Phép Chiếu Nghịch Tập Hợp (Over-Projections)

### Khái Niệm Phép Chiếu Rộng Rãi:
Sau khi SVD thành công $\hat{M}_{\text{EU}} = U \Sigma V^T$, chúng ta thu được chùm Vector riêng biệt đặc tả tính "*Châu Âu*" nắm giữ tại hàng thứ tự đầu tiên của đa giác $V^T$ (Kí hiệu $V_{\text{top}}$).

Phép màu giải thích nằm ở bước sau: Thay vì giới hạn khảo sát trên 10 nước Châu Âu, ta lấy **toàn bộ 30.000 tokenizer còn lại của hệ BERT**, trừ đi $\mu_{\text{EU}}$, rồi nhân tích vô hướng đổ bóng toàn bộ 30.000 từ này lên trục $V_{\text{top}}$:
$$ 
\text{Projections} = (E_{\text{all\_tokens}} - \mu_{\text{EU}}) \cdot V_{\text{top}} 
$$

### Diễn Dịch Chóp Đồ Thị (Extremes Projections):
Thống kê 30 token có tích vô hướng văng ra xa nhất trên Trục $V_{\text{top}}$ (Top positive / Top negative Projections) mở ra chân trời cơ chế máy học:
- Ở dải cực âm của Trục Châu Âu, chúng ta bắt gặp những từ vựng không hề nằm trong nhóm gốc đào tạo nhưng cùng một hệ trục địa lý ngôn ngữ như: *Latvian, Tallinn, Vilnius, Estonian*.
- Ở dải của ma trận Chữ Số, các cực đoan dự đoán kéo theo sự xuất hiện của các chuỗi text format số (VD: *Seven, Null, Zero, Divided*), chứng tỏ trục không gian toán học có khả năng nối kết hình dáng số ("7") với ký hiệu văn bản ("Seven").

---

## 4. Kết luận

Sự phân mảng Ma trận con (Submatrices Extracting) cung cấp một khung kính lúp mạnh mẽ giảm bớt nhiễu loạn ngẫu nhiên của toàn bộ thư viện ngôn ngữ tự nhiên. Phương pháp lấy SVD tạo ra ma trận V, rồi đem toàn bộ đại dương Embeddings phản kích dội ngược chiếu rọi lên $V$ chính là một chiếc kính rọi đèn soi sáng cấu trúc nội mạc (Mech Interpretability) cho thấy cách hàng tỷ ma trận thông số Neural Network móc nối khái niệm của con người thành mạng nhện tính toán.

---

## Tài liệu tham khảo

1. **Turian, J., et al. (2010).** *Word representations: A simple and general method for semi-supervised learning.* ACL.
2. **Deerwester, S., et al. (1990).** *Indexing by latent semantic analysis.* JASIS.
3. Tài liệu thực hành định lượng *SVD projections of related embeddings*.
