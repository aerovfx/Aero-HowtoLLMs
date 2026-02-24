# Phân Rã Token, Nhúng Và Phân Cụm Biểu Tượng Emojis Bằng Đồ Thị Mật Độ

## Tóm tắt

Các Mô hình Ngôn ngữ Lớn (LLMs) như ChatGPT thường xuyên lạm dụng và thể hiện mức biểu cảm cao thông qua các biểu tượng cảm xúc (Emojis). Từ góc nhìn điện toán, chúng ta xem một Emoji như một hình vẽ duy nhất, nhưng cấu trúc Tokenizer của GPT lại "đọc" chúng thông qua lăng kính cực kỳ phức tạp dựa trên quy chuẩn mã hóa Unicode đa kí tự. Bài viết này trình bày phân tích khoa học về sự hình thành các Vector nhúng (Embeddings) của Emoji, và ứng dụng kết hợp **t-SNE** cùng **DBSCAN** để phân cụm thế giới cảm xúc của AI.

---

## 1. Cấu Trúc Khối Bất Đối Xứng Giữa Ký Tự và Tokenizer

Hệ quy chuẩn máy tính nhận diện mọi ký tự thông qua tọa độ Thập lục phân (Hexadecimal Unicode). Ví dụ chữ cái 'A' là `U+0041` (hay mảng số nguyên Thập phân là $65$). Ở hệ tiêu chuẩn này, Emojis không có bất kỳ đặc quyền nào, chúng là những tổ hợp chuỗi giá trị Unicode có chỉ mục rất cao trên hệ 150.000 ký tự.

Vấn đề nghiêm trọng nảy sinh ở bộ cắt từ ngữ (Tokenizer) của GPT: *Hầu như không một Emoji nào được nén thành một Token duy nhất.*
Những biểu tượng trông rất bình thường thường bị cấu thành bởi 2 đến 4 tokens nối tiếp nhau. Lẽ ra, quy luật 1-token-1-đối-tượng đảm bảo vector $\vec{v}$ gánh trọn vẹn ngữ nghĩa khối. Tính phi tuyến tính này yêu cầu Mạng Nơ-ron (Neural Network) phải kết hợp (Attention mechanism) chùm token liên hoàn thành một hệ quy chiếu cảm xúc logic duy nhất. Trái với linh cảm, vector của token đầu tiên thuộc Emoji hoàn toàn không bao quát đủ hàm ý của Emoji gốc (Cosine Similarity giữa first-token và last-token của một Emoji chỉ rơi vào khoảng $\approx 0.3$).

---

## 2. Tính Toán Nhúng Emojis Và Quá Trình Hợp Nhất (Mean Pooling)

Vì một Emoji (giả sử Cười) bị cắt rách thành tổ hợp $K$ tokens $\left[ t_1, t_2, ..., t_K \right]$, chúng ta sẽ thu về một tập hợp các ma trận nhúng $\vec{e}_1, \vec{e}_2, ..., \vec{e}_K$.
Để có được một đại lượng Embeddings duy nhất $\vec{E}_{\text{emoji}}$ nhằm tính toán khoảng cách vector từ hoặc tương quan góc (Cosine Similarity), phương án nền móng là tính Trung bình cộng vector (Vector Ave18-RAGe / Mean Pooling):

$$
\vec{E}_{\text{emoji}} = \frac{1}{K} \sum_{i=1}^{K} \vec{e}_i
$$

Bằng cách tạo một ma trận hỗn hợp $N \times 768$ chiều (giả sử chọn tập $N=32$ Emojis), toàn bộ đám mây cảm xúc đã được định chuẩn hóa lên không gian nơ-ron bậc cao của khối lượng Transformers.

*(Lưu ý: Trung bình cộng token vector trong Word Embeddings chỉ là kỹ thuật đơn giản. Đối với mô hình sâu (Deep layers), ta nên trích vector tọa độ của token cuối cùng ở lớp attention thứ 12 để gom hết dữ kiện contextual từ các token trước).*

---

## 3. Phân Cụm Ý Niệm Emojis Bằng Đồ Thị t-SNE và DBSCAN

Nhằm chẩn đoán xem AI có thật sự phân biệt được "Nhóm Tim", "Nhóm Cười", "Nhóm Chó Mèo" với nhau ở cấp vector, ta trải mạng t-SNE và DBSCAN:

### 3.1 Nén t-SNE xuống hệ mặt phẳng Euclidean 2D
Giản đồ t-SNE sử dụng phân phối Gaussian chuẩn để kéo sập khối tọa độ 768 chiều xuống một sàn phẳng 2 tọa độ (2D Coordinates). Kết quả trên 32 mặt emoji sẽ tạo nên các dải ngân hà liên đới chặt chẽ.

### 3.2 Chuẩn Hóa Z-Score (Standardization)
Trước khi chạy DBSCAN, kết quả đồ thị t-SNE buộc phải được quy đổi sang một trung tâm chuẩn hóa khoảng cách độ lệch (Standard Deviation Units):
$$
Z = \frac{X - \mu}{\sigma}
$$
Phép dịch tâm $Z-score$ này bảo toàn nguyên vẹn tính chất hình học tương đối nhưng đem toàn bộ trục tung và trục hoành thu gọn vào khoảng từ $-2$ đến $2$. Việc này cung cấp sức mạnh định dạng bán kính cực độ cho DBSCAN.

### 3.3 Phân cụm Epsilon ($\epsilon$) qua DBSCAN
Trong hàm DBSCAN, tham số cốt tử là $\epsilon$ (khoảng cách tối đa để kết nối hai điểm hạt nhân liên hoàn thành 1 cụm). Do tọa độ đã bị chuẩn hóa $Z-Score$, việc chọn $\epsilon = 0.3$ mang ý nghĩa "Kết nối mọi điểm lân cận trong vòng bán kính 0.3 Độ lệch chuẩn phương sai".

Kết quả nhận được rất xuất sắc: Ma trận nhúng GPT-2 tụ hợp các nhóm Cười, nhóm Tình Yêu (Tim), và Nhóm Động vật vào những khối liền kề biệt lập. GPT thực sự hiểu phương sai ngữ nghĩa của đồ họa Unicode hệt như từ vựng của tiếng người.

---

## Tài liệu tham khảo

1. **Eisner, B., et al. (2016).** *emoji2vec: Learning Emoji Representations from their Description.* EMNLP.
2. **Barbieri, F., et al. (2016).** *Does Multiword Expression help Word Representation?* EACL (Phân tích sự ảnh hưởng của token phân mảnh).
3. Tài liệu đào tạo *Investigating token embeddings - Tokenize, embed, and cluster emojis*.
