# Khai Thác Thuật Toán k-NN Cho Tìm Kiếm Từ Đồng Nghĩa Trên BERT

## Tóm tắt

K-Nearest Neighbors (k-NN - k Láng giềng gần nhất) là một thuật toán cốt lõi trong phân loại cụm học máy cổ điển (Machine Learning Classification). Tuy nhiên, khi kết hợp cùng Vector nhúng (Embeddings) của họ mô hình ngôn ngữ lớn (LLMs) như BERT, thuật toán này thể hiện khả năng tra cứu chéo từ đồng nghĩa (synonym search) ở mức độ đáng kinh ngạc. Bài viết dưới đây trình bày nguyên lý không gian của k-NN, phân biệt hai định chuẩn tính điểm Euclidean và Cosine Similarity, cũng như cách triển khai cho các chiều ẩn trong từ điển số hóa tự nhiên.

---

## 1. Nguyên Lý Của k-Nearest Neighbors (k-NN)

Trong mô hình k-NN, dữ liệu mới không hề được gắn nhãn trước theo một hệ số chặn hàm (Linear threshold). Cách tiếp cận này tuân theo một cơ chế định vị hình học đơn giản: **Dữ liệu thuộc về cộng đồng nào đang áp đảo xung quanh nó.**

1. Đầu vào là một Vector truy vấn vô danh (Unlabeled data point) $x$.
2. Hệ thống quét đếm khoảng cách từ $x$ đến toàn bộ những Vector $\vec{v}_i$ đang mang nhãn dữ liệu có sẵn trong bộ nhớ.
3. Tham số $k$ quy định sẽ lấy $k$ điểm có khoảng cách không gian sát với $x$ nhất (Nearest Neighbors). Lời khuyên là thiết lập $k$ thành số lẻ (VD: $k=3, 5, 7$) để chặn trường hợp hòa/cân bằng.
4. Lựa chọn ưu tiên theo nguyên lý Bầu chọn theo khuynh hướng (Majority Voting): nhãn hiệu chiếm đa số trong $k$ phần tử cận kề sẽ được gán cho $x$.

Trường hợp sử dụng để khám phá từ loại trong BERT, việc dự đoán theo lớp sẽ được thay thế bằng liệt kê $k$-vector gần nhất với từ gốc nhằm tìm ra các từ đồng nghĩa (vd: "Beauty" sẽ gọi ra "Gorgeous", "Elegance"). 

---

## 2. Số Học Khoảng Cách: Euclidean và Cosine Similarity

Không gian tọa độ của mảng Embeddings ma trận BERT sở hữu $D=768$ chiều. Bài toán tìm Láng giềng (Distance calculations) yêu cầu một thước đo chuẩn. Hai thước đo thông dụng đem lại hai góc nhìn dị biệt:

### 2.1. Chuẩn Khoảng Cách Hình Học (Euclidean Distance)
Lấy gốc từ định lý tam giác vuông trong không gian $N$-chiều, Euclidean đo đạc chiều dài thật sự của sợi dây nối giữa mũi tên vector token $\vec{v}$ và token mục tiêu $\vec{w}$:
$$ 
\delta(\vec{v}, \vec{w}) = \sqrt{\sum_{i=1}^{D} (v_i - w_i)^2} 
$$
Chuẩn Euclidean thể hiện tính tách biệt tuyệt đối (absolute spatial magnitude) của thông tin.

### 2.2. Chuẩn Tương Quan Góc (Cosine Similarity)
Trọng tâm đo lường sự đồng dạng không nằm ở lực độ dài, mà vứt bỏ tất cả giới hạn véc-tơ để tìm độ chênh góc giữa hai ngọn vector:
$$ 
\text{CosineSim}(\vec{v}, \vec{w}) = \frac{\vec{v} \cdot \vec{w}}{\|\vec{v}\| \|\vec{w}\|} \in [-1, 1] 
$$

**Sự Lệch Pha Đáng Lưu Ý:** Các vector có chung hướng nội hàm (Cosine Similarity hướng về 1) nhưng hoàn toàn có thể sở hữu Khoảng cách Euclidean kéo dãn ra khổng lồ nếu độ phủ vector (Norm of vector) bị đẩy cực xa gốc tọa độ. Do đó, việc tìm Láng giềng gần nhất k-NN trong cấu trúc BERT đòi hỏi nhà nghiên cứu phải xác định thuộc tính đang săn tìm là khoảng cách hay góc lệch nhạy cảm biểu diễn song song.

---

## 3. Khai Thác Tiền Xử Lý Giảm Chiều Bằng PCA/t-SNE

Khi ứng dụng tệp $k=5$ cho cụm từ "Beauty" xuyên thấu toàn bộ $30.000$ từ điển của BERT, gánh nặng toán học (Tính $30.000$ phép tính hàm mũ $L2-Norm$) có thể sẽ làm đình trệ bộ vi xử lý nếu hệ vector lớn như định dạng GPT hiện đại (với số token trên 1 triệu). 
Theo lý thuyết thông luật của Học Máy (Machine Learning), để tránh "Lời nguyền đa chiều" (Curse of Dimensionality), ma trận nên được phân rã bằng Principal Component Analysis (PCA) triệt tiêu quang phổ yếu (SVD variance noise) sinh ra một ma trận giảm chiều $D = 100$ trước khi hàm k-NN khởi chạy, đảm bảo chi phí thấp mà không đánh tụt độ nhạy tương quan ngữ nghĩa.

---

## 4. Kết luận

Mô hình k-Nearest Neighbors là khối hạt nhân trong mọi bộ truy vấn tìm điểm dữ liệu (Search Engines) ứng dụng vào Mạng Nơ-ron. Việc lạm dụng tính chất khoảng cách ở vùng Embeddings của BERT cho phép k-NN bứt phá khỏi cơ chế nhãn mác nhị phân, trở thành công cụ đắc lực giải phẫu hiện tượng đa nghĩa từ vựng cũng như khai thác vùng giao thoa khái niệm (Concept boundary overlapping).

---

## Tài liệu tham khảo

1. **Cover, T., & Hart, P. (1967).** *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory.
2. **Devlin, J., et al. (2018).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
3. Tài liệu đào tạo *Investigating token embeddings - kNN for synonym-searching in BERT*.
