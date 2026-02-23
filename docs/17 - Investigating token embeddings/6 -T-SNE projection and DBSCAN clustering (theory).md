# Phương Pháp T-SNE Và Thuật Toán Phân Cụm DBSCAN: Chiếu Không Gian Đa Chiều Cho LLMs

## Tóm tắt

Phân tích hình học vi mô trên các cụm không gian vector ngôn ngữ đang là mấu chốt của Machine Learning. Mọi nỗ lực tìm kiếm quy luật, khử nhiễu mã thông tin bên trong các LLMs đều vấp phải sự hạn chế quan sát đa chiều (>1000 chiều) của loài người. Bài báo này phân tích cơ sở toán vật lý cho hai kỹ thuật trụ cột: Làm phẳng không gian với thuật toán xác suất **t-SNE** (T-distributed Stochastic Neighbor Embedding) và Cắt lớp tổ hợp dữ liệu với phân cụm mật độ **DBSCAN** (Density-Based Spatial Clustering). Sự kết hợp này đưa ánh sáng đến cấu trúc "hộp đen" của Embeddings.

---

## 1. T-SNE: Nghệ Thuật Ép Không Gian Dựa Trên Xác Suất

Kỹ thuật t-SNE, được nghiên cứu và tiên phong bởi Geoffrey Hinton cùng cộng sự, chuyển đổi bài toán khoảng cách (Euclidean distance) thành bài toán tối ưu phân phối xác suất. Nếu hai vector nằm gần nhau theo luật hình học (nearest neighbors) tại gốc 1000 chiều đa ma trận, thì qua t-SNE, xác suất để chúng tiếp tục chạm nhau trên sàn 2 chiều (hoặc 3 chiều) là rất cao.

### 1.1 Tính Toán Phân Phối Ở Không Gian Điểm Ảnh Gốc
Đầu tiên, quy chuyển chuẩn hàm Softmax lên ma trận Euclidean. Tại lớp không gian bậc cao $X$, khả năng để vector $x_j$ nằm kề $x_i$ được biểu diễn bởi mật độ xác suất hàm mũ (Gaussian Gaussian Distribution):

$$ 
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
$$

Trong đó, $\sigma_i$ là phương sai (variance) chịu ảnh hưởng cấu hình phân tán kề lặp (Perplexity).

### 1.2 Chiếu Lên Chuẩn Bậc Thấp Và Tối Ưu Bằng Divergence
Hệ thống giả lập tiếp tục một chiều thấp $Y$ với cấu trúc Student t-Distribution nặng đuôi để ngăn cản hiện tượng đám đông nhồi nhét cực điểm (Crowding problem). Và mục đích vĩ đại của T-SNE là tinh chỉnh sao cho đồ thị phân phối khoảng cách cấu hình tại khối nhãn $Y$ mô phỏng chân xác nhất khối điểm $X$. Máy giải đạo hàm (Cost function gradient descent) thông qua việc kéo Min cho hàm chênh lệch **Kullback-Leibler (KL) Divergence**:

$$
C = \sum_{i} KL(P_i \parallel Q_i) = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

Sự trượt biến của Loss này khẳng định $Y$ đã tạo ra bóng ma 2 chiều sinh động của Mạng Nơ ron khổng lồ mà không phá hủy các quần tụ tương quan. Tính kết sinh của T-SNE là phi định chuẩn (Probabilistic/Non-deterministic). Mọi lần khởi động đều cho ra bản đồ khác trên nền tương đồng nhãn.

---

## 2. DBSCAN: Phân Lớp Không Gian Liên Kết Mật Độ Lân Cận

Khi t-SNE đã biến đám mây tham số ngẫu nhiên xuống còn mảnh đất phẳng trực quan, sự cần khát đi tìm các gia đình cấu trúc tiếp tục mở ra. DBSCAN (Density-Based Spatial Clustering of Applications with Noise) bỏ rơi tư duy tìm tâm cụm cổ thủ của K-Means, DBSCAN tiến hành gom mạng lân cận mật độ liên hành:

### 2.1 Định Quy Biến Số 
Thuật toán phóng tia quét tìm kiếm quanh các node vector dựa trên hai siêu tham số nền tảng:
- Cự ly biên độ $\epsilon$ (Epsilon distance threshold): Độ dài ngưỡng tia bán kính bao phủ một vùng.
- Ngưỡng giới hạn quân số (MinPts): Số điểm tối thiểu phải lọt vào lưới $\epsilon$ để tạo thành một khối cộng đồng liên đoàn.

### 2.2 Đọc Điểm Gây Loãng (Noise) và Điểm Kết Tinh (Core points)
Mọi quần đảo nối chuỗi lẫn nhau nhờ $\epsilon$ hợp thức hóa thành những nhánh phân chùm hữu cơ vĩ đại. Những Vector lạc loài với khoảng cách xa ngoài chùm $\epsilon$ được thải trừ thành phần bù (Noise points - Những biến dị nhiễu không gây ảnh hưởng đến trung tâm tổ chức cụm biểu diễn). Mức độ khắt khe biến động tỷ lệ thuận cùng sự tăng số MinPts hoặc bóp nghẹt $\epsilon$.

---

## 3. Hình Thành Đồ Thị Tương Quan Ma Trận Gram (Gram Matrix)
Ở lớp phân lớp toán học sâu hơn, cả t-SNE hay phân tập DBSCAN đều giải phẫu thông qua Ma trận Đồ Đồng Cấu Gram (Gram Matrix) của một bộ vi xử lý Vector nhúng:
$$
G_{E} = E \cdot E^T 
$$
Khi các vector được phân bổ đơn vị với lượng Vector-norm chuẫn L2, Gram Matrix lập tức hóa thân thành khối ảnh chiếu Cosine Similarity Matrix. Nó tiết lộ những kiến trúc lưới đồ thị sắc sảo đang giấu nhẻm ở đám mây khối $n$-nghiệm phức loạn. 

---

## 4. Kết luận
Bộ đôi Toán-Xác Suất t-SNE kết hợp DBSCAN cung phụng khả năng thám sát kỳ diệu, biến hệ thập nguyên ngàn chiều của Machine Learning thu gọn vào tầm tay hình học lớp đại cương. Thay vì bóp cong cấu trúc để ép vào chuẩn tâm (Centroids error), phép chiếu mật độ lân cận t-SNE giải trình nguyên vẹn sự kết nối thông qua đạo hàm KL và Epsilon threshold.

---

## Tài liệu tham khảo

1. **Laurens van der Maaten, L., & Hinton, G. (2008).** *Visualizing Data using t-SNE.* Journal of Machine Learning Research.
2. **Ester, M., et al. (1996).** *A density-based algorithm for discovering clusters in large spatial databases with noise (DBSCAN).* KDD.
3. **Schubert, E., et al. (2017).** *DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN.* 
4. Tài liệu bài giảng *Investigating token embeddings - T-SNE and DBSCAN (theory)*.
