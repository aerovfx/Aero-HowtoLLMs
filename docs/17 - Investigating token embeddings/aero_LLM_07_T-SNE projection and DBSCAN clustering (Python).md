# Phân Cụm Ngữ Nghĩa Qua Phép Chiếu t-SNE & Mật Độ DBSCAN (Python)

## Tóm tắt

Mạng thông tin mã véc-tơ trong các mô hình tự hồi quy mạnh mẽ như GPT-2 thường bị đóng gói ở lớp màng 768 chiều. Để mắt thường linh trưởng có thể tìm ra những vi quần thể đồng dạng kết nối từ kho vựng, hai thuật toán siêu năng lực được kết dính lại: **t-SNE** (T-Distributed Stochastic Neighbor Embedding) đóng vai trò thợ ép phẳng ma trận xuống 2 Chiều Không Gian, và thuật toán **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) đóng vai trò kẻ dò tìm biên giới gạ lọc các tổ hợp có độ kết dính mật độ lõi cao.
Nghiên cứu dưới đây sẽ lột trần từng ngóc ngách của quá trình tổ chức cấu trúc của Gram-Matrix.

---

## 1. Gram-Matrix: Bức Tranh Tổng Thể Nội Tại Tiền Mật Độ

Để tránh quá tải thị giác, ta cắt gọn một Ma trận con 100 Tokens đầu tiên $\mathbf{E}_{\text{sub}} \in \mathbb{R}^{100 \times 768}$ từ GPT-2. 
Lập tức tạo ngay Gram-Matrix $\mathbf{G} = \mathbf{E}_{\text{sub}} \cdot \mathbf{E}_{\text{sub}}^T \in \mathbb{R}^{100 \times 100}$. Hình ảnh chéo đối xứng phơi bày một trật tự sâu sắc không thể thấy khi nhìn ngang Matrix:
Các ô chấm vuông đậm đặc xuất hiện liên kết chéo cho các nhóm Tín hiệu đặc thù: (Chữ số Arab, Dấu câu Punctuation, Hệ thống Cấu trúc chữ cái Alphabetical Capital - Lowercase). Gram-Matrix dọn đường trước ranh giới hệ ý niệm.

---

## 2. Sự Nén Màng Và Cơ Chế Hoạt Động Của Thuật Toán t-SNE

Với $\text{perplexity} = 5$ (đại diện cho ngưỡng độ linh hoạt tìm lân cận - smoothing parameter), t-SNE nén nhẹp hệ thống tọa độ Vector bằng định lý xác suất khoảng cách điểm từ $768D$ co về $\mathbf{Y} \in \mathbb{R}^{100 \times 2}$.

**Lỗ hổng của Phép Xác Suất t-SNE:**
Do bản tính thiết kế hàm phân phối lân cận Gauss ngẫu nhiên (Probabilistic initializations) của hàm loss Kulllback-Leibler, biểu đồ 2D cho ra những mảng đảo cụm token cực kỳ khác nhau qua mỗi lần tái chạy Model (Stochastic solving). Ví dụ: Nhóm Chữ cái in hoa $X, Y, Z$ vẫn kết dính nhau nhưng có thể xuất hiện lúc ở tọa độ Góc Đông Bắc, lúc thì ở Góc Tây Nam. Dẫu vậy, khoảng cách giữa các quần thể bản ngã (Global clusters structures) luôn bảo lưu đặc trưng cô đặc phi tuyến.
t-SNE chỉ hạ tầng chiều cao tạo đám mây rải rắc chứ không phân cụm, đây là lúc DBScan tung chiêu.

---

## 3. Khóa Quần Thể Nhờ Đường Biên Mật Độ DBScan

Cấu hình DBSCAN: 
- `Epsilon = 6.0` (Độ dài đường kính vòng lân cận giới hạn)
- `Min_samples = 3` (Số điểm tối thiểu để đạt mốc lõi Core-Point để gọi là một cụm).

Đưa ma trận 2D của t-SNE vào chảo lửa DBScan, thuật toán sẽ đi săn các chuỗi liền kề bằng lưới quét bán kính. 
Hệ thống sẽ dán các số nguyên ngẫu hình (Integer labels) cho các tụ điểm: *Group 0, Group 1...*
Đặc biệt, hệ thống sinh ra điểm $\text{Label} = -1$ . Đây là các Outlier Noises (Trôi dạc cô độc). Ví dụ: Nếu chỉ có 2 chữ cái `[y, Y]` nằm gần nhau, nhưng vì `Min_samples = 3`, vòng tròn Epsilon không đủ dân số nên thuật toán hất bỏ chúng về lại nhóm phân ly.

Sự lợi hại của DBScan đánh gục thuật toán *K-Means clustering* truyền thống vì nó không đòi hỏi kỹ sư phải "Đoán Mò Số Học" có sẵn định kiến quy mô bao nhiêu cụm (k). DBscan tự do giãn nở như màng bọt sinh học hễ thấy mật độ dầy sẽ lập tức khoanh tròn lại tổ chức cho ta.

---

## 4. Hiện Tượng Căng Tràn Thuật Toán (Parameters Breakdown)

Toàn bộ hệ thống t-SNE kết dính DBscan là một sự kết hợp mong manh.
Khi vô tình hoặc cố ý điều chỉnh `Epsilon = 16.0`: DBscan phá vỡ đường cương định vị, phóng bán kính quét cực kỳ thô thiển dẫn đến hiện tượng gộp dính toàn bộ $100$ điểm chóp vào một Bồn trũng khổng lồ không còn khả năng phân rã. 
Hoặc với biến thiên nhỏ hơn khi Epsilon giật xuống, vô số cụm siêu nhỏ bị bẻ vụn phân liệt thành các phân khu sai lệch ngữ nghĩa (Ví dụ: `galaxy` bị gộp lộn xộn vào với các cụm Token cú pháp `syntax`, `regex`, `codex`). 

Điều này gióng lên một hồi chuông khoa học: Mô phỏng không gian giảm chiều kích (Visualization of Reduced Dimension) cho Machine Learning không phải cứ nhìn thấy cụm tụ thì chúng mang chung một luồng nội hàm. Việc các tổ hợp gộp dính lại với nhau hay lảng tránh nhau hoàn toàn có thể là tác phẩm từ sự điều phối thủ công tham số ngoại lai Parameters bởi chính con người áp đặt, tạo ra điểm mờ về thiên kiến giải thích phân liệt học thuật. 

---

## Tài liệu tham khảo

1. **Laurens van der Maaten & Geoffrey Hinton (2008).** *Visualizing Data using t-SNE.* Journal of Machine Learning Research.
2. **Ester, M., et al. (1996).** *A density-based algorithm for discovering clusters in large spatial databases with noise (DBSCAN).* KDD.
3. Tài liệu mô phỏng kỹ thuật thực hành phân tích *T-SNE projection and DBSCAN clustering.*
