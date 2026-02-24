# Phân Khảo Cấu Trúc Cụm (Clusters): Dấu Câu Nội Bộ vs Dấu Câu Kết Thúc Tập 2

## Tóm tắt (Abstract)
Thí nghiệm này là mảnh ghép hoàn thiện cho phân tích Mutual Information (MI) và Covariance trên các dấu câu. Chúng tôi tiến hành mở rộng không gian trực quan hóa (Visualizations) lên toàn bộ 25 khối ẩn chức năng (layers) của GPT-2 Medium. Bằng cách xây dựng lưới đồ thị $5 \times 5$, quá trình hoà trộn các cụm (clusters blending curve) được phơi bày. Đặc biệt, nghiên cứu này giải đáp nghịch lý về sự bùng nổ biên độ Covariance đi ngược chiều với sự suy giảm thông tin MI ở các tầng sâu, củng cố lý thuyết về tác động của độ lệch chuẩn (Standard Deviation) lên các phép đo giữ nguyên tỷ lệ (Scale-dependent metrics).

---

## 1. Mở Đầu (Introduction)
Phần 1 đã chỉ ra một lát cắt tại Layer 1: Covariance phân hóa dấu câu nội bộ (internal punctuation) thành 3 cụm riêng biệt, trong khi M.I hầu như không thấy rõ điều này. Thế nhưng, trí tuệ của LLM là một dòng chảy liên tục. Bước sóng chức năng của các Attention Blocks bắt đầu từ việc hiểu từ mộc (Embeddings) cho đến tổng hợp ngữ cảnh phức hợp (Deep layers).
Bài viết này tập trung dựng hình một thước phim quay chậm quá trình tiến hóa của 2 chỉ số đo lường kia dọc xuyên suốt 25 Layers, nhằm tìm ra điểm giới hạn nơi GPT-2 ngưng xử lý định nghĩa "cục bộ" rẽ nhánh của dấu phẩy/chấm.

---

## 2. Trực Quan Hóa Quỹ Đạo Phân Cụm (Cluster Trajectories)

### 2.1. Hiện Tượng Chập Điểm Ở Trạm Nhúng (Layer 0) 
Khi xuất dữ liệu tại Layer 0 (Embedding layer + Position encoding), đồ thị scatter plot của Covariance chỉ hiển thị đúng 2 điểm nén đặc (dù bản chất chứa 250 mẫu).
**Lý giải:** Ở tầng đầu tiên, mạng lưới chưa hề kích hoạt Attention. Mỗi token dẫu mang ý nghĩa gì thì Embedding Vector nội tại của nó là một mặt nạ hằng số bất biến. Do dấu câu mà ta lập trình trích xuất đều nằm cứng ở vị trí index 20, cả Positional Encoding cũng giống nhau. Vì vậy, LLM xử lý cả 250 câu như 1 mẫu độc bản tại tọa độ này.

### 2.2. Sự Tan Rã Của Các Cụm Nơi Tầng Sâu
Tiến hành nội suy ma trận $5 \times 5$ grid trên 24 transformer blocks thực hành:
- Các cụm Covariance $3$-Clusters sắc nét ở Layer 1 tiếp tục duy trì và bắt đầu có dấu hiệu loãng dần.
- Đến khoảng **Layer 7, 8 và 9** (1/3 chặng đường), các cụm này chính thức tan chảy và hợp nhất thành một dải mây dữ liệu (cloud of dots) phi cấu trúc. 
Hiện tượng này khẳng định: Các khối tầng nông (Early layers) chịu trách nhiệm nhận diện và phân loại rạch ròi các vai trò cú pháp phụ (như Dấu phẩy loại A, loại B). Một khi thông tin đã được tổng hợp xong, các khối tầng sâu hơn (Deep layers) không còn lưu trữ vách ngăn cú pháp này nữa, mà dốc toàn lực cho ngữ nghĩa dự đoán tương lai (Predictive Semantics).

---

## 3. Mâu Thuẫn Đo Lường Ở Đáy Mô Hình (Measurement Discrepancy)

### 3.1. Sự Trái Ngược Giữa Hai Sóng Đồ Thị
Khoan cắt biểu đồ phân phối Histogram trung bình dọc theo 25 Layers, một hiện tượng bất thường xuất hiện:
- **Mutual Information** có xu hướng trượt giảm dần khi đi sâu vào các Layers cuối.
- **Covariance** lại nổ tung, bắn dựng đứng lên không trung (tăng vọt lên hàng chục lần so với layer 1).

### 3.2. Hiệu Ứng Khuyếch Đại Phương Sai
Đây không phải là lỗi thuật toán. Sự trái ngược bắt nguồn từ sự khác biệt lõi của đơn vị đo:
- **MI dùng đơn vị Bits / Nats:** Là đại lượng "miễn nhiễm hệ số tỷ lệ" (Scale-independent). Nó thuần túy đo mức độ chắc chắn của xác suất.
- **Covariance giữ nguyên hệ số gốc (Scale-dependent):** Khi hệ thần kinh nội tại của LLM đi về các tầng sâu, Phương sai kích hoạt (Variance of activations) vươn lên rất lớn. Tín hiệu lan truyền lớn làm biên độ dao động nổ tung. Do $Cov(X,Y)$ tỷ lệ thuận với cường độ tín hiệu gốc, giá trị của nó tự động khuếch đại theo mà không mang lại thêm bất kỳ thông tin nội tại mới nào. 

Để giải quyết và đưa mức Covariance về một thước đo chuẩn mực, ta phải ép nó xuống bằng độ lệch chuẩn, tức là sử dụng **Hệ số tương quan Pearson (Pearson Correlation)**. 

---

## 4. Kết Luận
Thực nghiệm quét 25 layers dấy lên hai bài học lớn cho giới nghiên cứu Khả năng giải thích AI (Explainable AI):
1. **Lớp mạng quyết định chức năng:** LLM xử lý cú pháp tinh vi (như các loại dấu câu phân cách) chủ yếu diễn ra cục bộ ở 10 Layers đầu. 
2. **Cạm bẫy toán học của Covariance:** Đừng bao giờ so sánh tuyệt đối chỉ số Covariance giữa Layer 1 và Layer 24. Trừ khi bạn dùng hệ số Pearson để chuẩn hóa rào cản khuếch đại phương sai, M.I mới là công cụ so chiếu chéo Layers an toàn.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu mã code trích xuất từ `aero_LLM_14_CodeChallenge Clusters in internal vs. terminal punctuation (part 2).md` (Mã lệnh vòng lặp 25 layers sinh lưới lô-gíc $5 \times 5$, so đồ thị phân dải M.I và Cov, diễn giải Scale-dependence).
