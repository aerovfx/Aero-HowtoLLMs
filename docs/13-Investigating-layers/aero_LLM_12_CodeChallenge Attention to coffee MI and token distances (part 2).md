# Thử Thách Lập Trình (Code Challenge): MI Và Khoảng Cách Token (Phần 2)

## Tóm tắt (Abstract)
Thực nghiệm này nâng cấp khảo sát về tác động của khoảng cách vật lý giữa hai từ giống nhau ("coffee") đối với Mutual Information (MI), mở rộng trên toàn bộ biểu đồ 48 Layers của GPT-2 XL. Quá trình tính toán diễn ra song song trên cả biến thể nhánh $Attention$ và mạng $MLP$. Thông qua việc kết hợp các tiêu chuẩn kiểm định mạnh như *T-tests độc lập*, *Chuyển đổi Fisher Z-Transform cho hệ số tương quan*, và *Hiệu chỉnh đa so sánh FDR (False Discovery Rate/Bonferroni)*, báo cáo vạch ra ranh giới rẽ nhánh rõ rệt giữa nhiệm vụ dung nạp bối cảnh mở rộng của Attention và cơ chế nhớ tĩnh của MLP. 

---

## 1. Mở Đầu (Introduction)
Ở Phần 1, chúng ta đã chứng minh cơ bản tại Layer 3: Hai từ cùng gốc đứng càng xa nhau thì thông tin chia sẻ nội bộ của chúng càng nghèo nàn. Tuy nhiên, kiến trúc máy học LLM là một hành trình đi dần vào chiều sâu (depth propagation). Phần 2 đặt ra hai giải pháp nâng cao hơn: 
1. Diễn giải sự thay đổi hiệu ứng khoảng cách này xuyên qua 48 Transformer blocks.
2. Đối chiếu trực diện vai trò tạo lập liên kết (M.I) giữa hạt nhân truy vấn song song ($Attention\ C\_proj$) và hạt nhân tuyến tính xử lý hàm tiến ($MLP\ C\_proj$).

---

## 2. Nâng Cấp Phương Pháp Thống Kê (Methodology Expansions)

### 2.1. Vòng Lặp Trải Phẳng (Laminar Loop)
Thiết lập mảng 3 chiều ma trận `my_results = (2 x 48 x 2)` tương đương: [Attention / MLP] $\times$ [Layers 1...48] $\times$ [Average MI / Kendall tau correlation]. Việc loại bỏ nhiễu Z-score $> 4$ (Outliers Trimming) vẫn luôn được duy trì ở toàn bộ các cấp tính toán.

### 2.2. Kiểm Định T-Test Giữa MLP Và Attention
Để xác nhận MI tại nhánh Attention có thực sự khác biệt so với MI của nhánh MLP ngay tại cùng một Layer hay không, ta lấy mảng dữ liệu (Tất cả Pairwise MI non-zero) của hai bên và cho chạy mô hình $Independent\ T-Test$ (Thu được $t-statistic$ và $p-value$). Để ngăn chặn sai lầm loại I do "test mỏi tay" 48 lần, bộ hiệu chỉnh đa biến Bonferroni hoặc FDR được kích hoạt.

### 2.3. Chuyển Đổi Fisher Z-Transform Cho So Sánh Correlation
Để so sánh hai hệ số tương quan (Kendall) của Attention và MLP, ta không thể dùng T-test vì nó không phải mẫu phân bổ đo lường tuyệt đối. Ta sử dụng Fisher Z-transform:
$$ Z = \frac{ \text{arctanh}(r_{att}) - \text{arctanh}(r_{mlp}) }{\sqrt{2 / (N - 3)}} $$
Kiểm tra Z-score này trên Phân phối tích lũy chuẩn (Normal CDF) sẽ cho phép xác định độ khác biệt mang ý nghĩa thống kê của lực hút nghịch biến giữa hai phân mảng.

---

## 3. Khám Phá Biểu Đồ Lớp (Analysis & Visualizations)

### 3.1. Sự Trỗi Dậy Của Attention Chống Lại MLP
Biểu đồ *Average M.I Profile* trình bày một khuynh hướng lôi cuốn:
- **Tầng Nông (Early Layers):** Cơ chế $MLP$ chứa M.I cao hơn so với $Attention$. Giai đoạn đầu, MLP bám sát vào định nghĩa thô của từ tĩnh, bảo toàn bộ nhớ về mặt khái niệm độc lập. Do đó các Token giống nhau "tương thông" thông tin rất lớn.
- **Tầng Sâu (Deep Layers):** Quỹ đạo $Attention$ đi lên tiệm cận trên, kéo mức trung bình chia sẻ M.I ngày một mạnh, trái ngược với $MLP$ rơi rớt cắm mỏ và đi ngang rập khuôn. Lý giải cơ học: Càng chìm sâu, Attention bị áp lực phải kết nối "ngữ cảnh vĩ mô". Để có thể đoán từ tiếp theo, nó phải lôi kéo lịch sử chồng chéo từ cực xa $\to$ nó chủ động làm giàu thông tin cho mọi liên kết cặp của chữ "coffee". 

### 3.2. Chênh Lệch Tương Quan Nghịch Biến (Kendall Correlation Stats)
Khuynh hướng khoảng cách xa sinh ra MI yếu luôn đạt biểu số Correlation Negative (Xoay quanh khoảng $-0.5$). Biểu đồ Z-value cho thấy sự phân ly rõ rệt: $Attention$ xử lý vấn đề token xa nhau mượt mà và linh động hơn nhiều so với hệ tĩnh tại $MLP$ sau Tầng thứ 10. 

### 3.3. So Sánh Thuật Toán Thủ Công Và Scikit-Learn
Thực hiện chạy toàn bộ hệ quy trình với nhân KDE Scikit-learn (Mất tầm khoảng 2 phút do Data Cặp nhỏ). So sánh trực quan đối chứng cho thấy: Các sai khác về đồ thị Laminar hoàn toàn mang tính chất tịnh tiến vô hại. Mọi tỷ lệ tương đối (Relative Values) giữa các không gian được bảo toàn tuyệt đối, gia cố thêm niềm tin rằng thuật toán tính Histogram MI Manual là giải pháp thay thế hoàn hảo cho tập dữ liệu Big Data.

---

## 4. Kết Luận
Bằng việc triển khai kiểm định độ lệch cực đỉnh và đo lường khoảng cách từ định hạng, tính năng Mutual Information là một trạm radar nhạy bén để bắt sóng cơ học lõi: $MLP$ đóng khuôn khái niệm ở tầng cao, còn $Attention$ đan kết mạng nhện vĩ mô dải dài tít tận đáy phễu.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh khảo sát tĩnh: `aero_LLM_12_CodeChallenge Attention to coffee MI and token distances (part 2).md` (Thiết lập hàm Fisher Z-Transform, Independent T-Test, Loop Laminar Analysis, so gánh đặc tính Attention - MLP).
