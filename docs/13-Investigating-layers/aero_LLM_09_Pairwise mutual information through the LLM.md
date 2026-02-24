# Phân Tích Thông Tin Tương Hỗ Dọc Theo Các Tầng Của Mô Hình Ngôn Ngữ (Pairwise Mutual Information Through LLMs)

## Tóm tắt (Abstract)
Tiếp nối lý thuyết cơ bản về Information Theory, bài viết này trình bày quy trình tính toán và trực quan hóa **Thông Tin Tương Hỗ Chéo (Pairwise Mutual Information)** trên bộ tham số hoạt động thực tế của LLM (cụ thể là GPT-2). Bằng cách sử dụng đoạn văn bản đầu vào lấy từ Wikipedia (chủ đề Cà phê Thổ Nhĩ Kỳ), chúng ta duyệt toàn bộ token dọc theo các tọa độ $Hidden\ States$ (768 chiều không gian). Kết quả thảo luận về sự bù trừ giữa sức mạnh tính toán và độ mượt thống kê (Scikit-Learn vs. Thuật toán thủ công), đồng thời cung cấp cách diễn giải biểu đồ phân bổ Mutual Information (MI) khi hệ thống chìm dần xuống độ sâu của kiến trúc.

---

## 1. Mở Đầu (Introduction)
Mutual Information (MI) là một giải pháp toán học sắc bén định lượng độ chia sẻ thông tin phi tuyến không thể bị bắt lỗi bởi quy mô biến thiên. Tuy nhiên, cách bạn nhóm dữ liệu (tokens data array) đóng vai trò quyết định đến cái nhìn vi mô vào bên trong cỗ máy:
- **Hướng 1 (Sử dụng ở mục này):** Tính toán MI dọc qua toàn bộ Tokens để khảo sát độ liên quan giữa *Từng cặp Không gian nơ-ron (Pairs of Hidden dimensions)* - Liệu chiều không gian X của dữ liệu có cung cấp thông tin gì về chiều không gian Y không? 
- **Hướng 2 (Dùng cho các Code Challenge sau):** Tính toán MI dọc qua các chiều ẩn để so sánh độ tương hỗ giữa *Từng cặp Token với nhau (Pairs of Tokens)*. 

Mục tiêu chính trong phần này là thử nghiệm áp dụng Hướng 1 và phân tích bài toán giới hạn tính toán siêu quy mô (Computation Limits) của Mutual Information.

---

## 2. Phương Pháp Luận Và Triển Khai (Methodology)

### 2.1. Nạp Hàm Kích Hoạt GPT-2 
Tương tự những phần trước:
- **Ví dụ đầu vào:** 1 đoạn text 94 tokens về "Turkish coffee". 
- **Lấy biến:** Tập Tensor $Hidden\ States$ cỡ $94\ (\text{tokens}) \times 768\ (\text{dimensions})$.  
Tại tầng ẩn số 3 (Layer 3), nếu ta bốc chiều không gian thứ X và trục chiều Y, sau đó biểu diễn 94 toạ độ token lên một Scatter Plot, hiện tượng nhìn thấy thường là **đám mây điểm vô cực (isotropic cloud)**. Đo MI của một đám sương mù này sẽ cho kết quả tịnh tiến bằng $0$.  

### 2.2. Vòng Lặp Phân Tích Ma Trận Cặp (Pairwise Matrix)
Với mỗi 1 trạm Block Transformer, có đến 768 chiều dữ liệu, việc lập ma trận Tương quan thông tin yêu cầu tạo hệ toạ độ $768 \times 768$. Ma trận kết xuất có tính chất Đối xứng chéo (Symmetry: $MI_{x,y} = MI_{y,x}$), vì thế để tiết kiệm một nửa công suất quét, ta sử dụng vòng lặp tịnh tiến chéo (từ giá trị $i+1$).

---

## 3. Khám Phá Rào Cản Lượng Tử MI (Analysis Results)

### 3.1. Sự Tắt Nghẽn Khi Sử Dụng Thư Viện Tích Hợp (Scikit-Learn Limitations)
Bộ ước tính Kernel Density Estimator ($KDE$) của thư viện `sklearn.feature_selection.mutual_info_regression` rất ưu việt trong việc khử nhiễu sai số khi hàm đếm gặp giá trị xác suất không. Nhưng khi đặt vào tổ hợp vòng lặp ma trận cặp hàng chục ngàn bước ($768 \times 768$), tốc độ là một thảm họa (nó có thể ngốn vài giờ đồng hồ cho một layer cỏn con). 

### 3.2. Hiệu Ứng Dịch Chuyển (Global Shift Bias) Của Thuật Toán Thủ Công
Chạy đoạn code MI Manual từ File 08 chỉ mất khoảng 4-6 phút. Tuy nhiên ta thấy:
- **Khoảng lệch tĩnh (Constant offset bias):** Thay vì đỉnh đồ thị ở tiệm cận $0.0$, điểm bắt đầu Histogram nhô lên ở ngưỡng xấp xỉ $\approx 1.0$.
- **Lý giải:** Hiện tượng undersampling khoảng phân bổ xác suất và lỗi xuất hiện nhiều Bin bằng 0 khiến các biến MI tự đẩy số điểm của toàn hệ lên.
Tuy nhiên, tin tốt là sự chênh lệch này là *Tịnh tiến toàn khối (Global shift)*. Toàn bộ các giá trị MI tương quan với nhau (Relative values) vẫn hoàn toàn được bảo lưu chính xác. Đối với các biểu đồ đối sánh LLMs, giá trị tuyệt đối không cần thiết bằng tính "Tương đối". 

### 3.3. Biến Động Qua Độ Sâu Mô Hình
Khi tiến hành ghép kết ma trận MI qua toàn bộ 13 layer (1 gốc nhúng + 12 transformer blocks). Kết quả làm chúng ta chú ý:
- Layer Embedded khởi điểm (Input layer): Đáy phổ sóng Mutual Information nhọn và dạt sang một bên. 
- 12 Transformer Layers còn lại: Nhảy múa chung ở một đồ thị chập gần như y hệt.
Do cách chúng ta cắt mạng dựa trên các Token theo chiều ngang để tìm liên kết kích hoạt, độ biến thiên Mutual Information theo chiều sâu gần như tĩnh. Lẽ dĩ nhiên, hiện tượng này sẽ khác hẳn nếu ta cắt theo khía ngang để khảo sát từng Pair of Tokens. 

---

## 4. Kết Luận (Conclusion)
Đo lường MI dọc thông qua một LLM cung cấp 3 nền tảng đúc kết: 
1. MI cực kỳ quyền năng và linh hoạt cho nhiều kiểu bóc tách (Theo Dimension hoặc theo Tokens). 
2. Luôn thận trọng diễn giải Giá trị Tương quan (Relative interpretations) khi dùng giải pháp phân nhóm Histogram tay, không tập trung vào giá trị tuyệt đối. 
3. Liên kết không gian ngang (Across Dimensions) không phải là cấu trúc duy nhất để tìm hiểu cách thức mô hình tổng hợp ngôn ngữ ngữ nghĩa. Kỹ thuật đào sâu sự tương đồng cặp Tokens mới là chìa khoá cho cách giải thích văn cảnh ngôn ngữ học.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh khảo sát tĩnh: `aero_LLM_09_Pairwise mutual information through the LLM.md` (Hướng dẫn duyệt vòng lặp matrix, tính chất đối xứng $MI(x,y) = MI(y,x)$ và xử lý bias tính toán).
