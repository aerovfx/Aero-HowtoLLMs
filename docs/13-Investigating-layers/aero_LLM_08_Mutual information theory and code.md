# Lý Thuyết Thông Tin: Đo Lường Entropy Và Mutual Information

## Tóm Tắt (Abstract)
Nghiên cứu này trình bày cách ứng dụng cốt lõi của **Lý thuyết thông tin (Information Theory)** vào các phép đo lường thống kê cho dữ liệu hoạt động của nơ-ron: Entropy và Mutual Information (Thông tin tương hỗ). Không giống như phương sai hay hệ số tương quan vốn mang bản chất tuyến tính và chỉ phù hợp với phân phối chuẩn, hai khái niệm toán học này giúp định lượng mức độ hỗn loạn (độ bất định) và khả năng quy nạp phi tuyến giữa hai biến số liên tục. Sự so sánh tính toán thủ công qua Histogram và thư viện Scikit-learn cũng được đề cập nhằm thiết lập tiền đề đoạt giải cơ học trên các ma trận nơ-ron sau này.

---

## 1. Khái Niệm Về Entropy Trong Lý Thuyết Thông Tin

Trái ngược với "Entropy nhiệt động lực học" tập trung vào sự hỗn loạn của hệ vật lý, **Entropy Shannon** mang màu sắc của "Sự bất ngờ" (Surprise) và "Khả năng dự đoán" (Predictability). 
- Một sự kiện có xác suất bằng $1$ (như mặt trời mọc vào ngày mai) không có sự bất ngờ $\to \text{Entropy} = 0$.
- Một sự kiện như tung đồng xu (xác suất $0.5$) rất khó dự đoán $\to \text{Entropy}$ đạt cực đại.

### 1.1. Công Thức Toán Học
Dành cho một biến biến thiên ngẫu nhiên (hoặc các đặc trưng categorical/continuous bins):
$$ H(X) = - \sum_{i=1}^{n} P(x_i) \log P(x_i) $$
Do $P(x_i) \in [0, 1]$ nên hệ số logarit sẽ mang dấu âm, dấu trừ phía ngoài giúp triệt tiêu và giữ giá trị Entropy $H$ luôn dương.

### 1.2. Xử Lý Các Trùng Lặp Số Học (Numerical Errors)
Do đặc thù logarit không xác định tại mốc 0, khi thực nghiệm phân vùng histogram trên một dữ liệu nơ-ron dày đặc, nhiều bin sẽ xuất hiện giá trị $P=0$. Để khắc phục, công thức code thực tế thêm cực trị tàn dư nhỏ (epsilon $\epsilon$) vào lõi tính:
$$ H(X) = - \sum P(X) \log(P(X) + \epsilon) $$
Nếu $P=0$, $\log(\epsilon) \times 0$ vẫn sẽ triệt tiêu trở về $0$, tránh sụp đổ vòng lặp hàm hàm log.

---

## 2. Đo Lường Sự Đồng Biến: Mutual Information (MI)

Nếu cho 2 biến $X$ và $Y$, **Mutual Information - $I(X;Y)$** là tỷ trọng mức độ thông tin bạn có thể luận ra từ biến kia, thông qua việc biết biến còn lại. Khác với "Covariance" (Hiệp phương sai), biến MI đặc biệt xuất sắc trong việc tóm tắt các khuynh hướng cấu trúc hình học hỗn hợp.

### 2.1. Tiếp Cận Bằng Biểu Đồ Venn (Entropy Giao Thoa)
Có thể đo lường MI bằng cách tính toán hàm lượng Entropy nguyên bản và Entropy hợp bộ (Joint-Entropy):
$$ I(X;Y) = H(X) + H(Y) - H(X,Y) $$
Nói cách khác, nó là phần "giao nhau" của giới hạn độ bất định giữa $X$ và $Y$. 

### 2.2. Tiếp Cận Bằng Phương Trình Phân Phối Cụ Thể
$$ I(X;Y) = \sum_{x \in X} \sum_{y \in Y} P(x,y) \log \left( \frac{P(x,y)}{P(x)P(y)} \right) $$

---

## 3. Thực Nghiệm Phương Pháp Luận Và Sai Số (Methodology in Praxis)

Dữ liệu đặc tính nơ-ron là các phân phối biến liên tục (continuous arrays), không phải các danh mục (discrete). Điều này tạo ra một rào cản đo lường khi ta buộc phải ép dữ liệu về các mặt lưới tần suất 2D (2D Histograms).

1. **Sai số do thủ công chia Histograms:**
   - Khi đo lường biến mảng $x$ và biến vô định $y$ không liên kết (Tức $I = 0$ tuyệt đối theo lý thuyết), việc gom nhóm dữ liệu thủ công vào 15 bins hoặc phân tách bằng phân vị (Percentiles) vẫn trả về kết quả ảo $(I \approx 0.4 \to 0.5)$. Kết quả Histogram đính kèm một lực lượng "sai lệch tĩnh" (constant bias).
2. **Khắc phục bằng Công Cụ Cốt Lõi (Scikit-Learn Regression):**
   - Thay vì đếm điểm số theo ô, phương pháp Non-parametric Kernel Density Estimators thuộc hàm thư viện `mutual_info_regression` của Sklearn cho phép định đoán chính xác nhất dải phân bổ xác suất, giúp đẩy Mutual Information trả về trân diện ở ngưỡng xấp xỉ $0.0$.
   - **Đánh đổi:** Hàm Sklearn chạy cực lỳ chậm. Do đó ở các kiến trúc LLMs phân giải hàng tỷ thông số, ta vẫn ưu tiên Histogram Method vì thực chất độ lệch Bias luôn đi ngang tự nhiên, không làm sai khác tính đối chiếu tỷ lệ.

---

## 4. Kết Luận
Bài viết diễn giải góc nhìn định lượng mới về không gian hoạt động hệ thống. Mutual Information bộc lộ sự hữu hiệu khi bỏ qua khái niệm tuyến tính định chuẩn và chỉ quan tâm duy nhất đến "vật chất giao thoa về tính bất định" của hệ. Kỹ thuật này sẽ là mảnh ghép tiền đề cho phép bóc tách cấu trúc luồng của Network mà không hề bận tâm tới biên độ khuếch đại (Scaling factor) của từng tầng chức năng.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh thí nghiệm liên kết: `aero_LLM_08_Mutual information theory and code.md` (Cách sử dụng Histogram 2D vs Scikit Learn; xây dựng công thức Shannon Entropy và Pairwise Mutual Information tĩnh).
