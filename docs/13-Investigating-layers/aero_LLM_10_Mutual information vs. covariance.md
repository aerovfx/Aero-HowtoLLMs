# Phân Tích Đối Chiếu Đo Lường Tương Quan: Mutual Information và Covariance

## Tóm tắt (Abstract)
Báo cáo này tập hợp và so sánh trực tiếp hai giải pháp toán học đo lường sự phụ thuộc thống kê (statistical dependencies) phổ biến đối với tín hiệu lưới (Network Activations): **Thông Tin Tương Hỗ (Mutual Information - MI)** và **Hiệp Phương Sai (Covariance)**. Thông qua kiểm thử dữ liệu mô phỏng tuyến tính và phi tuyến tính (simulated linear/non-linear data), nghiên cứu phơi bày ưu/nhược điểm cốt lõi của từng phương pháp. Đặc biệt, phân tích làm rõ bản chất "không dấu" (unsigned) của MI so với tính chất phân cực (signed) của Covariance, cũng như tốc độ tính toán và độ nhạy cảm của chúng đối với không gian đo kiểm.

---

## 1. Mở Đầu (Introduction)
Covariance và Mutual Information đều nhằm trả lời một câu hỏi cơ bản: *"Khi tôi biết hoạt động của Tín hiệu X, tôi có thể dự đoán được phần nào hành vi của Tín hiệu Y hay không?"*
Dù có chung mục đích, cách chúng đánh giá dữ liệu lại nằm ở hai hệ quy chiếu khác biệt. Việc hiểu rõ ranh giới toán học của hai công cụ này đóng vai trò quan trọng trước khi áp dụng chúng để bóc tách hành vi của Mạng Neural.

---

## 2. Lý Thuyết Phương Pháp Cốt Lõi (Core Methodologies)

### 2.1. Hiệp Phương Sai (Covariance)
Covariance là một đo lường "tuyến tính" thuần tuý và được lấy trực tiếp trên giá trị tuyệt đối của dữ liệu.
Đối với 2 biến trung tâm hóa (mean-centered) X và Y:
$$ Cov(X,Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y}) $$
**Ưu điểm:**
- Nhanh, mạnh và cực kỳ ổn định về mặt số học.
- Bảo tồn tỷ lệ (scale) của dữ liệu (Ví dụ: dữ liệu đơn vị "mét" thì covariance đơn vị "mét vuông"). Tính chất này đặc biệt hữu dụng với các bài toán truy vết biên độ (Magnitude Tracking).
- Định dạng có dấu (Signed): Nó báo cho bạn biết X và Y là đi lên cùng nhau (Dấu +) hay nghịch biến (Dấu -). 

### 2.2. Thông Tin Tương Hỗ (Mutual Information - MI)
MI không lấy theo số liệu gốc mà phân rã dữ liệu vào ma trận Histogram trước, sau đó tính toán trên không gian xác suất (probability distribution).
$$ I(X;Y) = \sum_{x} \sum_{y} P(x,y) \log \left( \frac{P(x,y)}{P(x)P(y)} \right) $$
**Ưu điểm:**
- Đoán nhận được cả cấu trúc tương quan tuyến tính lẫn phi tuyến tính (đường cong).
- Giải phóng khỏi rào cản tỷ lệ (scale-independence). MI của 1 triệu hay 1 tỷ cũng không làm thay đổi giá trị thông tin nền tảng.

---

## 3. Khám Phá Qua Dữ Liệu Mô Phỏng (Analysis & Results)

### 3.1. Phản Ứng Với Dữ Liệu Phi Tuyến (Non-linear Dependency)
Khi sinh một mảng dữ liệu có mô hình parabol (Dạng phễu) hoặc hàm sóng (Cosine) thì kết quả bộc lộ cực kỳ rõ ràng ranh giới của 2 kỹ thuật:
- **Covariance** mù lòa trước các đường cong gập và bị triệt tiêu, và trả về kêt quả xấp xỉ $0$.
- **Mutual Information** ngay lập tức nhận ra trật tự ẩn này và sinh ra lượng thông tin tích cực cao $> 0$.

### 3.2. Tính Phân Cực Và Tính Không Dấu (Signed vs Unsigned metrics)
Đối với biến mô phỏng tuyến tính, khi quét $Covariance$ theo một dải biến thiên đảo cực từ $+0.9$ (Thuận nghịch) xuống thẳng $-0.9$ (Trái nghịch):
- Biểu đồ $Covariance$ tuân thủ mô hình đường thẳng tịnh tiến âm dương hoàn hảo.
- Biểu đồ $MI$ bẻ phễu thành chữ U. Lý do là MI là chỉ số **Unsigned (không dấu)**. Nó chỉ quan tâm đến sức mạnh của thông tin dự đoán. Dù đường truyền nghịch biến (-0.9) hay đồng biến (0.9), "mức độ gợi ý thông tin cho MI" là như nhau và đều $\to \text{Max}$. Để làm cho Covariance có đồ hình tương quan như MI, chỉ cần bình phương Covariance (Squared Covariance).

---

## 4. Kết Luận
Covariance và Mutual Information là hai con dao phân tích của Data Science, mỗi loại sở hữu một công năng chế tác riêng biệt:
- Nếu mạng LLM đang được đánh giá là một khối phân phối giả định (normally distributed data) chứa các tính toán ma trận ma sát tuyến tính đơn thuần $\to$ **Covariance** là giải pháp tối ưu vì sự bền bỉ, dễ diễn dịch âm/dương, và tính toán tức thời.
- Nếu bạn cần nhặt ra các bí mật ẩn giấu dưới dạng quan hệ cấu trúc phức tạp, bất chấp khoảng đo đạc vô cùng hẹp hoặc nhiễu loạn $\to$ **Mutual Information** sẽ là chiếc la bàn đo lường độ bất định thông tin sâu sắc. Tuy nhiên cần hết sức cảnh giác với sai số do hiện tượng lấy mẫu dưới số lượng (Undersampling probabilities).

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh so sánh tĩnh: `aero_LLM_10_Mutual information vs. covariance.md` (Hướng dẫn lập các biến x và y2 (hàm mũ hai, hàm sóng cos) nhằm hiển thị đồ thị chữ U so gánh Covariance signed và MI unsigned metrics).
