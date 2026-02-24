# Toán học trong Học sâu: Hàm Logarit và Ứng dụng trong Tối ưu hóa (Logarithms)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về hàm logarit, một công cụ toán học không thể thiếu trong lĩnh vực tối ưu hóa và học sâu. chúng ta phân tích mối quan hệ nghịch đảo giữa logarit tự nhiên và hàm số mũ $e^x$, tính chất đơn điệu (monotonicity) giúp duy trì thứ tự cực trị của hàm mục tiêu, và khả năng "giãn cách" (stretching) các giá trị cực nhỏ về phía âm vô cùng. Nghiên cứu nhấn mạnh rằng việc chuyển đổi xác suất sang không gian logarit không chỉ cải thiện độ chính xác số học (numerical precision) mà còn giúp các phép toán đạo hàm trở nên ổn định hơn trong quá trình huấn luyện mạng nơ-ron.

---

## 1. Bản chất và Mối quan hệ Nghịch đảo

Hàm logarit tự nhiên ($\ln$ hoặc $\log$) là phép toán ngược của hàm số mũ tự nhiên:
- **Tính triệt tiêu:** $\log(e^x) = x$. Khả năng này cực kỳ hữu ích trong việc "mở khóa" các tham số nằm trong số mũ của các hàm kích hoạt như Softmax hay Sigmoid.
- **Đồ thị:** Ngược lại với $e^x$ tăng trưởng bùng nổ, logarit tăng trưởng rất chậm và chỉ xác định với giá trị dương ($x > 0$). Khi $x$ tiến dần về 0, logarit tiến về âm vô cùng ($-\infty$).

---

## 2. Tính đơn điệu và Ý nghĩa trong Tối ưu hóa

Một hàm số được gọi là **đơn điệu** (monotonic) nếu thứ tự của các giá trị đầu vào được bảo toàn ở đầu ra:
- **Nguyên lý Tối ưu:** Vì logarit là hàm đồng biến (monotonic increasing), nên việc cực tiểu hóa một giá trị $x$ cũng tương đương với việc cực tiểu hóa $\log(x)$.
- **Hệ quả:** Trong học sâu, thay vì trực tiếp tối ưu hóa xác suất (thường là các số rất nhỏ), chúng ta tối ưu hóa giá trị logarit của xác suất đó. Điều này giúp mô hình tìm ra các tham số tối ưu mà không làm thay đổi bản chất của bài toán gốc nhưng lại có lợi thế về mặt tính toán.

---

## 3. Khả năng Giãn cách và Độ chính xác Số học

Một trong những thách thức lớn nhất của máy tính là xử lý các số thực cực nhỏ (ví dụ: $0.0000000001$).
- **Vấn đề số học:** Các số rất nhỏ nằm sát nhau khiến máy tính khó phân biệt và dễ gây ra lỗi làm tròn (numerical precision errors).
- **Giải pháp Logarit:** Hàm logarit "kéo giãn" khoảng cách giữa các số nhỏ này trên trục tung. Khoảng cách giữa các giá trị gần 0 trong không gian logarit lớn hơn rất nhiều so với không gian tuyến tính, giúp thuật toán tối ưu hóa "nhìn thấy" các thay đổi nhỏ nhất của mô hình.

---

## 4. Thực thi trong Môi trường Python

Việc tính toán logarit trong Python rất đơn giản thông qua thư viện NumPy:
- **Hàm `np.log()`:** Mặc định tính logarit tự nhiên (cơ số $e$).
- **Ứng dụng thực tế:** Trong các hàm mất mát như Cross-Entropy, logarit được dùng để chuyển đổi các tích xác suất phức tạp thành các tổng đơn giản, từ đó tăng tốc độ tính toán đạo hàm và ổn định quá trình lan truyền ngược (backpropagation).

---

## 5. Kết luận
Hàm logarit không đơn thuần là một phép tính lớp 12 mà là "kính hiển vi" của nhà nghiên cứu AI. Nó cho phép chúng ta làm việc trong một không gian số học ổn định, bảo toàn được các thuộc tính quan trọng của dữ liệu trong khi loại bỏ các rào cản về độ chính xác máy tính. Việc thấu hiểu logarit là chìa khóa để bước vào thế giới của Entropy và Cross-Entropy – những thước đo nòng cốt xác định sự thành bại của một mô hình học sâu.

---

## Tài liệu tham khảo (Citations)
1. Vai trò của hàm logarit trong tối ưu hóa và học sâu dựa trên `aero_LL_07_Logarithms.md`. Phân tích quan hệ nghịch đảo với $e^x$, tính đơn điệu và lợi thế về độ chính xác số học đối với các giá trị nhỏ.
