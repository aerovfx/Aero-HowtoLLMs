# Học sâu: Thực thi Lan truyền ngược trong PyTorch

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về cách thức triển khai cơ chế lan truyền ngược (backpropagation) trong thư viện PyTorch thông qua một ví dụ tối ưu hóa hàm số cơ bản. chúng ta phân tích quy trình huấn luyện 5 bước tiêu chuẩn: từ việc làm sạch gradient (`zero_grad`), thực hiện lan truyền xuôi, tính toán mất mát, tính toán gradient ngược (`backward`) đến việc cập nhật trọng số (`step`). Nghiên cứu thực hiện thực nghiệm so sánh với lời giải giải tích (analytic solution) từ vi phân để chứng minh tính chính xác và hiệu quả của thuật toán Hạ giang trong việc tìm kiếm điểm cực tiểu của hàm mục tiêu, cung cấp nền tảng vững chắc cho việc huấn luyện các mô hình AI phức tạp với hàng tỷ tham số.

---

## 1. Quy trình Huấn luyện 5 Bước tiêu chuẩn

Trong PyTorch, việc huấn luyện mô hình được lặp lại qua các kỷ nguyên (epochs) theo trình tự nghiêm ngặt sau:

1. **`optimizer.zero_grad()`:** Xóa bỏ các gradient từ vòng lặp trước. Đây là bước bắt buộc vì PyTorch có cơ chế tích lũy gradient mặc định (có lợi cho việc huấn luyện các mô hình khổng lồ trên phần cứng hạn chế, nhưng cần được xóa sạch trong hầu hết các trường hợp thông thường).
2. **Forward Pass & Loss Calculation:** Đưa dữ liệu qua mô hình để nhận dự đoán $\hat{y}$ và so sánh với thực tế $y$ để tính toán hàm mất mát (loss).
3. **`loss.backward()`:** Tính toán đạo hàm (gradient) của hàm mất mát đối với tất cả các tham số có thuộc tính `requires_grad=True`.
4. **`optimizer.step()`:** Cập nhật các trọng số dựa trên các gradient vừa tính được và tốc độ học (learning rate).
5. **Monitoring (Tùy chọn):** Lưu trữ lịch sử mất mát hoặc in báo cáo tiến độ để theo dõi quá trình hội tụ.

---

## 2. Đối chứng với Giải tích

Nghiên cứu thực hiện tối ưu hóa hàm số $f(x) = 3x^2 - 2x + 3$ để minh chứng cơ chế học:
- **Lời giải giải tích:** Bằng cách tính đạo hàm $f'(x) = 6x - 2$ và đặt bằng 0, ta tìm được điểm cực tiểu chính xác tại $x = 1/3 \approx 0.333$.
- **Thực nghiệm PyTorch:** Sau 80 kỷ nguyên huấn luyện, mô hình khởi tạo tại $x = -1$ đã hội tụ về giá trị $\approx 0.32$.
- **Phân tích:** Dù không đạt đến con số tuyệt đối do các yếu tố như tốc độ học và số lượng vòng lặp, nhưng kết quả cho thấy mô hình đã di chuyển chuẩn xác về phía cực tiểu toàn cục của hàm số.

---

## 3. Vai trò của Gradient trong Đối tượng Tensor

Khi một Tensor được khởi tạo với `requires_grad=True`, nó không chỉ lưu trữ một con số mà còn là một cấu trúc dữ liệu phức tạp:
- **`w.grad`:** Lưu trữ giá trị đạo hàm hiện tại. Khi giá trị này tiến gần về 0, điều đó có nghĩa là mô hình đã ở rất gần điểm tối ưu.
- **Dòng chảy Gradient:** Nếu `w.grad` mang dấu âm, thuật toán sẽ đẩy trọng số sang bên phải (tăng giá trị) và ngược lại, đảm bảo mô hình luôn di chuyển ngược hướng với độ dốc của hàm mất mát.

---

## 4. Tầm quan trọng của Trực quan hóa

Việc theo dõi quỹ đạo mất mát (loss trajectory) là kỹ năng thiết yếu:
- Một đường cong đi xuống mượt mà và tiệm cận một giá trị ổn định là dấu hiệu của một quá trình huấn luyện thành công.
- Sự sai lệch nhỏ giữa kết quả tìm được và lời giải lý thuyết trong các mô hình đơn giản nhắc nhở nhà nghiên cứu về bản chất xấp xỉ của học sâu trong thực tế.

---

## 5. Kết luận
Triển khai lan truyền ngược trong PyTorch là sự kết hợp giữa sự tiện lợi của tự động hóa và sự chặt chẽ của giải tích. Dù các mô hình thực tế có thể phức tạp đến mức không thể trực quan hóa, nhưng cơ chế 5 bước và nguyên lý cập nhật trọng số vẫn giữ nguyên giá trị cốt lõi. Làm chủ quy trình này là điều kiện tiên quyết để xây dựng và tinh chỉnh các hệ thống học sâu hiệu quả, từ những bài toán hồi quy đơn giản đến những kiến trúc Transformer tiên tiến nhất hiện nay.

---

## Tài liệu tham khảo (Citations)
1. Quy trình 5 bước huấn luyện và phân tích gradient dựa trên `aero_LL_07_Backprop in Pytorch.md`. Thuyết minh về sự khác biệt giữa lời giải giải tích và xấp xỉ số trong tối ưu hóa mạng nơ-ron. village.
