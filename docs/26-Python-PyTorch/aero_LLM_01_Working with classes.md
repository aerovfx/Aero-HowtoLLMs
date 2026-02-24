# Nhập môn PyTorch: Cơ sở về Lập trình Hướng đối tượng (Working with Classes)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về các nguyên lý cơ bản của Lập trình hướng đối tượng (Object-Oriented Programming - OOP) trong ngữ cảnh phát triển ứng dụng với thư viện PyTorch. Chúng ta phân tích hai khái niệm cốt lõi: Thuộc tính (Attribute) và Phương thức (Method), đồng thời minh chứng rằng các cấu trúc dữ liệu quen thuộc như Danh sách (List) thực chất là các lớp (classes) tiền định nghĩa trong Python. Nghiên cứu thực hiện thực nghiệm trên đối tượng danh sách để quan sát cơ chế tác động của các phương thức lên trạng thái của đối tượng, cung cấp nền tảng tư duy cần thiết để xây dựng các kiến trúc mạng nơ-ron tùy chỉnh sau này.

---

## 1. Hệ thuật ngữ trong Lập trình Hướng đối tượng

Dù OOP là một khái niệm lập trình tổng quát, nó đóng vai trò xương sống trong PyTorch khi mọi mô hình học máy đều được cấu trúc dưới dạng các lớp kế thừa.
- **Lớp (Class):** Một bản thiết kế hoặc khuôn mẫu định nghĩa các đặc tính chung.
- **Đối tượng/Thực thể (Object/Instance):** Một sản phẩm cụ thể được tạo ra từ bản thiết kế của lớp.
- **Thuộc tính (Attribute):** Các biến số gắn liền với đối tượng, đại diện cho trạng thái (ví dụ: các trọng số trong một lớp mạng).
- **Phương thức (Method):** Các hàm số gắn liền với đối tượng, đại diện cho hành vi (ví dụ: quá trình lan truyền tiến - forward pass).

---

## 2. Đối tượng Danh sách trong Python (Class List)

### 2.1. Bản chất của Danh sách
Trong Python, khi chúng ta khởi tạo một danh sách bằng lệnh `L = list([1, 2, 3])`, chúng ta thực chất đang tạo ra một thực thể của lớp `list`. 

### 2.2. Khám phá Thuộc tính bằng Hàm `dir()`
Hàm `dir(đối_tượng)` là công cụ quan trọng để liệt kê toàn bộ "kho vũ khí" mà một thực thể sở hữu. Kết quả trả về bao gồm các thuộc tính ẩn (bắt đầu bằng dấu gạch dưới `__`) và các phương thức công khai mà chúng ta có thể tương tác trực tiếp.

---

## 3. Tương tác với Phương thức qua Ký hiệu Dấu chấm

Toàn bộ quá trình tương tác với đối tượng dựa trên cú pháp `tên_đối_tượng.tên_phương_thức()`:
- **Phương thức không tham số:** `L.reverse()` làm đảo ngược thứ tự các phần tử trong danh sách hiện tại. Đây là hành động thay đổi trạng thái tại chỗ (in-place).
- **Phương thức có tham số:** `L.append(99)` yêu cầu một giá trị đầu vào để thêm vào cuối danh sách.
Điểm khác biệt quan trọng giữa hàm thông thường và phương thức là phương thức luôn có quyền truy cập mặc định vào dữ liệu bên trong đối tượng mà nó thuộc về.

---

## 4. Tầm quan trọng của OOP trong PyTorch
Việc nắm vững cách vận hành của các lớp là bước đệm để hiểu cấu trúc `nn.Module`. Mỗi lớp trong mạng nơ-ron (như Conv2d hay Linear) đều là một đối tượng chứa các thuộc tính (trọng số) và các phương thức (forward) để xử lý tensor. Nhà nghiên cứu AI cần thành thạo việc đọc và viết các lớp để có thể tùy biến các kiến trúc mô hình phức tạp một cách khoa học và có hệ thống.

---

## 5. Kết luận
Lập trình hướng đối tượng giúp đóng gói các thành phần phức tạp thành các thực thể dễ quản lý. Việc hiểu rõ mối quan hệ giữa thực thể, thuộc tính và phương thức thông qua các ví dụ cơ bản như danh sách giúp lập trình viên tự tin hơn khi tiếp cận các thư viện chuyên sâu như PyTorch, nơi tính bao gói (encapsulation) và sự kế thừa (inheritance) là những kỹ thuật tối ưu hóa hàng đầu.

---

## Tài liệu tham khảo (Citations)
1. Cơ chế hoạt động của lớp và đối tượng trong Python dựa trên `aero_LL_01_Working with classes.md`. Phân tích hệ thuật ngữ thuộc tính/phương thức, ứng dụng hàm dir() và thực nghiệm trên lớp list.
