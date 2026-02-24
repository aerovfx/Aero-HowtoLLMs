# Nhập môn PyTorch: Kỹ thuật Xây dựng Lớp tùy chỉnh (Creating Custom Classes)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu quy trình thiết kế mã nguồn cho các lớp (classes) tùy chỉnh trong Python, ứng dụng vào việc mô phỏng cấu trúc của một mô hình học sâu. Chúng ta phân tích cơ chế khởi tạo đối tượng thông qua phương thức constructor `__init__`, vai trò của từ khóa `self` trong việc quản lý bộ nhớ cục bộ, và cách thức định nghĩa các phương thức hành vi để tương tác với thuộc tính bên trong. Nghiên cứu cũng khảo sát các "phương thức ma thuật" (magic methods) như `__str__` để chuẩn hóa hiển thị thông tin. Đây là kỹ năng nền tảng để nhà nghiên cứu hiện thực hóa các kiến trúc mạng nơ-ron phức tạp như Transformer hay GPT.

---

## 1. Khởi tạo Cấu trúc: Phương thức Constructor (`__init__`)

Mọi lớp tùy chỉnh đều cần một xuất phát điểm để xác định các thông số ban đầu.
- **Cú pháp:** `def __init__(self, ...):` là phương thức đặc biệt tự động thực thi khi một thực thể (instance) được tạo ra.
- **Từ khóa `self`:** Đóng vai trò là tham chiếu đến chính đối tượng đang được khởi tạo. Thông qua `self`, Python phân biệt được các thuộc tính của các đối tượng khác nhau được tạo ra từ cùng một lớp.
- **Thiết lập Thuộc tính:** Việc gán `self.weights = 10` đảm bảo rằng mỗi mô hình khi sinh ra đều có một trạng thái nội bộ riêng biệt, không bị lẫn lộn giữa các phiên bản.

---

## 2. Định nghĩa Hành vi thông qua Phương thức Tùy chỉnh

Các phương thức bên trong lớp cho phép thực hiện các phép toán trên chính dữ liệu của lớp đó:
- **Truy xuất Thông tin:** Phương thức `how_many_units` minh chứng khả năng tính toán dựa trên các thuộc tính sẵn có (`layers` và `units`) mà không cần nhận tham số ngoài.
- **Tương tác và Cập nhật:** Phương thức `train_the_model(self, x)` mô phỏng quá trình huấn luyện bằng cách nhận dữ liệu ngoài (`x`) và cập nhật trực tiếp vào thuộc tính `weights`. Điều này minh họa cho tính bao gói (encapsulation) – nơi logic xử lý dữ liệu nằm ngay bên cạnh dữ liệu.

---

## 3. Chế độ Hiển thị và Phương thức Ma thuật (Magic Methods)

Để đối tượng có khả năng giao tiếp thân thiện với con người, chúng ta sử dụng `__str__`.
- **Cơ chế:** Khi gọi hàm `print(đối_tượng)` hoặc `str(đối_tượng)`, Python sẽ tìm kiếm phương thức `__str__` để trả về một chuỗi văn bản đại diện. 
- **Ứng dụng:** Giúp nhà nghiên cứu nhanh chóng nhận diện tên mô hình và các thông số cấu hình chính mà không cần truy cập thủ công vào từng thuộc tính.

---

## 4. Chu kỳ Sống và Tính Nhất quán của Đối tượng

Thực nghiệm cho thấy:
- **Tính lũy kế:** Gọi phương thức huấn luyện nhiều lần sẽ làm thay đổi trạng thái thuộc tính một cách liên kết (trọng số tăng dần).
- **Khởi tạo lại (Reset):** Việc gọi lại lệnh khởi tạo `x = Model(...)` sẽ xóa bỏ toàn bộ lịch sử huấn luyện cũ và đưa các tham số về trạng thái mặc định trong `__init__`. Đây là cơ chế quan trọng để đảm bảo tính lặp lại (reproducibility) trong các thí nghiệm học máy.

---

## 5. Kết luận
Xây dựng lớp tùy chỉnh là đỉnh cao của việc tổ chức mã nguồn trong PyTorch. Trong học sâu, mỗi lớp mạng nơ-ron không chỉ là một mảng số liệu mà là một thực thể sống động với các thuộc tính (ma trận trọng số, tham số điều chuẩn) và các phương thức (forward pass, back propagation). Việc làm chủ tư duy hướng đối tượng cho phép nhà nghiên cứu xây dựng các hệ thống AI linh hoạt, dễ mở rộng và mang tính module hóa cao.

---

## Tài liệu tham khảo (Citations)
1. Quy trình xây dựng lớp tùy chỉnh và quản lý thuộc tính đối tượng dựa trên `aero_LL_02_Creating custom classes.md`. Phân tích phương thức init, từ khóa self, vai trò của __str__ và ứng dụng trong đào tạo mô hình.
