# Nhập môn Python: Kỹ thuật Xây dựng Hàm (Creating Functions)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu quy trình thiết kế và triển khai các hàm tự định nghĩa (user-defined functions) trong Python. Chúng ta phân tích cấu trúc cú pháp nền tảng bắt đầu bằng từ khóa `def`, vai trò của thụt lề (indentation) trong việc xác định phạm vi khối mã, và cơ chế quản lý biến cục bộ (local scope). Nghiên cứu cũng đi sâu vào cách xử lý đa đầu vào và đa đầu ra thông qua kiểu dữ liệu Tuple, đồng thời phân biệt tính chất biến đổi (mutability) của Danh sách và tính bất biến (immutability) của Tuple. Đây là kiến thức then chốt để module hóa mã nguồn trong các dự án AI quy mô lớn.

---

## 1. Cấu trúc và Định nghĩa Hàm

### 1.1. Từ khóa `def` và Cú pháp Nền tảng
Để tạo một hàm mới, ta sử dụng từ khóa `def` (define), theo sau là tên hàm và cặp ngoặc đơn `()`.
- **Dấu hai chấm (`:`):** Đây là ký hiệu bắt buộc để đánh dấu điểm bắt đầu của thân hàm.
- **Thụt lề (Indentation):** Python sử dụng khoảng trắng (thường là 2 hoặc 4 dấu cách) để xác định các dòng mã thuộc về hàm. Mọi dòng mã không thụt lề sẽ được coi là nằm ngoài hàm.

### 1.2. Kích hoạt Hàm (Calling)
Một hàm đã viết nhưng chưa được thực thi (run cell) sẽ không được Python nhận diện. Khi gọi hàm mà thiếu cặp ngoặc đơn `()`, Python sẽ chỉ trả về thông tin định danh của hàm đó thay vì thực thi logic bên trong.

---

## 2. Truyền tham số và Cơ chế Đầu vào

Hàm có thể nhận vào các biến gọi là "tham số" (arguments).
- **Tính đa hình:** Do Python là ngôn ngữ định kiểu động, một hàm cộng (`add_two_numbers`) có thể hoạt động trên cả số (thực hiện phép cộng số học) và chuỗi ký tự (thực hiện phép nối chuỗi) tùy thuộc vào dữ liệu truyền vào.
- **Lỗi đối số:** Nếu hàm được định nghĩa không có tham số nhưng lại nhận vào dữ liệu khi gọi, Python sẽ báo lỗi `TypeError` về số lượng đối số vị trí.

---

## 3. Phạm vi Biến và Kết quả Đầu ra

### 3.1. Phạm vi Cục bộ (Local Scope)
Các biến được tạo ra bên trong hàm chỉ tồn tại trong suốt quá trình hàm thực thi. Khi hàm kết thúc, các biến này sẽ bị xóa khỏi bộ nhớ ("scope destroyed"). Điều này giúp ngăn chặn việc xung đột tên biến trong các hệ thống lớn.

### 3.2. Từ khóa `return` và Đa đầu ra
Để đưa kết quả tính toán ra bên ngoài hàm, ta sử dụng từ khóa `return`.
- **Trả về nhiều giá trị:** Python hỗ trợ trả về nhiều kết quả cùng lúc bằng cách ngăn cách chúng bằng dấu phẩy. Kết quả này sẽ được đóng gói thành một **Tuple**.

---

## 4. Phân tích Kiểu dữ liệu: Tuple vs List

Trong kết quả trả về đa giá trị, chúng ta bắt gặp kiểu dữ liệu Tuple:
- **Parentheses `()`:** Ký hiệu của Tuple (khác với ngoặc vuông `[]` của List).
- **Tính bất biến (Immutability):** Không giống như danh sách, phần tử của Tuple không thể bị thay đổi sau khi khởi tạo. Điều này đảm bảo tính an toàn cho dữ liệu đầu ra của hàm.
- **Giải nén (Unpacking):** Lập trình viên có thể gán trực tiếp kết quả của một hàm đa đầu ra vào các biến riêng biệt: `a, b = my_function()`.

---

## 5. Kết luận
Xây dựng hàm là kỹ năng chuyển đổi từ người sử dụng công cụ sang người tạo ra công cụ. Việc hiểu rõ về thụt lề, phạm vi biến và cơ chế `return` cho phép lập trình viên xây dựng những khối mã an toàn, dễ bảo trì và tối ưu hóa cho việc tái sử dụng trong các quy trình huấn luyện mô hình ngôn ngữ phức tạp.

---

## Tài liệu tham khảo (Citations)
1. Kỹ thuật định nghĩa và vận hành hàm trong Python dựa trên `aero_LLM_04_Creating functions.md`. Phân tích từ khóa `def`, thụt lề khối mã, phạm vi biến cục bộ và kiểu dữ liệu Tuple.
