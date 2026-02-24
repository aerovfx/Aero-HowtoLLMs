# Nhập môn Python: Vòng lặp For và Kỹ thuật Lặp (For Loops)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về vòng lặp `for`, một công cụ điều khiển luồng (flow control) quan trọng để tự động hóa các tác vụ lặp đi lặp lại trong Python. Chúng ta phân tích cấu trúc cú pháp của vòng lặp, khái niệm về đối tượng có thể lặp (iterable), và cơ chế thay đổi giá trị của biến chỉ mục (looping variable) qua từng lần lặp. Nghiên cứu cũng thực hiện các thực nghiệm về "mã hóa mềm" (soft coding) để tăng tính linh hoạt cho chương trình, đồng thời triển khai một thuật toán phức tạp hơn là dãy số Fibonacci để minh chứng cho việc kết hợp vòng lặp bên trong các hàm tự định nghĩa.

---

## 1. Cấu trúc và Cơ cấu Vận hành của Vòng lặp For

### 1.1. Cú pháp cơ bản
Vòng lặp `for` trong Python được thiết lập theo cấu trúc:
`for biến in đối_tượng_lặp:`
- **Biến lặp:** Là biến thay đổi giá trị sau mỗi chu kỳ (iteration). Nó có thể là số, chuỗi ký tự hoặc bất kỳ phần tử nào trích xuất từ tập dữ liệu.
- **Đối tượng lặp (Iterable):** Là các cấu trúc dữ liệu có thể duyệt qua từng phần tử một như `range()`, danh sách (list), mảng (array) hoặc chuỗi (string).
- **Phạm vi (Scope):** Tương tự như hàm, thân của vòng lặp được xác định bằng khoảng trắng thụt lề (thường là 2 hoặc 4 dấu cách).

### 1.2. Quy tắc đếm và Cận biên
Khi sử dụng `range(start, stop)`, vòng lặp sẽ bắt đầu từ giá trị `start` (bao hàm) và kết thúc ngay trước giá trị `stop` (loại trừ). Đây là một quy tắc nhất quán trong Python giúp lập trình viên quản lý chính xác số lần thực thi mã nguồn.

---

## 2. Kỹ thuật Lập trình Nâng cao với Vòng lặp

### 2.1. Mã hóa mềm (Soft Coding)
Thay vì sử dụng các con số cứng (hard-coding) trong vòng lặp, chúng ta sử dụng các hàm như `len()` để xác định số lần lặp dựa trên độ dài thực tế của dữ liệu.
- **Lợi ích:** Khi kích thước tập dữ liệu thay đổi, vòng lặp sẽ tự động thích ứng mà không cần sửa đổi mã nguồn thủ công. Điều này cực kỳ quan trọng trong việc xử lý các lô dữ liệu (batches) có kích thước khác nhau trong huấn luyện mô hình.

### 2.2. Theo dõi Quá trình (Logging)
Vòng lặp thường được sử dụng để in ra các thông báo trạng thái. Bằng cách kết hợp phép nối chuỗi (`+`) và hàm ép kiểu `str()`, lập trình viên có thể tạo ra các dòng báo cáo chi tiết về giá trị hiện tại của biến tại mỗi bước lặp.

---

## 3. Thực nghiệm Thuật toán: Dãy số Fibonacci
Để minh chứng cho sức mạnh của vòng lặp khi kết hợp với hàm và mảng động, chúng ta triển khai thuật toán tạo dãy Fibonacci (trong đó mỗi số là tổng của hai số đứng trước nó).
- **Phương thức `.append()`:** Được sử dụng để mở rộng danh sách một cách động bên trong vòng lặp.
- **Tích hợp trong hàm:** Việc đóng gói toàn bộ logic vòng lặp vào một hàm (ví dụ: `fib_seq(n)`) cho phép tạo ra các dãy số có độ dài bất kỳ chỉ với một câu lệnh đơn giản.

---

## 4. Các lỗi thường gặp và Lưu ý
- **Lỗi nhập thư viện:** Khi sử dụng các hàm từ thư viện bên ngoài (như `np.linspace`) bên trong vòng lặp, cần đảm bảo thư viện đó đã được nạp ở đầu tệp mã nguồn.
- **Tính nhất quán của thụt lề:** Việc trộn lẫn giữa 2 và 4 khoảng trắng trong cùng một khối lặp sẽ gây ra lỗi cú pháp.

---

## 5. Kết luận
Vòng lặp `for` là nền tảng của mọi thuật toán huấn luyện và suy luận trong AI. Từ việc duyệt qua từng lớp của mạng nơ-ron đến việc xử lý từng token trong một văn bản, khả năng kiểm soát luồng lặp hiệu quả giúp chuyển đổi các phép toán đơn lẻ thành các quy trình tự động hóa mạnh mẽ và linh hoạt.

---

## Tài liệu tham khảo (Citations)
1. Cấu trúc và ứng dụng của vòng lặp For trong Python dựa trên `aero_LLM_01_For loops.md`. Phân tích khái niệm iterable, kỹ thuật soft coding và triển khai dãy số Fibonacci.
