# Nhập môn Python: Kỹ thuật Nội suy Chuỗi và F-strings (String Interpolation)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về các phương pháp xử lý và hiển thị thông tin văn bản trong Python, trọng tâm là kỹ thuật nội suy chuỗi (string interpolation). Chúng ta phân tích sự tiến hóa từ phương pháp nối chuỗi (concatenation) truyền thống, qua nội suy bằng toán tử `%`, đến giải pháp hiện đại F-strings. Nghiên cứu đi sâu vào khả năng định dạng số thực dấu phẩy động, quản lý khoảng trắng và canh lề dữ liệu trong các báo cáo lặp. Đây là kỹ năng tối quan trọng trong việc xây dựng các bản tin trạng thái (status updates) và nhật ký huấn luyện (training logs) cho các mô hình ngôn ngữ lớn.

---

## 1. Sự hạn chế của Phép nối chuỗi truyền thống
Phương pháp sử dụng dấu cộng (`+`) để kết hợp văn bản và số liệu thường gây ra hai vấn đề chính:
- **Cưỡng ép kiểu dữ liệu:** Bắt buộc phải sử dụng hàm `str()` để chuyển đổi số thành văn bản trước khi nối.
- **Lỗi cú pháp và Thẩm mỹ:** Dễ bỏ sót các khoảng trắng phân cách và tạo ra mã nguồn rườm rà, khó bảo trì.

---

## 2. Kỹ thuật Nội suy Cổ điển (Toán tử %)
String Interpolation cho phép chèn các biến vào đúng vị trí trong một chuỗi định sẵn bằng cách sử dụng các ký hiệu giữ chỗ:
- **`%g`:** Đại diện cho một số thực hoặc số nguyên.
- **`%s`:** Đại diện cho một chuỗi ký tự.
- **Cấu trúc:** `"Văn bản %g" % biến`.
Mặc dù sạch sẽ hơn phép nối chuỗi, phương pháp này vẫn còn hạn chế về tính linh hoạt khi xử lý các biểu thức toán học phức tạp.

---

## 3. Cuộc cách mạng F-strings (Formatted Strings)

### 3.1. Cú pháp và Ưu điểm
F-strings (xuất hiện từ Python 3.6) được kích hoạt bằng tiền tố `f` trước dấu ngoặc kép: `f"Văn bản {biến}"`.
- **Tính trực quan:** Biến được đặt trực tiếp trong dấu ngoặc nhọn `{}` ngay tại vị trí hiển thị.
- **Hiệu năng:** Cho phép thực hiện các phép toán trực tiếp bên trong chuỗi (ví dụ: `{i**4}`).

### 3.2. Kiểm soát Định dạng đầu ra
F-strings cung cấp các công cụ mạnh mẽ để làm đẹp dữ liệu thông qua dấu hai chấm `:` sau tên biến:
- **Độ chính xác thập phân:** `:.2f` ép buộc hiển thị đúng 2 chữ số sau dấu phẩy, giúp các bảng số liệu trở nên đồng nhất.
- **Độ rộng hiển thị (Padding):** `:6` chỉ định tổng số ký tự tối thiểu cho dữ liệu. Điều này cực kỳ hữu ích để căn thẳng hàng các con số có số chữ số khác nhau (ví dụ: số 9 và số 100) trong danh sách in.

---

## 4. Thực nghiệm Canh lề Dữ liệu
Trong các vòng lặp tạo báo cáo, việc không định dạng sẽ khiến dữ liệu bị xô lệch do sự khác biệt về độ dài chữ số. Nghiên cứu chỉ ra rằng việc kết hợp đồng thời độ rộng (`width`) và độ chính xác (`precision`), ví dụ `:{width}.{precision}f`, là chìa khóa để tạo ra những báo cáo nhật ký (logs) chuyên nghiệp, dễ đọc bằng mắt thường.

---

## 5. Kết luận
Làm chủ F-strings là một bước tiến quan trọng trong việc nâng cao thẩm mỹ mã nguồn Python. Trong nghiên cứu LLM, nơi các thông số như tỷ lệ mất mát (loss) và độ chính xác (accuracy) cần được theo dõi liên tục, khả năng định dạng chuỗi linh hoạt giúp nhà nghiên cứu nhanh chóng nắm bắt xu hướng vận hành của mô hình mà không bị phân tâm bởi sự lộn xộn của dữ liệu thô.

---

## Tài liệu tham khảo (Citations)
1. Kỹ thuật định dạng chuỗi ký tự và F-strings trong Python dựa trên `aero_LL_01_String interpolation and f-strings.md`. Phân tích so sánh giữa phép nối chuỗi, nội suy toán tử % và các tùy chọn canh lề định dạng.
