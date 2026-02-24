# Nhập môn Python: Từ điển và Cấu trúc Cặp Khóa-Giá trị (Dictionaries)

## Tóm tắt (Abstract)
Báo cáo này giới thiệu về "Từ điển" (Dictionary), một kiểu dữ liệu mạnh mẽ trong Python chuyên dùng để tổ chức thông tin theo ánh xạ cặp khóa-giá trị (key-value pairs). Khác với danh sách (list) sử dụng chỉ mục số, từ điển hoạt động tương tự như một bảng tra cứu, cho phép truy xuất dữ liệu thông qua các nhãn định danh duy nhất. Nghiên cứu thực hiện phân tích các phương thức khởi tạo, khả năng lưu trữ không đồng nhất (mixing types) và các phương thức truy vấn tập hợp như `.keys()`, `.values()` và `.items()`. Đây là cấu trúc dữ liệu nền tảng để quản lý các siêu tham số (hyperparameters) và kết quả hoạt hóa trong nghiên cứu mô hình ngôn ngữ.

---

## 1. Bản chất và Khởi tạo Từ điển

### 1.1. Khái niệm Bảng tra cứu (Lookup Table)
Từ điển là một tập hợp các mục dữ liệu, trong đó mỗi giá trị (value) được liên kết với một khóa (key) cụ thể. Để lấy được thông tin ("mở khóa"), lập trình viên cần cung cấp đúng từ khóa tương ứng.

### 1.2. Cấu trúc Cú pháp
- **Ký hiệu:** Sử dụng dấu ngoặc nhọn `{}`. (Lưu ý: ngoặc vuông `[]` dành cho danh sách, ngoặc nhọn `{}` dành cho từ điển).
- **Khởi tạo:** Có thể bắt đầu bằng một từ điển rỗng thông qua hàm `dict()` hoặc định nghĩa trực tiếp các cặp khóa-giá trị bên trong `{}`.

---

## 2. Quản lý Cặp Khóa-Giá trị (Key-Value Management)

### 2.1. Thiết lập và Truy xuất
- **Thiết lập:** `d['tên_khóa'] = giá_trị`. Giá trị có thể là bất kỳ kiểu dữ liệu nào: chuỗi, số nguyên, hoặc thậm chí là một danh sách.
- **Truy xuất:** Để lấy giá trị, ta gọi `d['tên_khóa']`. Khác với danh sách, từ điển không sử dụng vị trí số (0, 1, 2) để truy xuất trừ khi số đó được định nghĩa làm khóa.

### 2.2. Tính linh hoạt của dữ liệu
Từ điển cho phép lưu trữ hỗn hợp nhiều kiểu dữ liệu. Ví dụ: một khóa có thể giữ tên người (chuỗi), trong khi khóa khác giữ phạm vi tuổi (danh sách các số nguyên). Khả năng này cực kỳ hữu ích trong việc đóng gói các cấu trúc dữ liệu phức tạp vào một biến duy nhất.

---

## 3. Các Phương thức Truy vấn Tập hợp

Để làm việc với các từ điển lớn hoặc được nhập từ các thư viện bên ngoài, Python cung cấp các phương thức chuyên dụng:
- **`.keys()`:** Trả về danh sách tất cả các khóa hiện có trong từ điển.
- **`.values()`:** Trả về danh sách tất cả các giá trị được lưu trữ.
- **`.items()`:** Trả về các cặp khóa-giá trị dưới dạng "iterable" (thành phần có thể lặp lại). Đây là công cụ quan trọng để vận hành các vòng lặp `for` nhằm duyệt qua toàn bộ dữ liệu trong từ điển.

---

## 4. Tối ưu hóa Khả năng Đọc mã
Tương tự như danh sách, việc định nghĩa từ điển trên nhiều dòng được khuyến khích trong thực hành lập trình tốt:
- Mỗi cặp khóa-giá trị nằm trên một dòng riêng biệt.
- Cho phép thêm chú thích (`#`) bên cạnh để giải thích vai trò của từng tham số.
Điều này giúp mã nguồn trở nên minh bạch và dễ bảo trì, đặc biệt là trong các cấu hình mô hình học máy phức tạp.

---

## 5. Kết luận
Từ điển là xương sống của việc quản lý thông tin có cấu trúc trong Python. Sự khác biệt về cú pháp (ngoặc nhọn) và cơ chế truy xuất (theo khóa thay vì chỉ mục số) là những điểm mấu chốt cần nắm vững. Việc sử dụng thành thạo từ điển sẽ giúp lập trình viên tổ chức mã nguồn một cách khoa học, tạo điều kiện thuận lợi cho việc xử lý các đối tượng dữ liệu phức tạp trong AI và khoa học dữ liệu.

---

## Tài liệu tham khảo (Citations)
1. Cấu trúc từ điển và ánh xạ cặp khóa-giá trị trong Python dựa trên `aero_LLM_05_Dictionaries.md`. Phân tích các phương thức truy vấn `.keys()`, `.values()` và `.items()`.
