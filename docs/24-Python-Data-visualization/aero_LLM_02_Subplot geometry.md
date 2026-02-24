# Nhập môn Python: Hình học và Cấu trúc Biểu đồ con (Subplot Geometry)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về kỹ thuật tạo biểu đồ con (subplots) trong Matplotlib, một phương pháp thiết yếu để trình bày đa chiều thông tin trong cùng một không gian hình ảnh. Chúng ta phân tích cơ chế phân bổ trục tọa độ (axes) theo ma trận, cách thức điều chỉnh kích thước tổng thể thông qua tham số `figsize`, và kỹ thuật xử lý chồng lấn nhãn dán bằng hàm `tight_layout()`. Nghiên cứu cũng thực hiện các thực nghiệm về việc kết hợp vòng lặp `for` và hàm `enumerate()` để tự động hóa quy trình vẽ biểu đồ trên quy mô lớn, giúp tối ưu hóa thời gian phân tích kết quả thực nghiệm trong nghiên cứu AI.

---

## 1. Cấu trúc Ma trận của Biểu đồ con
Hàm `plt.subplots()` không chỉ tạo ra một hình ảnh đơn thuần mà là một lưới các khung hình độc lập.
- **Cấu trúc trả về:** Hàm trả về một cặp đối tượng `(fig, axes)`. Trong đó `fig` đại diện cho toàn bộ cửa sổ hình ảnh, và `axes` là một mảng NumPy chứa các khung tọa độ bên trong.
- **Biến ẩn (`_`):** Trong trường hợp không cần can thiệp vào các thuộc tính cấp cao của hình ảnh, lập trình viên thường sử dụng dấu gạch dưới `_` làm biến giữ chỗ cho `fig` để làm sạch bộ nhớ và mã nguồn.

---

## 2. Quản lý Không gian và Kích thước

### 2.1. Tham số `figsize`
Kích thước của hình ảnh được xác định bởi tham số `figsize=(chiều_rộng, chiều_cao)`. 
- **Lưu ý:** Các đơn vị này thường tương ứng với inch trên lý thuyết, nhưng thực tế sẽ thay đổi tùy thuộc vào độ phân giải và mức độ phóng to của màn hình người dùng. Việc lựa chọn tỷ lệ (aspect ratio) phù hợp là rất quan trọng để tránh làm biến dạng dữ liệu.

### 2.2. Kỹ thuật Bố cục Chặt chẽ (`tight_layout`)
Một trong những lỗi phổ biến khi vẽ nhiều biểu đồ là hiện tượng các nhãn trục (tick marks) của biểu đồ này đè lên tiêu đề của biểu đồ kia. Hàm `plt.tight_layout()` là giải pháp tự động để điều chỉnh biên và khoảng cách giữa các khung hình, đảm bảo tính thẩm mỹ và khả năng đọc của báo cáo khoa học.

---

## 3. Truy cập và Điều khiển Trục tọa độ
Vì `axes` là một mảng NumPy, chúng ta sử dụng các quy tắc chỉ mục (indexing) đã học để vẽ dữ liệu vào đúng vị trí mong muốn:
- **Lưới 1D (Dãy hàng hoặc cột):** `axes[0]`, `axes[1]`...
- **Lưới 2D (Ma trận):** `axes[hàng, cột]`. Ví dụ: `axes[0, 0]` truy cập vào biểu đồ ở góc trên cùng bên trái.
- **Thực thi lệnh:** Thay vì dùng `plt.plot()`, ta dùng `axes[i].plot()` để vẽ dữ liệu vào một khung hình cụ thể.

---

## 4. Tự động hóa với Vòng lặp For và Enumerate
Trong các nghiên cứu phức tạp (như so sánh hiệu năng mô hình qua các epoch khác nhau), việc vẽ thủ công từng biểu đồ là không khả thi.
- **Kỹ thuật:** Sử dụng `for i, ax in enumerate(axes):` để duyệt qua từng khung hình trong mảng.
- **Ứng dụng:** Kết hợp chỉ mục `i` từ `enumerate` để thay đổi các tham số tính toán (ví dụ: lũy thừa của X) trong mỗi lần lặp, cho phép tạo ra các bảng so sánh dữ liệu một cách nhất quán và nhanh chóng.

---

## 5. Kết luận
Làm chủ hình học biểu đồ con là chìa khóa để tạo ra các báo cáo phân tích dữ liệu chuyên nghiệp. Khả năng sắp xếp và tự động hóa quy trình vẽ biểu đồ không chỉ giúp tiết kiệm không gian trình bày mà còn cho phép người xem so sánh trực tiếp các biến số, từ đó làm nổi bật lên những mối tương quan tiềm ẩn trong dữ liệu thực nghiệm.

---

## Tài liệu tham khảo (Citations)
1. Kỹ thuật tạo biểu đồ con và quản lý bố cục trong Matplotlib dựa trên `aero_LL_02_Subplot geometry.md`. Phân tích hàm subplots, tham số figsize, kỹ thuật tight_layout và tự động hóa vòng lặp.
