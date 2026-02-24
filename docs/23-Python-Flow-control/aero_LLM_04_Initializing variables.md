# Nhập môn Python: Kỹ thuật Khởi tạo Biến và Cấp phát Bộ nhớ (Initializing Variables)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về tầm quan trọng của việc khởi tạo biến trong lập trình Python, đặc biệt là khi làm việc với các khối dữ liệu mảng. Chúng ta phân tích các phương thức khởi tạo thông qua thư viện NumPy như `np.zeros()` và `np.full()`, cũng như cơ chế xử lý giá trị đặc biệt `NaN` (Not a Number). Nghiên cứu thực hiện thực nghiệm so sánh giữa hai chiến lược quản lý dữ liệu: cấp phát bộ nhớ trước (pre-allocation) và mở rộng danh sách động (dynamic appending). Kết quả chỉ ra rằng việc khởi tạo trước mang lại ưu thế vượt trội về hiệu năng tính toán và hiệu quả sử dụng bộ nhớ, đồng thời thúc đẩy tư duy lập trình cấu trúc trong nghiên cứu mô hình ngôn ngữ lớn.

---

## 1. Vấn đề của Biến chưa định nghĩa (NameError)
Trong Python, việc cố gắng gán giá trị vào một chỉ mục của một biến chưa được khởi tạo (ví dụ: `R[i] = x`) sẽ dẫn đến lỗi `NameError`. Điều này xảy ra do Python yêu cầu thực thể mẹ (danh sách hoặc mảng) phải tồn tại trong bộ nhớ trước khi các phần tử thành phần được truy cập.

---

## 2. Các phương thức Khởi tạo với NumPy

### 2.1. Khởi tạo mảng Số không và Giá trị cố định
- **`np.zeros(n)`:** Tạo một mảng gồm $n$ số không. Mặc định các số này ở định dạng số thực dấu phẩy động (float).
- **`np.full(size, value)`:** Tạo mảng với kích thước chỉ định và lấp đầy bởi một giá trị cụ thể (ví dụ: `-99` hoặc `np.nan`).

### 2.2. Giá trị NaN (Not a Number)
`np.nan` là một thực thể toán học đặc biệt dùng để đại diện cho các giá trị thiếu hoặc không xác định. Trong khoa học dữ liệu, việc khởi tạo bằng `NaN` giúp lập trình viên dễ dàng nhận diện và loại bỏ các nhiễu dữ liệu trong quá trình hậu xử lý.

### 2.3. Khởi tạo Ma trận đa chiều
Để khởi tạo các cấu trúc 2D hoặc 3D, ta truyền vào một **Tuple** xác định kích thước: `np.zeros((hàng, cột))`. Việc chỉ định kiểu dữ liệu bằng tham số `dtype=int` cho phép tối ưu hóa không gian lưu trữ khi không cần đến độ chính xác thập phân.

---

## 3. So sánh Hiệu năng: Cấp phát trước vs. Thêm động

### 3.1. Cơ chế Thêm động (Appending)
Lập trình viên có thể bắt đầu với một danh sách trống `[]` và sử dụng `.append()` để mở rộng nó bên trong vòng lặp. Cách tiếp cận này linh hoạt nhưng tiềm ẩn rủi ro về hiệu năng khi quy mô dữ liệu lớn dần.

### 3.2. Ưu thế của Cấp phát trước (Pre-allocation)
Việc xác định trước kích thước mảng (ví dụ dùng `np.zeros`) và điền giá trị qua chỉ mục mang lại 3 lợi ích cốt lõi:
1. **Tốc độ:** Máy tính không cần thực hiện các thao tác cấp phát lại vùng nhớ liên tục như khi `append`.
2. **Bộ nhớ:** Sử dụng tài nguyên RAM hiệu quả và ổn định hơn.
3. **Tính minh bạch:** Buộc nhà nghiên cứu phải lập kế hoạch cụ thể về cấu trúc tensor (kích thước batch, chiều embedding) trước khi bắt đầu thực thi chương trình.

---

## 4. Kỹ thuật Mã hóa Mềm (Soft Coding)
Để mã nguồn trở nên chuyên nghiệp, các tham số về kích thước mảng nên được lưu trữ trong biến (ví dụ: `array_size = 15`). Khi thay đổi giá trị này ở đầu script, toàn bộ các hàm khởi tạo và vòng lặp liên quan sẽ tự động cập nhật, đảm bảo tính nhất quán và giảm thiểu sai sót thủ công.

---

## 5. Kết luận
Khởi tạo biến không chỉ là một thao tác kỹ thuật để tránh lỗi cú pháp mà còn là một chiến lược tối ưu hóa quan trọng. Trong kỷ nguyên của các mô hình LLM với hàng tỷ tham số, việc làm chủ kỹ thuật pre-allocation và sử dụng linh hoạt các thư viện như NumPy là điều kiện bắt buộc để xây dựng các hệ thống AI có hiệu năng cao và khả năng mở rộng tốt.

---

## Tài liệu tham khảo (Citations)
1. Các phương pháp khởi tạo biến và tối ưu hóa bộ nhớ trong Python dựa trên `aero_LLM_04_Initializing variables.md`. Phân tích hàm `zeros`, `full`, giá trị `NaN` và ưu thế của pre-allocation.
