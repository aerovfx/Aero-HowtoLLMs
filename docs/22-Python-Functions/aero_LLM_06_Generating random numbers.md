# Nhập môn Python: Kỹ thuật Tạo số Ngẫu nhiên với NumPy (Generating Random Numbers)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu các phương pháp tạo số ngẫu nhiên thông qua module `numpy.random`, một công cụ thiết yếu trong việc khởi tạo trọng số (weights) cho các mô hình học sâu. chúng ta phân tích các phân phối xác suất khác nhau bao gồm: phân phối Chuẩn (Gaussian), phân phối Đều (Uniform) và các hàm tạo số nguyên ngẫu nhiên. Nghiên cứu cũng đi sâu vào cách cấu trúc dữ liệu từ mảng một chiều (vector) đến mảng đa chiều (matrix) thông qua hệ thống dấu ngoặc vuông lồng nhau, cùng với kỹ thuật lấy mẫu ngẫu nhiên từ một tập dữ liệu có sẵn bằng hàm `choice()`.

---

## 1. Vai trò của Số ngẫu nhiên trong Học sâu (Deep Learning)
Trong lĩnh vực AI, các mô hình ngôn ngữ lớn (LLM) không bắt đầu với các quy tắc định sẵn. Thay vào đó, chúng bắt đầu như một "tờ giấy trắng" chứa đầy các con số ngẫu nhiên làm trọng số. Quá trình huấn luyện thực chất là việc điều chỉnh hàng triệu con số ngẫu nhiên này trở nên có ý nghĩa. Do đó, khả năng tạo ra các khối dữ liệu ngẫu nhiên quy mô lớn là yêu cầu tiên quyết đối với mọi thư viện tính toán.

---

## 2. Phân phối Chuẩn và Cấu trúc Mảng Đa chiều

### 2.1. Phân phối Chuẩn (Normal/Gaussian Distribution)
Hàm `np.random.randn()` trích xuất các con số từ một quần thể có giá trị trung bình (mean) bằng $0$ và độ lệch chuẩn (standard deviation) bằng $1$.
- **Đặc điểm:** Kết quả bao gồm cả số âm và số dương, tập trung nhiều quanh giá trị 0.

### 2.2. Phân biệt Vector và Ma trận qua Dấu ngoặc
Python sử dụng hệ thống ngoặc vuông lồng nhau để biểu diễn chiều của dữ liệu:
- **Vector (1D):** `[1, 2, 3]` - Một cặp ngoặc bao quanh dãy số.
- **Ma trận (2D):** `[[1, 2], [3, 4]]` - Hai lớp ngoặc. Lớp bên trong đại diện cho các hàng (rows), lớp bên ngoài bao bọc toàn bộ các hàng để tạo thành ma trận.

---

## 3. Phân phối Đều và Số nguyên Ngẫu nhiên

### 3.1. Phân phối Đều (Uniform Distribution)
Sử dụng hàm `np.random.uniform(low, high, size)`. Trong phân phối này, mọi giá trị trong khoảng từ `low` đến `high` đều có xác suất xuất hiện như nhau. Đây là lựa chọn lý tưởng khi muốn đảm bảo dữ liệu đầu vào không bị thiên kiến về một vùng giá trị cụ thể.

### 3.2. Số nguyên Ngẫu nhiên (`randint`)
Hàm `np.random.randint()` tạo ra các số nguyên ngẫu nhiên. 
- **Quy tắc Cận biên:** Tương tự như hàm `range`, hàm này sử dụng **cận trên loại trừ**. Ví dụ: `randint(0, 5)` sẽ chỉ trả về các số từ 0 đến 4.

---

## 4. Kỹ thuật Lấy mẫu Ngẫu nhiên (`choice`)
Hàm `np.random.choice(mảng_nguồn, số_lượng)` cho phép trích xuất ngẫu nhiên các phần tử từ một tập hợp dữ liệu đã định nghĩa trước.
- **Ứng dụng:** Thường được dùng để tạo các tập con (subsets) ngẫu nhiên từ dữ liệu huấn luyện hoặc thực hiện các thuật toán Monte Carlo.
- **Cơ chế:** Mỗi phần tử trong mảng nguồn có xác suất được chọn tương đương nhau (xác suất bằng $1/n$ với $n$ là độ dài mảng).

---

## 5. Kết luận
Module `numpy.random` cung cấp một bộ công cụ toàn diện để mô phỏng tính ngẫu nhiên trong toán học. Việc hiểu rõ sự khác biệt giữa các kiểu phân phối (Chuẩn vs. Đều) và quy tắc loại trừ cận trên của số nguyên giúp lập trình viên kiểm soát chính xác cấu trúc dữ liệu đầu vào cho các thuật toán học máy phức tạp.

---

## Tài liệu tham khảo (Citations)
1. Thao tác tạo số ngẫu nhiên với NumPy dựa trên `aero_LLM_06_Generating random numbers.md`. Phân tích phân phối Gaussian, Uniform và hàm `choice()`.
