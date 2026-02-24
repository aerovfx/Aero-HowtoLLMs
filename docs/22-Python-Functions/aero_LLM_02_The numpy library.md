# Nhập môn Python: Thư viện NumPy và Thao tác Số học (The NumPy Library)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về NumPy, một thư viện nền tảng và thiết yếu cho tính toán khoa học trong hệ sinh thái Python. Chúng ta phân tích triết lý thiết kế "nhẹ" của Python thuần (Base Python) và cách các thư viện bên thứ ba mở rộng chức năng này. Nghiên cứu đi sâu vào cấu trúc dữ liệu mảng n-chiều (`ndarray`), sự khác biệt giữa mảng NumPy và danh sách truyền thống, cùng với các phương thức khởi tạo dữ liệu như `np.linspace()` và `np.arange()`. Báo cáo cũng chỉ ra những mâu thuẫn trong quy tắc về cận biên (cận trên bao hàm vs. cận trên loại trừ) giữa các hàm khác nhau, nhấn mạnh tầm quan trọng của việc kiểm chứng thực nghiệm khi làm việc với các thư viện mới.

---

## 1. Bản chất của Thư viện trong Python
Thư viện (Library) là một tập hợp các hàm và thuộc tính có liên quan được đóng gói cùng nhau.
- **Triết lý "Lightweight":** Python cơ bản được thiết kế để cài đặt nhanh chóng và chạy được trên mọi phần cứng từ máy chủ đến điện thoại. Do đó, nó chỉ chứa những hàm cốt lõi nhất.
- **NumPy (Numerical Python):** Là gói phần mềm chuyên dụng cho xử lý số học. Mọi thuật toán học sâu đều dựa trên các phép toán ma trận và tensor, vì vậy NumPy là một trong những thư viện được sử dụng nhiều nhất trong nghiên cứu AI.

---

## 2. Quản lý Thư viện và Cú pháp Truy cập

### 2.1. Nhập thư viện (Importing)
Để sử dụng các hàm của NumPy, ta cần nạp nó vào vùng làm việc.
- **Dạng rút gọn:** `import numpy as np`. Việc sử dụng bí danh `np` giúp tiết kiệm thời gian gõ mã và là tiêu chuẩn chung trong cộng đồng lập trình viên thế giới.

### 2.2. Truy cập qua dấu chấm (Dot Notation)
Để gọi một hàm từ thư viện, ta sử dụng cú pháp: `tên_thư_viện.tên_hàm`. Ví dụ: `np.mean(danh_sách)` cho kết quả trung bình cộng của các phần tử.

---

## 3. Mảng N-chiều (ndarray) - Cấu trúc Dữ liệu Nòng cốt
Mảng NumPy (`ndarray`) khác biệt đáng kể so với danh sách (list) về hiệu suất và khả năng tính toán:
- **Kiểu dữ liệu:** Kết quả của các phép toán NumPy thường trả về kiểu `float64` (độ chính xác 64-bit), cung cấp độ tin cậy cao hơn trong các tính toán khoa học.
- **Chuyển đổi:** Chúng ta có thể chuyển đổi một danh sách thông thường thành mảng NumPy bằng hàm `np.array()`. Điều này cho phép thực hiện các phép toán đại số tuyến tính mà danh sách thuần túy không hỗ trợ.

---

## 4. Các Hàm Khởi tạo và Quy tắc Cận biên

### 4.1. `np.linspace()` vs `np.arange()`
- **`np.linspace(start, stop, num)`:** Tạo ra một dãy số cách đều nhau giữa `start` và `stop`. Hàm này sử dụng **cận trên bao hàm** (kết quả có chứa số `stop`).
- **`np.arange(stop)`:** Tương đương với hàm `range()` của Python nhưng trả về một mảng. Hàm này sử dụng **cận trên loại trừ** (kết quả dừng lại ngay trước `stop`).

### 4.2. Sự mâu thuẫn về Cận trên
Một trong những thách thức đối với người mới bắt đầu là sự không nhất quán giữa các hàm:
- Một số hàm (như cắt lát và `arange`) loại trừ điểm cuối.
- Một số hàm khác (như `linspace`) bao hàm cả điểm cuối.
*Khuyến nghị:* Khi sử dụng các hàm mới, lập trình viên nên chạy thử nghiệm nhỏ để xác nhận hành vi của cận trên trước khi áp dụng vào các tính toán quy mô lớn.

---

## 5. Kết luận
NumPy không chỉ cung cấp các hàm toán học mà còn mang lại một hệ thống dữ liệu mảng hiệu năng cao, làm nền tảng cho mọi thư viện học sâu hiện đại như PyTorch và TensorFlow. Việc hiểu rõ cách nhập thư viện và quản lý các loại cận biên là bước chuẩn bị quan trọng để xử lý các khối lượng dữ liệu khổng lồ trong nghiên cứu LLM.

---

## Tài liệu tham khảo (Citations)
1. Thư viện NumPy và các hàm khởi tạo dữ liệu số học dựa trên `aero_LLM_02_The numpy library.md`. Phân tích kiểu dữ liệu `ndarray` và sự khác biệt về quy tắc cận trên giữa `linspace` và `arange`.
