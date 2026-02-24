# Nhập môn PyTorch: Kỹ thuật Tạo số Ngẫu nhiên và Phân phối Dữ liệu (Random Numbers)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu các phương pháp khởi tạo dữ liệu ngẫu nhiên trong PyTorch, một quy trình thiết yếu để khởi tạo trọng số mạng nơ-ron và xáo trộn tập dữ liệu huấn luyện. chúng ta phân tích sự tương đồng giữa PyTorch và NumPy, cơ chế các phân phối thống kê (Gaussian và Uniform), và kỹ thuật hoán vị ngẫu nhiên (random permutation). Nghiên cứu cũng đi sâu vào việc giải quyết các lỗi phổ biến trong chỉ mục đa chiều khi thực hiện xáo trộn dữ liệu (shuffling), cung cấp các quy tắc thực nghiệm để đảm bảo tính nhất quán giữa hình dạng Tensor (shape) và logic truy cập bộ nhớ.

---

## 1. Khởi tạo theo Phân phối Xác suất

PyTorch cung cấp các hàm chuyên biệt để tạo ra các tập hợp số liệu có đặc tính thống kê định sẵn:
- **Phân phối Chuẩn (Gaussian):** Hàm `torch.randn(hàng, cột)` tạo ra Tensor có giá trị trung bình (mean) xấp xỉ 0 và độ lệch chuẩn (standard deviation) bằng 1. Đây là phương pháp phổ biến nhất để khởi tạo các tham số ban đầu cho mô hình AI.
- **Số nguyên Ngẫu nhiên:** Hàm `torch.randint(thấp, cao, size=(shape))` cho phép tạo ra các chỉ số ngẫu nhiên trong một khoảng xác định. Lưu ý: tham số `size` phải là một bộ tuple đại diện cho kích thước các chiều.

---

## 2. Kỹ thuật Hoán vị và Xáo trộn Dữ liệu (Shuffling)

Trong huấn luyện mô hình ngôn ngữ, việc thay đổi thứ tự các mẫu dữ liệu (Data Shuffling) là bắt buộc để tránh hiện tượng mô hình học thuộc lòng trình tự đầu vào.
- **Hàm `randperm(n)`:** Trả về một dãy số từ 0 đến n-1 đã được xáo trộn ngẫu nhiên, không lặp lại và không bỏ sót phần tử nào.
- **Quy trình Thực thi:**
    1. Tạo một dãy chỉ số ngẫu nhiên bằng `randperm`.
    2. Sử dụng dãy chỉ số này để truy cập vào Tensor dữ liệu gốc.
    Kết quả là một phiên bản dữ liệu mới với các phần tử đã được thay đổi vị trí một cách ngẫu nhiên nhưng vẫn bảo toàn giá trị.

---

## 3. Quản lý Chỉ mục trong Không gian Đa chiều

Một lỗi hệ thống thường gặp là sự không khớp giữa số lượng chiều của Tensor và số lượng chỉ số được cung cấp:
- **Vấn đề:** Khi một Tensor có shape `[1, 43]`, việc truy cập bằng một chỉ số duy nhất sẽ dẫn đến lỗi logic vì hệ thống mong đợi thông tin cho cả hai chiều.
- **Giải pháp (Quy tắc dấu phẩy):** Số lượng dấu phẩy trong lệnh truy cập chỉ mục phải tương ứng với số lượng chiều được liệt kê trong thuộc tính `.shape`. Đối với Tensor có chiều đơn hình (singleton dimension), ta phải chỉ định rõ chỉ số `0` cho chiều đó (ví dụ: `tensor[0, id_ngẫu_nhiên]`).

---

## 4. Tương quan hệ sinh thái PyTorch - NumPy
Phần lớn các hàm tạo dãy số như `linspace` đều có mặt trong cả hai thư viện với cú pháp tương đồng. Điểm khác biệt duy nhất nằm ở kiểu dữ liệu trả về (Tensor vs. Array). Sự tương đồng này cho phép nhà nghiên cứu chuyển đổi linh hoạt các kỹ thuật tiền xử lý dữ liệu giữa hai môi trường mà không cần tái cấu trúc logic tính toán.

---

## 5. Kết luận
Khả năng điều khiển tính ngẫu nhiên là chìa khóa để xây dựng các mô hình học máy mạnh mẽ và linh hoạt. Việc làm chủ các hàm khởi tạo và kỹ thuật xáo trộn dữ liệu, kết hợp với tư duy quản lý chiều không gian chặt chẽ, giúp nhà nghiên cứu kiểm soát tốt quá trình hội tụ của mô hình và đảm bảo tính khách quan của các kết quả thực nghiệm trong lĩnh vực xử lý ngôn ngữ tự nhiên.

---

## Tài liệu tham khảo (Citations)
1. Phương pháp tạo số ngẫu nhiên và kỹ thuật hoán vị trong PyTorch dựa trên `aero_LL_05_Random numbers.md`. Phân tích phân phối chuẩn, hàm randperm và quản lý lỗi chỉ mục đa chiều.
