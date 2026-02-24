# Nhập môn PyTorch: Kỹ thuật Tái cấu trúc và Biến đổi Hình dạng Tensor (Reshaping Tensors)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu các phương thức thay đổi cấu trúc hình học của Tensor trong PyTorch mà không làm thay đổi nội dung dữ liệu bên trong. Chúng ta phân tích cơ chế chuyển vị (transpose), làm phẳng (flatten), và tái cấu trúc (reshape/view). Nghiên cứu đi sâu vào kỹ thuật sử dụng tham số `-1` để suy luận kích thước tự động và hàm `unsqueeze` để bổ sung các chiều đơn hình (singleton dimensions). Đây là những thao tác kỹ thuật cốt lõi để chuẩn bị dữ liệu đầu vào cho các lớp mạng nơ-ron, đảm bảo sự tương thích về mặt kích thước giữa các tầng kiến trúc khác nhau trong một mô hình LLM.

---

## 1. Phép Chuyển vị (Transpose)

Chuyển vị là hành động hoán đổi vị trí giữa các hàng và các cột (hoặc các chiều dữ liệu nói chung).
- **Đối với Ma trận 2D:** Sử dụng phương thức `.t()` để nhanh chóng hoán đổi hàng thành cột.
- **Đối với Tensor đa chiều:** Sử dụng hàm `torch.transpose(tensor, dim0, dim1)`. Khác với ma trận, trong không gian đa chiều (ví dụ 3D), lập trình viên phải chỉ định rõ bộ đôi chiều dữ liệu nào cần hoán đổi (ví dụ: hoán đổi giữa các "lớp" ma trận và các "hàng" bên trong ma trận đó).

---

## 2. Kỹ thuật Làm phẳng và Tái cấu trúc

### 2.1. Phép Làm phẳng (Flattening)
Phương thức `.flatten()` trải toàn bộ dữ liệu đa chiều thành một vectơ duy nhất (1D tensor). Trong đại số tuyến tính, đây gọi là quá trình vectơ hóa (vectorization). Thao tác này thường được dùng khi chúng ta quan tâm đến đặc tính phân phối số học của toàn bộ dữ liệu (như vẽ biểu đồ histogram) thay vì cấu trúc không gian của chúng.

### 2.2. Phương thức `view` và `reshape`
Đây là hai công cụ dùng để tổ chức lại các phần tử vào một hình dạng mới:
- **Nguyên tắc bảo toàn:** Tổng số phần tử phải giữ nguyên. Ví dụ: một ma trận 2x3 (6 phần tử) có thể chuyển thành 1x6 hoặc 3x2, nhưng không thể chuyển thành 1x4.
- **Tham số `-1` (Tự động hóa):** Bằng cách đặt `-1` cho một chiều, chúng ta yêu cầu PyTorch tự tính toán kích thước chiều đó dựa trên tổng số phần tử hiện có, giúp mã nguồn trở nên linh hoạt và giảm thiểu rủi ro tính toán sai thủ công.

---

## 3. Quản lý Chiều Đơn hình (Unsqueeze)

Hàm `.unsqueeze(dim)` chèn thêm một chiều có kích thước là 1 vào vị trí chỉ định.
- **Ý nghĩa:** Một vectơ có 9 phần tử (shape `[9]`) sau khi `unsqueeze(0)` sẽ trở thành một ma trận có 1 hàng và 9 cột (shape `[1, 9]`). 
- **Ứng dụng:** Thao tác này cực kỳ quan trọng khi mô hình yêu cầu dữ liệu đầu vào phải có chiều "Batch size", ngay cả khi chúng ta chỉ xử lý một mẫu dữ liệu duy nhất.

---

## 4. Tư duy Hình học: Mô hình "Lát bánh mì"
Để hiểu Tensor 3D, hãy tưởng tượng mỗi ma trận 2D là một lát bánh mì. 
- Một Tensor có hình dạng `[2, 3, 3]` tương đương với 2 lát bánh mì đặt chồng lên nhau, mỗi lát có kích thước 3x3.
- Việc chuyển vị giữa các chiều `0` và `2` thực chất là thay đổi góc nhìn từ "chồng bánh" sang "cạnh bánh", biến đổi cấu trúc từ `[2, 3, 3]` thành `[3, 3, 2]`.

---

## 5. Kết luận
Làm chủ kỹ thuật tái cấu trúc Tensor là kỹ năng "nhào nặn" dữ liệu cần thiết của mọi kỹ sư AI. Khả năng dịch chuyển linh hoạt giữa các chiều không gian cho phép nhà nghiên cứu tối ưu hóa hiệu suất tính toán và đảm bảo luồng thông tin được dẫn dắt chính xác qua các khối xử lý phức tạp của mạng nơ-ron sâu.

---

## Tài liệu tham khảo (Citations)
1. Kỹ thuật biến đổi hình dạng và chiều dữ liệu trong PyTorch dựa trên `aero_LL_04_Reshaping tensors.md`. Phân tích phép chuyển vị, làm phẳng, tái cấu trúc view/reshape và bổ sung chiều đơn hình.
