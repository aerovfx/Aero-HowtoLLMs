# Toán học trong Học sâu: Hàm Softmax và Diễn giải Xác suất (Softmax)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về hàm Softmax, một phép biến đổi phi tuyến quan trọng trong các bài toán phân loại đa lớp của học sâu. chúng ta phân tích vai trò của số mũ tự nhiên ($e$) trong việc tạo ra các đầu ra không âm, cơ chế chuẩn hóa dữ liệu về dạng phân phối xác suất (tổng bằng 1), và ý nghĩa của việc chuyển đổi các giá trị thô (logits) thành các mức độ tin cậy có thể diễn giải được. Nghiên cứu thực hiện thực nghiệm so sánh phương pháp tính toán thủ công trong NumPy và sử dụng module `torch.nn` trong PyTorch, qua đó làm rõ tính chất co giãn phi tuyến của hàm số đối với các giá trị đầu vào cực biên.

---

## 1. Cơ sở Toán học: Số mũ Tự nhiên ($e$)

Hàm Softmax dựa trên hằng số Euler $e \approx 2.718$. Hai đặc tính của hàm số mũ $e^x$ quyết định tính khả thi của Softmax:
- **Tính Dương tuyệt đối:** $e^x$ luôn lớn hơn 0 với mọi giá trị $x$ (ngay cả khi $x$ âm). Điều này đảm bảo xác suất đầu ra không bao giờ bị âm.
- **Tốc độ Tăng trưởng:** Hàm số mũ khuếch đại các giá trị lớn và thu nhỏ các giá trị nhỏ một cách nhanh chóng, tạo ra sự phân tách rõ rệt giữa các lớp đối tượng.

---

## 2. Công thức và Cơ chế Chuẩn hóa

Giả sử có một tập hợp các số thực $z$, hàm Softmax cho phần tử thứ $i$ được định nghĩa là:
$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
- **Tử số:** Chuyển đổi giá trị thô sang không gian số mũ.
- **Mẫu số:** Tổng của toàn bộ các giá trị sau khi lấy số mũ, đóng vai trò là hệ số chuẩn hóa.
- **Hệ quả:** Tập hợp đầu ra luôn nằm trong khoảng $(0, 1)$ và có tổng bằng chính xác $1.0$. Đặc tính này cho phép chúng ta coi đầu ra của mạng nơ-ron như một phân phối xác suất.

---

## 3. Diễn giải trong context Học sâu (Logits to Probs)

Các mô hình AI thường xuất ra các con số tùy ý (gọi là logits) không có ý nghĩa trực tiếp. Hàm Softmax đóng vai trò là một "bộ thông dịch":
- **Gán nhãn xác suất:** Chuyển đổi các số điểm thô thành xác suất cho từng danh mục (ví dụ: 0.9 xác suất là mèo, 0.05 là chó).
- **Tính phi tuyến:** Trong thực nghiệm, sự khác biệt nhỏ ở đầu vào (ví dụ từ 1 lên 2) tạo ra sự khác biệt rất lớn ở đầu ra sau khi qua Softmax. Ngược lại, các giá trị âm đều bị ép về gần 0, giúp mô hình tập trung vào các giả thuyết có khả năng cao nhất.

---

## 4. Thực thi Kỹ thuật: NumPy vs PyTorch

### 4.1. NumPy (Tiếp cận Thủ công)
Phép toán có thể thực hiện chỉ với một dòng mã: `np.exp(z) / np.sum(np.exp(z))`. Cách tiếp cận này giúp nhà nghiên cứu nắm vững bản chất toán học nhưng thiếu tối ưu hóa cho các tensor đa chiều phức tạp.

### 4.2. PyTorch (Tiếp cận Hướng đối tượng)
PyTorch cung cấp lớp `nn.Softmax(dim=...)`. Điểm lưu ý quan trọng là tham số `dim`:
- Phải chỉ định rõ chiều nào sẽ được chuẩn hóa (ví dụ `dim=0` cho vectơ hàng).
- PyTorch yêu cầu dữ liệu đầu vào phải là `torch.Tensor`, việc đưa vào một danh sách thông thường (`list`) sẽ dẫn đến lỗi logic.

---

## 5. Kết luận
Hàm Softmax là cầu nối giữa các phép toán đại số thô và ngôn ngữ xác suất của con người. Khả năng biến các tín hiệu điện toán phức tạp thành các phân phối xác suất chuẩn mực giúp các mô hình ngôn ngữ như GPT đưa ra các dự đoán từ kế tiếp một cách logic và có độ tin cậy cao. Việc làm chủ cả công thức toán học và kỹ thuật thực thi trong PyTorch là yêu cầu bắt buộc đối với bất kỳ kỹ sư AI nào.

---

## Tài liệu tham khảo (Citations)
1. Cơ sở toán học và ứng dụng của hàm Softmax trong mạng nơ-ron dựa trên `aero_LL_06_Softmax.md`. Phân tích hàm số mũ tự nhiên, cơ chế chuẩn hóa xác suất và thực nghiệm so sánh NumPy/PyTorch.
