# Bản Đồ Nhiệt Của Token Cho Cân Nhắc Định Tính (Text Heatmaps)

## Tóm tắt

Các phân tích nội tại của một Mô hình Ngôn ngữ Lớn sinh ra lượng lớn thông tin về số liệu vô tri khá trừu tượng. Để có cảm quan (intuition) về cách LLM hoạt động trên văn bản con người, phương pháp tạo ra **Bản đồ nhiệt văn bản** (Text Heatmaps) trở nên phổ biến. Bài viết nêu bật sự liên kết số tĩnh vào nền của chuỗi con từ tự nhiên, chuyển hóa thông số phân rã định lượng trở thành trực quan định tính.

---

## 1. Phương Pháp Lập Bản Đồ Nhiệt Văn Bản

Mỗi một `token` ($t_i$) ứng với một con số cụ thể thể hiện một đại lượng $X_i$ cho LLM. Kỹ thuật sau sử dụng sự đối sánh trực tiếp để tô màu vào hộp văn bản theo thông số liên kết.

### 1.1 Tính Toán Kích Cỡ

Do môi trường lập trình thường xuất dữ liệu thông qua cửa sổ hiển thị (như matplotlib), các chữ cái (characters) cần sử dụng một font đồng nhịp như Monospace để tính diện tích.

Với thiết lập: `Figure = 10 \times 2`, tỷ lệ cố định của 1 token sẽ được chuyển thành giá trị hình hộp (bounding box) cụ thể có tọa độ và chiều dài được lấy trực tiếp bởi thuật toán đồ họa. Từ đó lấy làm đơn vị cho $t_1, t_2...$

### 1.2 Biến Đổi Tỷ Lệ (Min-Max Scaling)

Để vẽ bản đồ nhiệt dựa trên sự chuyển sắc (color map - như đỏ nhạt sang đô), tập số nội tại cần được liên kết lên một khoảng giá trị tiêu chuẩn từ $0$ tới $1$. Phép biến đổi chuẩn được sử dụng là **Min-Max Scaling**.

Giả sử $x_i$ là số lượng ký tự trong chuỗi chữ $i$:

$$x_{norm} = \frac{x_i - X_{min}}{X_{max} - X_{min}}$$

Phép đổi chuẩn là tuyến tính (linear transformation). Nó không phá vỡ tính tương quan gốc rễ mà chỉ co ép số liệu vào khuôn khổ $[0,1]$ nhằm kết xuất màu thông qua hệ số RGB.

---

## 2. Ứng Dụng vào Ví Dụ Thực Tế

Ban đầu, thay vì gắn kích hoạt (activations) từ mạng Neural, bản vẽ Heatmap được giả lập thông qua độ dài dòng chữ `Lorem Ipsum`. Chữ có màu đỏ càng đậm ứng với các từ kéo dài (nhiều ký tự), chữ sáng trắng thuộc các phần tử từ vụn ngắn.

Điều này mô phỏng các giá trị logit nội bộ $Z$ (sẽ được tìm trong quá trình huấn luyện/trích xuất mô hình):
$$Z \rightarrow \text{Softmax}(\cdot) \rightarrow P_i \rightarrow X_i$$
Càng đậm màu tương đương với năng lực dự đoán tiếp theo càng chính xác định tính.

---

## 3. Thuận Lợi Và Rủi Ro

Mặc dù có nhiều lợi ích:
- Làm trực quan sự liên kết của vô vàn chỉ số mạng NN với quá trình sinh ra chữ của trí thông minh.
- Phân tách ra từng từ (hoặc Sub-word) rõ ràng.

Nhưng cũng hiện diện cả nguy cơ diễn giải sai lệch (over interpretation) vì nhiễu hoặc các mẫu ngẫu nhiên (noise and unrepresentative examples). Con người rất nhạy cảm với hình ảnh màu sắc và dễ gắn cho nó các quy luật giả (Phantom patterns), dù cho đôi khi số liệu đó bị sai hoặc lỗi.

---

## Tài liệu tham khảo

1. **Rethmeier, N. et al. (2020).** *Visualizing and Understanding the Interpretability of Natural Language.*
2. **Karpathy, A. (2015).** *The Unreasonable Effectiveness of Recurrent Neural Networks.* Blog.
3. **Elhage, N. et al. (2021).** *A Mathematical Framework for Transformer Circuits.* Anthropic.
