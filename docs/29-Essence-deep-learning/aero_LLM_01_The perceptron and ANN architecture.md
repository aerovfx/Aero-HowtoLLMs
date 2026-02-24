# Học sâu: Perceptron và Kiến trúc Mạng Nơ-ron Nhân tạo (ANN)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về các khối xây dựng cơ bản của mạng nơ-ron nhân tạo (Artificial Neural Networks - ANN), bắt đầu từ mô hình Perceptron. chúng ta phân tích cấu trúc của một đơn vị tính toán cơ bản bao gồm: các nút đầu vào (input nodes), các trọng số liên kết (weights), phép toán tích vô hướng (dot product) và vai trò của hàm kích hoạt phi tuyến (activation function). Nghiên cứu giải mã lý do tại sao một mô hình tuyến tính thuần túy là không đủ để giải quyết các bài toán phức tạp trong thế giới thực và thuyết minh về tầm quan trọng của thành phần định kiến (bias term) – tương đương với hệ số chặn (intercept) trong thống kê – để tối ưu hóa khả năng phân tách của các siêu phẳng trong không gian dữ liệu.

---

## 1. Cấu trúc của một Perceptron

Perceptron là "tế bào" cơ bản của mọi kiến trúc học sâu. Nó hoạt động như một cỗ máy tính toán đơn giản với quy trình ba bước:
1. **Tiếp nhận Đầu vào:** Một tập hợp các số thực đại diện cho dữ liệu đầu vào ($x_1, x_2, ..., x_n$).
2. **Trọng số và Tính toán:** Mỗi đầu vào được nhân với một trọng số tương ứng ($w_1, w_2, ..., w_n$), sau đó được cộng dồn lại.
3. **Đầu ra:** Kết quả tổng hợp (weighted sum) được chuyển đổi thành một giá trị đầu ra duy nhất.

Về bản chất toán học, Perceptron thực hiện phép tính **tích vô hướng** giữa véc-tơ đầu vào $x$ và véc-tơ trọng số $w$:
$$y = x^T w = \sum_{i=1}^{n} x_i w_i$$

---

## 2. Tính Tuyến tính và Giới hạn phân tách

Perceptron thuần túy là một **mô hình tuyến tính**. Điều này có nghĩa là nó chỉ thực hiện các phép cộng và nhân vô hướng:
- **Ưu điểm:** Cực kỳ hiệu quả trong việc giải quyết các bài toán "tuyến tính khả phân" (linearly separable), nơi chúng ta có thể tách biệt hai nhóm dữ liệu bằng một đường thẳng.
- **Hạn chế:** Không thể giải quyết các bài toán phức tạp hơn nơi dữ liệu bị trộn lẫn theo các mô hình cong hoặc xoắn ốc.
- **Quy tắc vàng:** Không nên dùng mô hình phi tuyến cho bài toán tuyến tính (gây phức tạp hóa vô ích) và tuyệt đối không thể dùng mô hình tuyến tính cho bài toán phi tuyến (không thể giải quyết được).

---

## 3. Hàm Kích hoạt (Activation Function)

Để mở rộng khả năng của mạng nơ-ron, chúng ta đưa kết quả của phép tính tuyến tính qua một hàm phi tuyến $\sigma$ (thường được gọi là hàm kích hoạt):
$$\hat{y} = \sigma(x^T w)$$
- **Ví dụ cơ bản:** Hàm signum (hàm dấu) trả về +1 nếu tổng lớn hơn 0 và -1 nếu ngược lại.
- **Vai trò:** Phá vỡ tính tuyến tính, cho phép mô hình học được các ranh giới quyết định phức tạp hơn. Trong học sâu hiện đại, chúng ta thường sử dụng các hàm như ReLU, Sigmoid hoặc Tanh.

---

## 4. Vai trò của Thành phần Định kiến (Bias Term)

Thành phần định kiến (bias) là một tham số độc lập không liên kết với dữ liệu đầu vào. Nó tương đương với biến $b$ trong phương trình đường thẳng $y = mx + b$:
- **Lý do cần thiết:** Nếu không có bias, mọi "đường ranh giới" mà mô hình tạo ra buộc phải đi qua gốc tọa độ $(0,0)$. Điều này làm hạn chế khả năng phân loại nếu các cụm dữ liệu nằm ở những vị trí xa gốc tọa độ.
- **Thực thi:** Trong các thư viện như PyTorch, bias được tích hợp sẵn theo mặc định. Nó cho phép mô hình dịch chuyển đường ranh giới linh hoạt trên không gian dữ liệu để tìm ra vị trí phân tách tối ưu nhất.

---

## 5. Kết luận
Perceptron là sự kết hợp hoàn hảo giữa giải tích tuyến tính và các phép toán phi tuyến đơn giản. Toàn bộ sức mạnh của các mô hình LLM khổng lồ thực chất là sự chồng chất của hàng tỷ đơn vị Perceptron này theo các kiến trúc đa tầng phức tạp. Thấu hiểu bản chất của trọng số, tích vô hướng và bias là bước đi đầu tiên để làm chủ quá trình huấn luyện và tối ưu hóa các mạng nơ-ron nhân tạo trong tương lai.

---

## Tài liệu tham khảo (Citations)
1. Nguyên lý kiến trúc và công thức toán học của Perceptron dựa trên `aero_LL_01_The perceptron and ANN architecture.md`. Phân tích vai trò của tích vô hướng, hàm kích hoạt phi tuyến và thành phần định kiến trong tối ưu hóa mạng nơ-ron.
