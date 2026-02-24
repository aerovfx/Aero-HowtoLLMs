# Giải phẫu Nội tại Mô hình bằng Hooks: Kỹ thuật Trích xuất Hoạt hóa (Extracting Activations via Hooks)

## Tóm tắt (Abstract)
Báo cáo này hướng dẫn phương pháp sử dụng "Hooks" – các hàm can thiệp đặc biệt trong PyTorch – để truy cập và trích xuất dữ liệu từ các lớp ẩn bên trong Transformer. Trong khi các phương thức thông thường chỉ cho phép quan sát Logits đầu ra hoặc Hidden States của toàn bộ khối Transformer, kỹ thuật Hook cho phép nhà nghiên cứu cô lập các thành phần vi mô như ma trận Query (Q), Key (K), Value (V) hoặc các lớp MLP. Báo cáo cũng thảo luận về cơ chế quản lý Hook (đăng ký và gỡ bỏ) và cách quản lý bộ nhớ thông qua việc ghi đè hoặc tích lũy dữ liệu.

---

## 1. Mở Đầu (Introduction)
Để thực hiện Diễn giải học cơ học (Mechanistic Interpretability), việc biết trọng số (weights) của mô hình là chưa đủ. Chúng ta cần biết cách các nơ-ron thực sự phản ứng (activations) khi dữ liệu cụ thể đi qua. Hooks đóng vai trò như các "cảm biến" được cấy vào dòng chảy dữ liệu của mô hình trong quá trình forward-pass, cho phép ta chụp lại trạng thái của bất kỳ nơ-ron nào mà không cần sửa đổi cấu trúc cốt lõi của mạng.

---

## 2. Cơ chế Hoạt động của PyTorch Hooks

### 2.1. Định nghĩa Hàm Hook
Một hàm Hook tiêu chuẩn nhận ba tham số đầu vào:
1. **Module:** Lớp (layer) mà hook được gắn vào.
2. **Input:** Dữ liệu đi vào lớp đó.
3. **Output:** Kết quả tính toán đi ra khỏi lớp đó.
Bên trong hàm này, ta có thể trích xuất `output`, thực hiện các phép toán (như tách các chiều Q, K, V) và lưu trữ kết quả vào một biến bên ngoài (thường là Dictionary hoặc List).

### 2.2. Đăng ký và Quản lý (Registration & Handles)
Sử dụng phương thức `register_forward_hook` để cấy hàm vào mô hình. Kết quả trả về là một `handle`, có thể được sử dụng để gỡ bỏ (`remove()`) hook khi không còn cần thiết, giúp tối ưu hóa hiệu năng và tránh rò rỉ bộ nhớ.

---

## 3. Quản lý Dữ liệu Hoạt hóa (Data Management)

### 3.1. Ghi đè (Overwriting via Dictionary)
Nếu lưu trữ dữ liệu vào một `Dictionary` với key là tên tầng, mỗi lượt forward-pass mới sẽ ghi đè lên dữ liệu cũ. Đây là cách tiếp cận phổ biến khi ta chỉ quan tâm đến phản hồi của mô hình đối với câu lệnh hiện tại. 
*Lưu ý:* Nếu câu lệnh mới có các token đầu tiên giống câu lệnh cũ, các hàng tương ứng trong ma trận hoạt hóa sẽ giống nhau do tính chất truyền tin theo trình tự.

### 3.2. Tích lũy (Accumulation via List)
Bằng cách sử dụng `List` và phương thức `append()`, ta có thể lưu trữ lịch sử hoạt hóa của tất cả các câu lệnh đã đi qua mô hình. Điều này hữu ích cho các phân tích thống kê diện rộng hoặc so sánh sự biến thiên của nơ-ron qua nhiều ngữ cảnh khác nhau.

---

## 4. Phân tích Dữ liệu trích xuất
Khi đã có dữ liệu qua Hook, ta có thể thực hiện các phân tích trực quan:
- **Scatter Plots:** So sánh hoạt hóa của hai token khác nhau trên toàn bộ các nơ-ron của một tầng.
- **Correlation Matrices:** Đo lường sự tương quan giữa các token. Quan sát thực nghiệm cho thấy token đầu tiên thường có độ tương quan thấp với phần còn lại do thiếu hụt ngữ cảnh tiền đề.

---

## 5. Kết Luận
Hooks là công cụ mạnh mẽ nhất để biến một mô hình "hộp đen" thành một hệ thống có thể quan sát được ở mọi cấp độ hạt. Việc làm chủ kỹ thuật này không chỉ giúp trích xuất dữ liệu mà còn đặt nền móng cho việc chỉnh sửa hoạt hóa (activation editing) – một kỹ thuật can thiệp nhân quả sâu sắc hơn sẽ được thảo luận ở các chương sau.

---

## Tài liệu tham khảo (Citations)
1. Kỹ thuật trích xuất hoạt hóa bằng Hooks trên GPT-2 dựa trên `aero_LLM_05_Extracting activations using hooks.md`. Phân tích sự khác biệt giữa cơ chế Overwriting và Concatenation.
