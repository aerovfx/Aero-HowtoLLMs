# Học sâu: Thực thi Lan truyền xuôi trong PyTorch

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về cách thức thư viện PyTorch lưu trữ và triển khai các phép toán lan truyền xuôi (forward pass) của mạng nơ-ron nhân tạo. chúng ta thực hiện phân tích kiến trúc mã nguồn thông qua việc xây dựng một mô hình ngôn ngữ đơn giản (mini-LM), từ khâu khởi tạo (initialization) với lớp `nn.Module` đến việc thực thi các tầng tuyến tính (`nn.Linear`) và hàm kích hoạt. Nghiên cứu thực hiện thực nghiệm đối chứng bằng cách tính toán thủ công các phép toán ma trận để minh chứng rằng các module bậc cao của PyTorch thực chất là sự trừu tượng hóa của các phép tính tích vô hướng và cộng định kiến (bias), cung cấp một cái nhìn sâu sắc về cơ chế vận hành của các mô hình ngôn ngữ lớn như ChatGPT.

---

## 1. Kiến trúc Mô hình trong PyTorch

Mọi mô hình học sâu trong PyTorch đều được xây dựng dựa trên lớp cơ sở `nn.Module`. Lớp này cung cấp các cơ chế quản lý tham số, gradient và các "móc" (hooks) cần thiết mà không cần lập trình viên phải viết lại từ đầu:
- **`__init__` (Khởi tạo):** Nơi định nghĩa các thành phần của mạng như các tầng nhúng (embedding layers), tầng ẩn (hidden layers) và tầng đầu ra. Tần suất xuất hiện của lớp `nn.Linear` đại diện cho các ma trận trọng số (weights) và định kiến (bias).
- **`forward` (Lan truyền xuôi):** Định nghĩa luồng dữ liệu đi từ đầu vào qua các phép biến đổi tuyến tính và phi tuyến để tạo ra dự đoán.

---

## 2. Giải mã lớp `nn.Linear`

Mặc dù trong mã nguồn chúng ta chỉ gọi đơn giản là một lớp, nhưng về bản chất toán học, PyTorch đang thực hiện:
$$\text{Output} = X \cdot W^T + b$$
- **Thực nghiệm đối chứng:** Nghiên cứu đã thực hiện tính toán thủ công bằng cách lấy ma trận trọng số và vector định kiến trực tiếp từ thuộc tính của mô hình, sau đó nhân với dữ liệu đầu vào. Kết quả cho thấy sự trùng khớp hoàn hảo với đầu ra của PyTorch.
- **Tầm quan trọng:** Việc thấu hiểu lớp `nn.Linear` giúp chúng ta nhận ra rằng các nơ-ron thực chất là các hàng/cột trong một ma trận lớn, và việc huấn luyện chính là tinh chỉnh các giá trị trong ma trận đó.

---

## 3. Quy trình Xử lý Ngôn ngữ (Tokenization)

Mô hình không thể xử lý văn bản trực tiếp. chúng ta cần một quy trình chuyển đổi:
1. **Token hóa:** Chia nhỏ văn bản thành các đơn vị (ký tự hoặc từ) và gán cho mỗi đơn vị một số nguyên đại diện.
2. **Nhúng (Embedding):** Chuyển đổi các số nguyên này thành các vector đặc trưng trong không gian đa chiều.
3. **Dự đoán Token tiếp theo (Next Token Prediction):** Mô hình tính toán xác suất cho tất cả các ký tự có thể có trong từ điển và chọn ký tự có "độ kích hoạt" (activation) cao nhất.

---

## 4. Mô phỏng cơ chế của ChatGPT

Dù mini-LM được xây dựng trong nghiên cứu này chỉ sử dụng các con số ngẫu nhiên, nhưng nguyên lý vận hành của nó tương đồng với các mô hình khổng lồ:
- **Dòng chảy:** Input Text $\rightarrow$ Numbers $\rightarrow$ Forward Pass $\rightarrow$ Next Token Prediction $\rightarrow$ Output Text.
- **Sự khác biệt:** Các mô hình thương mại như ChatGPT có hàng tỷ tham số và trải qua quá trình huấn luyện khổng lồ để các con số trong ma trận trọng số không còn là ngẫu nhiên, mà mang trong mình "tri túc" về ngôn ngữ và tri thức nhân loại.

---

## 5. Kết luận
Việc thực thi lan truyền xuôi trong PyTorch là sự cân bằng nghệ thuật giữa tính trừu tượng cấp cao và hiệu năng tính toán. Bằng cách khám phá các chi tiết bên dưới lớp vỏ của `nn.Module`, nhà nghiên cứu có thể thấu hiểu được "linh hồn" toán học của AI mà không bị sa lầy vào những chi tiết lập trình cấp thấp. Khả năng truy cập vào các trọng số và kích hoạt (activations) thông qua các hooks là tiền đề quan trọng cho việc nghiên cứu tính giải thích được (interpretability) của các hệ thống AI hiện đại.

---

## Tài liệu tham khảo (Citations)
1. Cơ chế thực thi forward pass và phân tích lớp `nn.Linear` dựa trên `aero_LL_06_Forward pass in Pytorch.md`. Thuyết minh về sự kế thừa từ `nn.Module` và quy trình token hóa trong các mô hình ngôn ngữ. village.
