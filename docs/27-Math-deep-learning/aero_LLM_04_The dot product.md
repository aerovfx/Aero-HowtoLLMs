# Toán học trong Học sâu: Tích vô hướng (The Dot Product)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về tích vô hướng (dot product), còn được gọi là tích vô hướng thực (scalar product), một phép toán đóng vai trò là "xương sống" tính toán cho hầu hết các kiến trúc học máy hiện đại. Chúng ta phân tích các hệ ký hiệu toán học phổ biến, cơ chế thực thi và ý nghĩa hình học của phép toán này. Nghiên cứu thực hiện thực nghiệm trên hai nền tảng NumPy và PyTorch để thẩm định tính chính xác của kết quả, đồng thời cảnh báo về sự khắt khe của PyTorch đối với tính đồng nhất của kiểu dữ liệu (Data Type Sensitivity). Kết quả khẳng định tích vô hướng thực chất là một cách gọi khác của tổ hợp tuyến tính có trọng số, nhưng với một khung lý thuyết rộng mở hơn trong đại số tuyến tính.

---

## 1. Hệ ký hiệu và Định nghĩa Toán học

Tích vô hướng kết nối hai vectơ có cùng số lượng phần tử để tạo ra một con số (số vô hướng) duy nhất.
- **Các dạng ký hiệu:** $a \cdot b$, $\langle a, b \rangle$, hoặc phổ biến nhất trong học sâu là $a^T b$ (vectơ $a$ chuyển vị nhân với vectơ $b$).
- **Bản chất phép toán:** Là tổng các tích của từng cặp phần tử tương ứng. 
- **Điều kiện tiên quyết:** Phép toán chỉ xác định khi hai vectơ có cùng số chiều. Nếu có sự chênh lệch về số lượng phần tử, tích vô hướng sẽ không thể thực hiện, tương tự như việc một nơ-ron không thể xử lý dữ liệu nếu thiếu hoặc thừa các kết nối trọng số.

---

## 2. Ứng dụng Đa phương diện trong AI và Toán học

Tích vô hướng không chỉ là một phép cộng nhân đơn thuần mà còn là phép đo lường sự tương đồng:
- **Trong NLP và LLM:** Sử dụng để tính độ tương đồng Cosine (Cosine Similarity) giữa các vectơ nhúng (embeddings), giúp mô hình hiểu được mối quan hệ ngữ nghĩa giữa các từ vựng.
- **Trong Xử lý tín hiệu:** Là nền tảng của các phép biến đổi Fourier và bộ lọc dữ liệu.
- **Trong Mạng nơ-ron:** Phục vụ quá trình lan truyền tiến (forward pass), phép tích chập (convolution) và tính toán ma trận Gram.

---

## 3. Thực thi Kỹ thuật: So sánh NumPy và PyTorch

### 3.1. Tính linh hoạt của NumPy
Hàm `np.dot()` trong NumPy rất mạnh mẽ và có khả năng tự động xử lý các tình huống trộn lẫn giữa số nguyên và số thực. Nó cũng được dùng rộng rãi cho cả nhân ma trận, điều này đôi khi gây nhầm lẫn cho người mới bắt đầu.

### 3.2. Tính khắt khe của PyTorch
Hàm `torch.dot()` trong PyTorch chỉ hoạt động trên các vectơ 1 chiều và yêu cầu tính đồng nhất tuyệt đối về kiểu dữ liệu:
- **Lỗi phổ biến:** Nếu một vectơ là số nguyên (`LongTensor`) và vectơ còn lại là số thực (`FloatTensor`), PyTorch sẽ báo lỗi thực thi.
- **Giải pháp:** Nhà nghiên cứu phải ép kiểu dữ liệu về `torch.float` để đảm bảo tính tương thích. Sự khắt khe này giúp ngăn ngừa các lỗi làm tròn số không mong muốn trong quá trình huấn luyện mô hình quy mô lớn.

---

## 4. Giải mã Ý nghĩa của Kết quả
Dù đầu vào là các vectơ có hàng nghìn chiều, kết quả của tích vô hướng luôn là một số duy nhất. Con số này phản ánh "điểm tương đồng" hoặc "mức độ kích hoạt" chung giữa hai vectơ. Trong mô hình ngôn ngữ, một tích vô hướng có giá trị lớn giữa vectơ câu hỏi và vectơ tài liệu cho thấy tài liệu đó có độ liên quan cao đến truy vấn.

---

## 5. Kết luận
Tích vô hướng là công cụ xử lý ngôn ngữ thực sự của máy tính. Việc hiểu rõ cơ chế của nó — từ các dấu ngoặc nhọn trong ký hiệu đến các thông báo lỗi kiểu dữ liệu trong mã nguồn — giúp nhà nghiên cứu làm chủ được cách thức mà AI "cảm nhận" và "so sánh" thông tin. Đây là bước đệm trực tiếp để tiến tới nhân ma trận, nơi hàng tỷ phép tích vô hướng được thực hiện đồng thời để tạo nên trí tuệ nhân tạo hiện đại.

---

## Tài liệu tham khảo (Citations)
1. Cơ sở toán học và thực thi tích vô hướng trên máy tính dựa trên `aero_LL_04_The dot product.md`. Phân tích hệ ký hiệu $a^T b$, ứng dụng trong Cosine Similarity và quản lý lỗi kiểu dữ liệu trong PyTorch.
