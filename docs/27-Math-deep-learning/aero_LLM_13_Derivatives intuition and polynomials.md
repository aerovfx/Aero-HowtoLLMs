# Toán học trong Học sâu: Trực giác về Đạo hàm và Đa thức (Derivatives)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về vai trò quyết định của đạo hàm trong cơ chế huấn luyện của các mô hình học sâu, đặc biệt là trong thuật toán lan truyền ngược (backpropagation) và hạ giang (gradient descent). chúng ta phân tích trực giác hình học của đạo hàm như độ dốc (slope) của hàm số tại từng điểm, đồng thời thiết lập mối liên hệ giữa các hàm kích hoạt quan trọng như ReLU, Sigmoid và các bản đạo hàm tương ứng của chúng. Nghiên cứu thực hiện các thực nghiệm trên thư viện SymPy để minh chứng quy tắc lũy thừa trong tính toán đạo hàm đa thức, tạo nền tảng lý thuyết để hiểu cách thức mô hình AI tự điều chỉnh các tham số nhằm cực tiểu hóa hàm mất mát.

---

## 1. Trực giác Hình học: Đạo hàm là Độ dốc

Đạo hàm của một hàm số tại một điểm cho biết hàm số đó đang thay đổi như thế nào đối với biến đầu vào $x$:
- **Hàm ReLU (Rectified Linear Unit):** Có độ dốc bằng 0 khi $x < 0$ (hàm phẳng) và độ dốc bằng 1 khi $x > 0$ (hàm tăng tuyến tính). Đạo hàm này cho phép thông tin đi qua hoặc bị chặn lại một cách dứt khoát.
- **Hàm Sigmoid:** Có hình chữ S mềm mại. Đạo hàm của nó đạt giá trị cực đại tại $x=0$ (nơi hàm số thay đổi nhanh nhất) và tiến dần về 0 khi $x$ tiến về vô cùng. Điều này phản ánh tốc độ "bão hòa" của nơ-ron.
- **Nguyên lý:** Đạo hàm dương nghĩa là hàm số đang tăng, đạo hàm âm nghĩa là hàm số đang giảm, và đạo hàm bằng 0 nghĩa là hàm số đang ở trạng thái dừng (có thể là cực trị).

---

## 2. Đại số Đạo hàm: Quy tắc Lũy thừa cho Đa thức

Đối với các hàm đa thức, đạo hàm được tính theo quy tắc hệ thống:
$$\frac{d}{dx}(ax^n) = nax^{n-1}$$
- **Cơ chế:** Đưa số mũ xuống làm hệ số nhân và giảm bậc của biến số đi 1 đơn vị. 
- **Ví dụ:** Đạo hàm của $x^2$ là $2x$, đạo hàm của $x^3$ là $3x^2$. 
Khả năng tính toán đạo hàm một cách tự động và chính xác là chìa khóa để các thư viện như PyTorch có thể huấn luyện những mô hình có hàng tỷ tham số.

---

## 3. Tại sao Học sâu cần Đạo hàm?

Deep Learning thực chất là một bài toán tối ưu hóa. Chúng ta định nghĩa một "hàm mất mát" (loss function) đo lường sai số của mô hình:
- **Hướng di chuyển:** Đạo hàm cho chúng ta biết "hướng" cần phải thay đổi các trọng số (weights) của mô hình để làm giảm sai số. 
- **Tối ưu hóa:** Bằng cách đi ngược hướng của đạo hàm (gradient), mô hình sẽ dần dần "trượt" xuống điểm có lỗi thấp nhất. Nếu không có đạo hàm, chúng ta sẽ không có la bàn để biết phải điều chỉnh mô hình theo hướng nào giữa hàng tỷ khả năng.

---

## 4. Thực thi Lập trình với SymPy

Nghiên cứu sử dụng SymPy – một thư viện toán học ký hiệu trong Python:
- **Biến ký hiệu (Symbols):** Khác với NumPy xử lý các mảng số thực, SymPy cho phép chúng ta làm việc với các biến đại số như $x$, $y$.
- **Hàm `sp.diff()`:** Cho phép tính toán chính xác biểu thức toán học của đạo hàm thay vì chỉ ước lượng bằng số. Kết quả trả về là một công thức đại số tường minh, giúp nhà nghiên cứu kiểm chứng các thuộc tính lý thuyết của hàm kích hoạt trước khi đưa vào huấn luyện thực tế.

---

## 5. Kết luận
Đạo hàm không chỉ là một công thức khô khan trong sách giáo khoa, mà là "động cơ" bên trong của mọi thuật toán AI hiện đại. Việc thấu hiểu trực giác về độ dốc và cách thức tính toán đạo hàm cơ bản giúp chúng ta kiểm soát được quá trình huấn luyện, nhận diện được các vấn đề như triệt tiêu gradient (vanishing gradients). Trong các chương tiếp theo, chúng ta sẽ mở rộng khái niệm này sang đạo hàm riêng phần và quy tắc chuỗi – những thành phần cốt lõi cấu thành nên thuật toán Hạ giang (Gradient Descent).

---

## Tài liệu tham khảo (Citations)
1. Trực giác hình học và đại số của đạo hàm trong deep learning dựa trên `aero_LL_13_Derivatives intuition and polynomials.md`. Phân tích độ dốc của ReLU/Sigmoid, quy tắc lũy thừa đa thức và ứng dụng SymPy trong toán học ký hiệu.
