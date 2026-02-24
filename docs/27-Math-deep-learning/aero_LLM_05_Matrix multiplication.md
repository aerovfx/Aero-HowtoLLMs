# Toán học trong Học sâu: Phép Nhân Ma trận (Matrix Multiplication)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về phép nhân ma trận, một kỹ thuật tính toán song song hóa hàng loạt các tích vô hướng (dot products) trong không gian đa chiều. chúng ta phân tích các quy tắc về kích thước (dimensionality rules) để xác định tính hợp lệ của phép toán, cơ chế ánh xạ từ hàng và cột sang ma trận kết quả, và sự khác biệt bản chất giữa nhân ma trận với nhân từng phần tử (Hadamard product). Nghiên cứu thực hiện các thực nghiệm trên NumPy và PyTorch để minh chứng cách thức tối ưu hóa mã nguồn thông qua toán tử `@`, đồng thời giải quyết các lỗi hệ thống liên quan đến hình dạng tensor và kiểu dữ liệu trong các mô hình ngôn ngữ lớn.

---

## 1. Bản chất và Quy tắc Kích thước

Phép nhân ma trận là một cấu trúc có tổ chức của các tích vô hướng, cho phép thực hiện hàng tỷ phép tính cùng lúc mà không cần sử dụng vòng lặp `for`.
- **Hệ tọa độ:** Ma trận được định nghĩa theo thứ tự **Hàng x Cột** ($m \times n$).
- **Điều kiện khả thi (Inner Dimensions):** Phép nhân $A \times B$ chỉ thực hiện được khi số cột của ma trận bên trái ($A$) bằng số hàng của ma trận bên phải ($B$). Ví dụ: $(5 \times 2) \times (2 \times 7)$ là hợp lệ, nhưng $(2 \times 7) \times (5 \times 2)$ thì không.
- **Kích thước kết quả (Outer Dimensions):** Ma trận mới sẽ có số hàng của $A$ và số cột của $B$.

---

## 2. Cơ chế Ánh xạ Tích vô hướng

Mỗi phần tử tại vị trí $(i, j)$ trong ma trận kết quả được tính bằng tích vô hướng của:
- **Hàng thứ $i$** của ma trận bên trái.
- **Cột thứ $j$** của ma trận bên phải.
Điều này giải thích tại sao nhân ma trận không có tính chất giao hoán ($A \cdot B \neq B \cdot A$). Việc thay đổi thứ tự nhân sẽ làm thay đổi hoàn toàn các cặp vectơ tham gia vào tích vô hướng.

---

## 3. Phân biệt các loại Phép nhân trên Máy tính

Cần phân biệt rõ hai loại phép toán thường gây nhầm lẫn trong lập trình:
- **Nhân Ma trận (Dot Product based):** Sử dụng toán tử `@` trong Python hoặc `torch.matmul()`. Đây là phép toán tạo ra các tổ hợp tuyến tính, đóng vai trò then chốt trong các lớp Dense và Attention.
- **Nhân Hadamard (Element-wise):** Sử dụng toán tử `*`. Phép toán này chỉ đơn giản là nhân các cặp phần tử tại cùng một tọa độ, không làm thay đổi kích thước và không tạo ra tổ hợp thông tin giữa các hàng/cột.

---

## 4. Thực thi và Tối ưu hóa trong PyTorch

PyTorch cung cấp các công cụ mạnh mẽ nhưng đòi hỏi sự khắt khe về mặt kỹ thuật:
- **Xử lý Hình dạng:** Nếu hai ma trận không khớp kích thước (ví dụ hai ma trận cùng là $5 \times 2$), chúng ta sử dụng phép chuyển vị `.T` để đưa về dạng $(5 \times 2) \times (2 \times 5)$, giúp phép toán trở nên khả thi.
- **Quản lý Kiểu dữ liệu:** Tương tự như tích vô hướng, `torch.matmul` yêu cầu các tensor phải có cùng kiểu (ví dụ cùng là `float32`). Sử dụng phương thức `.to()` hoặc `.float()` để chuẩn hóa dữ liệu trước khi nhân là một bước bắt buộc để tránh lỗi runtime.

---

## 5. Kết luận
Nhân ma trận là "động cơ vĩnh cửu" của trí tuệ nhân tạo. Khả năng nén hàng triệu phép tính nơ-ron vào một lệnh thực thi duy nhất không chỉ tối ưu hóa hiệu suất trên GPU mà còn cung cấp một khung lý thuyết mạch lạc để thiết kế các kiến trúc AI phức tạp. Việc nắm vững quy tắc "hàng nhân cột" và các toán tử tương ứng trong Python là kỹ năng sống còn của mọi nhà nghiên cứu trong kỷ nguyên đại mô hình.

---

## Tài liệu tham khảo (Citations)
1. Cơ chế và thực thi nhân ma trận trong học sâu dựa trên `aero_LL_05_Matrix multiplication.md`. Phân tích quy tắc kích thước nội/ngoại, so sánh với nhân Hadamard và ứng dụng toán tử @ trong PyTorch.
