# Toán học trong Học sâu: Phép Chuyển vị Vectơ và Ma trận (Transpose)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về phép toán chuyển vị (transpose), một công cụ điều chỉnh hướng (orientation) cơ bản nhưng thiết yếu trong đại số tuyến tính và học sâu. Chúng ta phân tích cơ chế toán học của việc hoán đổi hàng thành cột, đồng thời thực hiện các thực nghiệm so sánh cú pháp giữa hai thư viện NumPy và PyTorch. Nghiên cứu nhấn mạnh quy tắc bảo toàn nội dung dữ liệu qua phép chuyển vị kép và ứng dụng của nó trong việc chuẩn bị ma trận cho các phép nhân trọng số trong mạng nơ-ron.

---

## 1. Nguyên lý Toán học của Phép Chuyển vị

Ký hiệu: $v^T$ hoặc $M^T$ (với $T$ nằm ở số mũ).
- **Định nghĩa:** Phép chuyển vị là quá trình "lật" một đối tượng toán học qua đường chéo chính của nó, biến các hàng thành các cột và ngược lại.
- **Biến đổi Vectơ:** Một vectơ cột (đứng) sau khi chuyển vị sẽ trở thành một vectơ hàng (nằm ngang).
- **Tính chất Đối nghịch:** Việc thực hiện chuyển vị hai lần liên tiếp $((A^T)^T = A)$ sẽ đưa đối tượng về trạng thái định hướng ban đầu. Điều này cho phép chúng ta thay đổi hướng dữ liệu tạm thời để tính toán mà không làm mất đi cấu trúc gốc của dữ liệu.

---

## 2. Quy tắc Ánh xạ Ma trận

Khi chuyển vị một ma trận kích thước $m \times n$, ma trận mới sẽ có kích thước $n \times m$:
- **Phép gán chính xác:** Cột thứ nhất của ma trận gốc trở thành hàng thứ nhất của ma trận mới. Cột thứ hai trở thành hàng thứ hai, v.v.
- **Lưu ý:** Cần tránh nhầm lẫn giữa chuyển vị và phép quay (rotation). Phép quay có thể làm thay đổi thứ tự tương đối giữa các hàng, trong khi chuyển vị bảo toàn trật tự tuyến tính của các phần tử theo hệ tọa độ mới.

---

## 3. Thực thi trên Máy tính: NumPy và PyTorch

### 3.1. Cú pháp NumPy
Trong NumPy, vectơ hoặc ma trận thường được biểu diễn dưới dạng `ndarray`.
- **Sử dụng thuộc tính `.T`:** Đây là cách viết ngắn gọn và phổ biến nhất (ví dụ: `matrix.T`).
- **Hàm `np.transpose()`:** Cung cấp tính năng tương tự nhưng dưới dạng một lời gọi hàm độc lập.

### 3.2. Sự nhất quán trong PyTorch
PyTorch kế thừa phần lớn triết lý của NumPy để giảm thiểu rào cản học tập cho nhà nghiên cứu.
- **Tương đồng:** Cả hai thư viện đều hỗ trợ thuộc tính `.T`.
- **Khác biệt:** Điểm duy nhất cần lưu ý là kiểu dữ liệu đầu ra (`torch.Tensor` so với `numpy.ndarray`). Mặc dù kết quả số học hoàn toàn trùng khớp, nhưng việc duy trì kiểu dữ liệu nhất quán là bắt buộc để thực hiện các phép toán lan truyền ngược (backpropagation) trên GPU.

---

## 4. Ứng dụng trong Mô hình Ngôn ngữ
Trong các cơ chế Attention của LLM, việc chuyển vị ma trận là thao tác xảy ra liên tục (ví dụ: nhân ma trận Query với chuyển vị của ma trận Key: $QK^T$). Việc thấu hiểu cơ chế này giúp nhà nghiên cứu kiểm soát được dòng chảy của các tensor qua các lớp của mô hình, đảm bảo các phép toán tích vô hướng (dot product) được thực hiện chính xác trên các chiều vector tương ứng.

---

## 5. Kết luận
Chuyển vị là một phép toán đơn giản về mặt logic nhưng lại là "chìa khóa" kỹ thuật để kết nối các khối kiến trúc khác nhau trong học sâu. Việc nắm vững cách thực thi cả trên lý thuyết giấy và mã nguồn Python giúp lập trình viên linh hoạt hơn trong việc thiết kế các phép toán ma trận phức tạp, đồng thời tạo nền tảng vững chắc để tiếp cận các chủ đề nâng cao như tích chập (convolution) và cơ chế chú ý (attention).

---

## Tài liệu tham khảo (Citations)
1. Thao tác chuyển vị vectơ và ma trận trong môi trường lập trình Python dựa trên `aero_LL_02_Vector and matrix transpose.md`. Phân tích định hướng không gian, thuộc tính .T trong NumPy/PyTorch và tính chất chuyển vị kép.
