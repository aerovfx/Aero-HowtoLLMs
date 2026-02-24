# Toán học trong Học sâu: Hệ thuật ngữ và Kiểu dữ liệu trong Điện toán (Terms and Datatypes)

## Tóm tắt (Abstract)
Báo cáo này thiết lập một khung tham chiếu chung cho các thuật ngữ toán học và khoa học máy tính được sử dụng trong lĩnh vực học sâu. Chúng ta phân tích các đối tượng đại số tuyến tính từ số vô hướng (scalar) đến tensor đa chiều, đồng thời phân biệt khái niệm "kiểu dữ liệu" (data type) dưới hai góc nhìn: thống kê học và khoa học máy tính. Nghiên cứu nhấn mạnh tầm quan trọng của việc quản lý kiểu dữ liệu trong các thư viện NumPy và PyTorch, minh chứng rằng sự tương thích về định dạng lưu trữ là yếu tố quyết định tính thực thi của các thuật toán mạng nơ-ron.

---

## 1. Phân loại Đối tượng Đại số Tuyến tính

Trong toán học, kích thước và cách sắp xếp của các tập hợp số xác định tên gọi và tính chất hình học của chúng:
- **Số vô hướng (Scalar):** Một hằng số đơn lẻ dùng để co giãn (stretch) hoặc thu nhỏ (shrink) các vectơ.
- **Vectơ (Vector):** Một dãy số 1 chiều. Phân biệt giữa **Vectơ cột** (dạng đứng) và **Vectơ hàng** (dạng nằm ngang).
- **Ma trận (Matrix):** Bảng dữ liệu 2 chiều gồm các hàng và cột (tương tự bảng tính Excel).
- **Tensor:** Một khối dữ liệu đa chiều (từ 3D trở lên). Trong đồ họa máy tính và xử lý tín hiệu, Tensor là cấu trúc vạn năng để lưu trữ thông tin phức tạp.

---

## 2. Nhập nhằng Thuật ngữ: "Kiểu dữ liệu" (Data Type)

Cần phân biệt rõ hai định nghĩa thường gây nhầm lẫn cho người mới bắt đầu:
- **Trong Thống kê:** Đề cập đến đặc tính của biến số (định danh, thứ bậc, khoảng, tỷ lệ) để quyết định phương pháp phân tích thống kê phù hợp.
- **Trong Khoa học Máy tính (Trọng tâm của khóa học):** Đề cập đến định dạng lưu trữ vật lý trong bộ nhớ (ví dụ: `int` cho số nguyên, `float` cho số thực, `bool` cho logic).
- **Hệ quả:** Việc hiểu kiểu dữ liệu điện toán giúp nhà nghiên cứu điều phối bộ nhớ và đảm bảo tính chính xác của các phép toán dấu phẩy động (floating-point precision).

---

## 3. Hệ sinh thái Lưu trữ trong Python

Các thư viện khác nhau sử dụng các thuật ngữ khác nhau cho cùng một bản chất dữ liệu:
- **NumPy:** Gọi các cấu trúc đa chiều là `ndarray` (n-dimensional array).
- **PyTorch:** Nhất quán gọi mọi cấp độ dữ liệu (từ một con số đến một hypercube) là **Tensor**.
Sự khác biệt này yêu cầu nhà nghiên cứu phải thực hiện các bước chuyển đổi kiểu (type-casting) khi luân chuyển dữ liệu giữa các thư viện, ví dụ: chuyển từ danh sách (`list`) sang `numpy array`, sau đó snag `torch tensor` để tính toán trên GPU.

---

## 4. Tầm quan trọng của tính Đồng nhất Dữ liệu
Dù giá trị số học có thể giống nhau (ví dụ số 1 và 1.0), nhưng nếu kiểu dữ liệu không khớp, các hàm xử lý trong PyTorch sẽ báo lỗi. Việc nắm vững hệ thuật ngữ này không chỉ giúp đọc hiểu tài liệu kỹ thuật mà còn là chìa khóa để gỡ lỗi (debug) các mô hình LLM quy mô lớn, nơi sự sai lệch kiểu dữ liệu nhỏ nhất cũng có thể dẫn đến sự sụp đổ của toàn bộ quá trình huấn luyện.

---

## 5. Kết luận
Xây dựng một nền tảng thuật ngữ vững chắc là bước đi đầu tiên để làm chủ toán học trong học sâu. Việc hiểu rõ mối quan hệ giữa các cấu trúc toán học cổ điển và phương thức biểu diễn của chúng trên máy tính giúp nhà nghiên cứu thu hẹp khoảng cách giữa lý thuyết trừu tượng và thực thi mã nguồn, tạo tiền đề cho việc xây dựng các kiến trúc AI hiện đại và hiệu quả.

---

## Tài liệu tham khảo (Citations)
1. Hệ thuật ngữ toán học và kiểu dữ liệu máy tính trong học sâu dựa trên `aero_LL_01_Terms and datatypes in math and computers.md`. Phân tích đối tượng đại số tuyến tính, so sánh đa góc nhìn về kiểu dữ liệu và hệ sinh thái PyTorch/NumPy.
