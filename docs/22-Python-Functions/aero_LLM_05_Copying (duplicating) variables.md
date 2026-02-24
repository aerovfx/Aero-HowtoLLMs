# Nhập môn Python: Cơ chế Sao chép Biến và Quản lý Bộ nhớ (Copying Variables)

## Tóm tắt (Abstract)
Báo cáo này phân tích cơ chế sao chép biến trong Python, một khía cạnh thường gây ra lỗi logic nghiêm trọng cho lập trình viên. Chúng ta nghiên cứu sự khác biệt giữa phép gán (assignment) đơn thuần và việc tạo ra bản sao vật lý của dữ liệu. Thông qua hàm `id()`, nghiên cứu minh chứng rằng Python thường sử dụng các "con trỏ" (pointers) thay vì sao chép toàn bộ nội dung để tối ưu hóa bộ nhớ. Báo cáo cũng đề xuất các phương pháp kỹ thuật để tách rời (decouple) các biến, bao gồm kỹ thuật cắt lát (slicing), các phép toán ảo và ứng dụng thư viện `copy`. Đây là kiến thức nền tảng để bảo toàn tính toàn vẹn của dữ liệu gốc trong quá trình tiền xử lý và biến đổi tensor.

---

## 1. Hiện tượng "Con trỏ" và Phép gán mặc định
Trong Python, khi thực hiện lệnh `B = A`, chúng ta không tạo ra một bản sao mới. Thay vào đó, cả `A` và `B` đều cùng trỏ về một vị trí dữ liệu duy nhất trên ổ cứng.
- **Hệ quả:** Mọi thay đổi thực hiện trên `B` sẽ ngay lập tức phản ánh lên `A`. 
- **Công cụ kiểm chứng:** Hàm `id(biến)` cung cấp một mã số định danh duy nhất cho vị trí bộ nhớ của biến đó. Nếu `id(A) == id(B)`, chúng thực chất là một thực thể duy nhất dưới hai cái tên khác nhau.

---

## 2. Kỹ thuật Sao chép cho từng Kiểu dữ liệu

### 2.1. Đối với Danh sách (List)
Sử dụng toán tử cắt lát toàn phần `[:]` là cách nhanh nhất để tạo ra một bản sao độc lập:
*Ví dụ:* `B = A[:]`. Lúc này, Python sẽ cấp phát một vùng nhớ mới cho `B` và sao chép toàn bộ giá trị từ `A` sang.

### 2.2. Đối với Mảng NumPy và PyTorch
Một mẹo lập trình phổ biến là thực hiện phép cộng ảo với số không:
*Ví dụ:* `F = E + 0`. Phép toán này không thay đổi giá trị nhưng buộc Python phải tạo ra một đối tượng mảng mới để chứa kết quả, từ đó decoupling (tách rời) thành công hai biến.

---

## 3. Sao chép sâu với thư viện `copy`
Đối với các cấu trúc phức tạp như Từ điển (Dictionary) hoặc các danh sách lồng nhau (nested components), các mẹo trên có thể không hiệu quả. 
- **Giải pháp:** Sử dụng hàm `copy.deepcopy()`.
- **Đặc điểm:** Hàm này thực hiện việc sao chép theo đệ quy, đảm bảo mọi tầng dữ liệu bên trong đều được tạo mới hoàn toàn, tách biệt tuyệt đối với biến gốc.

---

## 4. Lưu ý về Quản lý Phiên làm việc (Session Management)
Khi thực hiện thao tác **Restart Session**, toàn bộ bộ nhớ tạm của Python sẽ bị xóa sạch:
- Các biến đã định nghĩa sẽ mất.
- Các hàm đã tạo sẽ biến mất.
- Các thư viện đã nhập (như `import numpy as np`) cần phải được thực hiện lại từ đầu. Đây là hành động cần thiết khi môi trường gặp lỗi treo hoặc khi muốn làm sạch workspace để đảm bảo tính tái lập (reproducibility) của thực nghiệm.

---

## 5. Kết luận
Hiểu rõ cơ chế quản lý bộ nhớ thông qua các định danh ID là chìa khóa để viết mã nguồn an toàn và hiệu quả. Việc sử dụng đúng kỹ thuật sao chép (từ cắt lát đơn giản đến sao chép sâu) giúp lập trình viên kiểm soát tuyệt đối luồng dữ liệu, ngăn chặn những thay đổi ngoài ý muốn lên các tập dữ liệu huấn luyện quan trọng trong nghiên cứu LLM.

---

## Tài liệu tham khảo (Citations)
1. Cơ chế sao chép biến và quản lý ID bộ nhớ trong Python dựa trên `aero_LLM_05_Copying (duplicating) variables.md`. Phân tích phép gán, kỹ thuật slicing, và hàm `copy.deepcopy()`.
