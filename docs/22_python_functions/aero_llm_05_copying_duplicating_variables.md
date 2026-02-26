
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [22 python functions](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
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
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Nhập môn Python: Hàm, Đầu vào và Đầu ra (Functions, Inputs and Outputs)](aero_llm_01_inputs_and_outputs.md) | [Xem bài viết →](aero_llm_01_inputs_and_outputs.md) |
| [Nhập môn Python: Thư viện NumPy và Thao tác Số học (The NumPy Library)](aero_llm_02_the_numpy_library.md) | [Xem bài viết →](aero_llm_02_the_numpy_library.md) |
| [Nhập môn Python: Các Phương pháp Tra cứu và Hỗ trợ (Getting Help)](aero_llm_03_getting_help_on_functions.md) | [Xem bài viết →](aero_llm_03_getting_help_on_functions.md) |
| [Nhập môn Python: Kỹ thuật Xây dựng Hàm (Creating Functions)](aero_llm_04_creating_functions.md) | [Xem bài viết →](aero_llm_04_creating_functions.md) |
| 📌 **[Nhập môn Python: Cơ chế Sao chép Biến và Quản lý Bộ nhớ (Copying Variables)](aero_llm_05_copying_duplicating_variables.md)** | [Xem bài viết →](aero_llm_05_copying_duplicating_variables.md) |
| [Nhập môn Python: Kỹ thuật Tạo số Ngẫu nhiên với NumPy (Generating Random Numbers)](aero_llm_06_generating_random_numbers.md) | [Xem bài viết →](aero_llm_06_generating_random_numbers.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
