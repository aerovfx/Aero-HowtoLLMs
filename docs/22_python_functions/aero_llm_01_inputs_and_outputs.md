
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
# Nhập môn Python: Hàm, Đầu vào và Đầu ra (Functions, Inputs and Outputs)

## Tóm tắt (Abstract)
Báo cáo này giới thiệu khái niệm về "Hàm" (Functions) trong Python – công cụ cốt lõi để đóng gói và tái sử dụng các khối mã nguồn. Chúng ta nghiên cứu cơ chế vận hành của hàm thông qua quan hệ Đầu vào (Input) và Đầu ra (Output), đồng thời phân tích các hàm dựng sẵn phổ biến như `sum()`, `len()` và `print()`. Nghiên cứu cũng đi sâu vào việc xử lý ngoại lệ khi truyền sai kiểu dữ liệu, cơ chế ẩn đầu ra khi thực hiện phép gán trong Notebook, và thực hiện một thực nghiệm tính toán giá trị trung bình để minh chứng cho nhu cầu sử dụng các thư viện bổ trợ như NumPy.

---

## 1. Khái niệm và Vai trò của Hàm
Trong lập trình, hàm là một tập hợp các dòng mã được thiết kế để thực hiện một tác vụ cụ thể.
- **Tính tái sử dụng:** Thay vì viết lại cùng một thuật toán nhiều lần, lập trình viên đóng gói nó vào một hàm và gọi tên hàm khi cần.
- **Tính cấu trúc:** Hàm giúp chia nhỏ các bài toán phức tạp (như huấn luyện mô hình) thành các module đơn giản, dễ kiểm soát.

---

## 2. Cơ chế Đầu vào và Đầu ra

### 2.1. Tham số Đầu vào (Parameters/Inputs)
Hàm nhận dữ liệu thông qua các dấu ngoặc đơn `()`. 
- **Ví dụ:** Hàm `sum(danh_sách)` nhận một danh sách số và trả về tổng của chúng.
- **Ràng buộc kiểu:** Mỗi hàm yêu cầu loại dữ liệu cụ thể. Việc truyền một chuỗi ký tự (`str`) vào hàm `sum()` sẽ gây ra lỗi `TypeError` vì toán tử cộng (`+`) bị quá tải (overloaded) và không thể xử lý hỗn hợp số và chữ theo cách thông thường.

### 2.2. Giá trị Đầu ra (Return Values/Outputs)
Khi một hàm thực thi xong, nó có thể trả về một kết quả.
- **Gán biến:** Kết quả có thể được lưu trữ vào một biến để sử dụng sau này (ví dụ: `kết_quả = sum(danh_sách)`).
- **Lưu ý về Notebook:** Khi kết quả của hàm được gán cho một biến ở dòng cuối cùng của ô mã, Notebook sẽ không hiển thị giá trị đó ra màn hình. Để xem kết quả, ta cần gọi tên biến đó ở một dòng riêng biệt.

---

## 3. Phân tích Thực nghiệm: Tính Giá trị Trung bình
Qua việc triển khai thuật toán tính trung bình cộng ($Average = \frac{\sum X}{n}$), chúng ta rút ra được hai quan sát quan trọng:

1. **Độ nhạy Chữ hoa/thường (Case Sensitivity):** Python coi `listCount` và `listcount` là hai thực thể hoàn toàn khác nhau. Một lỗi đánh máy nhỏ trong tên biến sẽ dẫn đến lỗi `NameError`.
2. **Hạn chế của Python Thuần (Base Python):** Python cơ bản không cung cấp sẵn hàm `mean()` hay `ave18_rage()`. Để thực hiện các phép toán thống kê này, lập trình viên phải tự xây dựng thuật toán hoặc sử dụng các thư viện chuyên dụng như NumPy.

---

## 4. Tầm quan trọng của các Thư viện (Libraries)
Việc tự viết mọi thuật toán (từ tính trung bình đến các phép toán ma trận phức tạp) là cực kỳ tốn thời gian và dễ sai sót. Đây là lý do tại sao hệ sinh thái Python dựa mạnh vào các thư viện:
- **NumPy:** Xử lý mảng và toán học số học.
- **PyTorch:** Xử lý tensor và học sâu.
- **Pandas:** Quản lý và phân tích dữ liệu bảng.

---

## 5. Kết luận
Hàm là đơn vị cơ bản cấu thành nên logic của mọi ứng dụng AI. Việc nắm vững cách tương tác giữa dữ liệu đầu vào và kết quả đầu ra, cùng với ý thức về các ràng buộc kiểu dữ liệu, là bước đệm thiết yếu để chuyển từ việc viết mã đơn lẻ sang xây dựng các hệ thống tự động hóa phức tạp. Trong các bài học tiếp theo, chúng ta sẽ khám phá cách mở rộng sức mạnh của hàm thông qua việc nhập (import) các thư viện phần mềm chuyên sâu.

---

## Tài liệu tham khảo (Citations)
1. Cơ sở về hàm và tương tác đầu vào/đầu ra trong Python dựa trên `aero_LLM_01_Inputs and outputs.md`. Phân tích hàm `sum()`, `len()` và nhu cầu về thư viện bên thứ ba.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[Nhập môn Python: Hàm, Đầu vào và Đầu ra (Functions, Inputs and Outputs)](aero_llm_01_inputs_and_outputs.md)** | [Xem bài viết →](aero_llm_01_inputs_and_outputs.md) |
| [Nhập môn Python: Thư viện NumPy và Thao tác Số học (The NumPy Library)](aero_llm_02_the_numpy_library.md) | [Xem bài viết →](aero_llm_02_the_numpy_library.md) |
| [Nhập môn Python: Các Phương pháp Tra cứu và Hỗ trợ (Getting Help)](aero_llm_03_getting_help_on_functions.md) | [Xem bài viết →](aero_llm_03_getting_help_on_functions.md) |
| [Nhập môn Python: Kỹ thuật Xây dựng Hàm (Creating Functions)](aero_llm_04_creating_functions.md) | [Xem bài viết →](aero_llm_04_creating_functions.md) |
| [Nhập môn Python: Cơ chế Sao chép Biến và Quản lý Bộ nhớ (Copying Variables)](aero_llm_05_copying_duplicating_variables.md) | [Xem bài viết →](aero_llm_05_copying_duplicating_variables.md) |
| [Nhập môn Python: Kỹ thuật Tạo số Ngẫu nhiên với NumPy (Generating Random Numbers)](aero_llm_06_generating_random_numbers.md) | [Xem bài viết →](aero_llm_06_generating_random_numbers.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
