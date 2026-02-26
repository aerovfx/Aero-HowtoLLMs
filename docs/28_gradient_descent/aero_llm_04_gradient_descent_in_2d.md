
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [28 gradient descent](index.md)

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
# Học sâu: Hạ giang trong Không gian 2 Chiều (2D Gradient Descent)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về việc mở rộng thuật toán Hạ giang (Gradient Descent) từ không gian một chiều sang không gian hai chiều, mô phỏng các bài toán tối ưu hóa thực tế hơn trong học sâu. chúng ta phân tích các khái niệm đạo hàm riêng (partial derivatives) và gradient – véc-tơ tập hợp các đạo hàm riêng theo mọi hướng. Nghiên cứu sử dụng hàm "Peaks" (một hàm thử nghiệm kinh điển trong toán học) để thực hiện các thực nghiệm tìm kiếm cực tiểu trên một bề mặt lồi lõm phức tạp. Kết quả cho thấy mặc dù chiều của không gian tăng lên, nguyên lý cập nhật tham số và cấu trúc thuật toán vẫn giữ được tính nhất quán và hiệu quả.

---

## 1. Đạo hàm riêng và Khái niệm Gradient

Khi làm việc với hàm số nhiều biến (ví dụ $f(x, y)$), sự thay đổi của hàm số phụ thuộc vào từng biến số một cách độc lập:
- **Đạo hàm riêng (Partial Derivative):** Là đạo hàm của hàm số theo một biến (ví dụ $x$), trong khi coi biến còn lại ($y$) là hằng số. Ký hiệu bằng biểu tượng "del" ($\partial$).
- **Gradient ($\nabla$):** Là một véc-tơ chứa tất cả các đạo hàm riêng của hàm số. Trong không gian 2D, gradient là $\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]$.
- **Ý nghĩa:** Gradient chỉ hướng mà hàm số tăng nhanh nhất. Trong học sâu, chúng ta luôn đi theo hướng ngược lại với gradient ($-\nabla$) để tìm điểm thấp nhất.

---

## 2. Bề mặt Lỗi và Thuật toán Hạ giang 2D

Nghiên cứu sử dụng hàm Peaks để tạo ra một "địa hình" có nhiều đỉnh núi và thung lũng:
- **Cấu trúc dữ liệu:** Điểm khởi tạo và điểm cực tiểu hiện tại không còn là một con số đơn lẻ mà là một cặp tọa độ $(x, y)$.
- **Cơ chế cập nhật:** Thay vì chỉ trượt trên một đường cong, mô hình giờ đây "lăn" trên một bề mặt. Vị trí mới được tính bằng cách trừ đi gradient (tổng hợp lực từ hai hướng $x$ và $y$) nhân với tốc độ học.
- **Tính đồng nhất:** Thuật toán về cơ bản vẫn giữ nguyên vòng lặp epochs đã học ở không gian 1D, minh chứng rằng Gradient Descent có khả năng mở rộng (scalability) cực tốt lên các không gian cao chiều.

---

## 3. Thực thi Kỹ thuật với SymPy và Lambdify

Do hàm Peaks có cấu trúc đại số rất phức tạp, việc tính đạo hàm bằng tay là cực kỳ khó khăn:
- **Toán học ký hiệu:** Sử dụng SymPy để tính toán các biểu thức đạo hàm riêng một cách chính xác tuyệt đối. 
- **Chuyển đổi (Lambdify):** Chuyển các công thức đại số của SymPy thành các hàm NumPy có thể thực thi nhanh chóng để tính toán các giá trị số cụ thể trong vòng lặp huấn luyện. Quy trình này mô phỏng cách các thư viện AI hiện đại tự động hóa việc tính toán gradient.

---

## 4. Phân tích Quỹ đạo (Trajectory Analysis)

Thông qua việc vẽ quỹ đạo di chuyển của mô hình trên bản đồ nhiệt (heatmap):
- **Phụ thuộc vào điểm khởi đầu:** Nếu khởi đầu ở gần một thung lũng nông, mô hình sẽ hội tụ về cực tiểu địa phương thay vì tìm đến cực tiểu toàn cầu sâu hơn. 
- **Xác suất:** Việc chạy mô hình nhiều lần với các điểm bắt đầu khác nhau cho thấy sự đa dạng của các giải pháp mà Gradient Descent có thể tìm thấy.
- **Tính liên tục:** Quỹ đạo di chuyển cho thấy mô hình luôn tìm con đường dốc nhất để đi xuống, giống như dòng nước chảy từ đỉnh núi xuống hồ.

---

## 5. Kết luận
Việc làm chủ Gradient Descent trong không gian 2D là cầu nối quan trọng để tiến tới các mô hình học sâu thực thụ. Chúng ta nhận thấy rằng véc-tơ gradient là một chiếc la bàn hoàn hảo, bất kể không gian là 2 chiều hay 2 tỷ chiều. Thấu hiểu đạo hàm riêng giúp nhà nghiên cứu biết cách kiểm soát từng thành phần trong mạng nơ-ron, từ đó tối ưu hóa quá trình huấn luyện và chẩn đoán các lỗi hội tụ một cách khoa học.

---

## Tài liệu tham khảo (Citations)
1. Mở rộng thuật toán Hạ giang và phân tích đạo hàm riêng dựa trên `aero_LL_04_Gradient descent in 2D.md`. Phân tích hàm Peaks, véc-tơ Gradient và thực thi Lambdify trong Python để tự động hóa tính toán gradient.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Học sâu: Tổng quan về Thuật toán Hạ giang (Gradient Descent)](aero_llm_01_overview_of_gradient_descent.md) | [Xem bài viết →](aero_llm_01_overview_of_gradient_descent.md) |
| [Học sâu: Vấn đề Cực trị Địa phương (Local Minima)](aero_llm_02_what_about_local_minima.md) | [Xem bài viết →](aero_llm_02_what_about_local_minima.md) |
| [Học sâu: Thực thi Hạ giang trong Không gian 1 Chiều (1D Gradient Descent)](aero_llm_03_gradient_descent_in_1d.md) | [Xem bài viết →](aero_llm_03_gradient_descent_in_1d.md) |
| 📌 **[Học sâu: Hạ giang trong Không gian 2 Chiều (2D Gradient Descent)](aero_llm_04_gradient_descent_in_2d.md)** | [Xem bài viết →](aero_llm_04_gradient_descent_in_2d.md) |
| [Học sâu: Thử thách Lập trình – Tốc độ học Cố định vs. Động (Fixed vs. Dynamic Learning Rate)](aero_llm_05_codechallenge_fixed_vs_dynamic_learning_rate.md) | [Xem bài viết →](aero_llm_05_codechallenge_fixed_vs_dynamic_learning_rate.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
