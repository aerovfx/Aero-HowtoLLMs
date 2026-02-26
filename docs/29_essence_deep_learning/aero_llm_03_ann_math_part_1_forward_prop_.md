
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [29 essence deep learning](index.md)

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
# Học sâu: Giải tích ANN Phần 1 – Lan truyền xuôi (Forward Propagation)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về cơ chế lan truyền xuôi (forward propagation), quy trình cơ bản để biến đổi dữ liệu đầu vào thành kết quả dự đoán trong mạng nơ-ron nhân tạo. chúng ta phân tích cách thức các phép toán tích vô hướng (dot product) kết hợp với các hàm kích hoạt phi tuyến để tạo ra ranh giới quyết định (decision boundary). Nghiên cứu thực hiện thực nghiệm trực quan hóa trên không gian đặc trưng 2D để minh chứng rằng mặc dù mỗi đơn vị nơ-ron (perceptron) thực hiện phép tính tuyến tính bên trong, nhưng việc áp dụng các hàm kích hoạt như Sigmoid hay ReLU đã biến đổi kết quả thành các giá trị có ý nghĩa xác suất, cho phép mô hình thực hiện các nhiệm vụ phân loại phức tạp.

---

## 1. Cơ chế Toán học của Lan truyền xuôi

Lan truyền xuôi là quá trình dữ liệu đi từ trái sang phải qua mạng nơ-ron. Đối với một perceptron đơn lẻ, quy trình này được tóm gọn qua hai bước:
1. **Thành phần Tuyến tính:** Tính tổng có trọng số của các đầu vào, bao gồm cả thành phần định kiến (bias).

$$

$$

z = x^T w + b

$$

$$

Trong đó, $b$ có thể được hấp thụ vào tích vô hướng bằng cách thêm một hằng số 1 vào véc-tơ đầu vào.
2. **Thành phần Phi tuyến (Hàm kích hoạt):** Kết quả $z$ được đưa qua một hàm phi tuyến $\sigma$ để tạo ra giá trị dự đoán cuối cùng $\hat{y}$.

$$

$$

\hat{y} = \sigma(z)

$$

$$

---

## 2. Trực quan hóa Ranh giới Quyết định (Decision Boundary)

Trong không gian đặc trưng (ví dụ: giờ học và giờ ngủ), mỗi bộ trọng số $w$ sẽ xác định một siêu phẳng phân tách:
- **Ý nghĩa:** Tại vị trí mà $\hat{y} = 0$ (hoặc $\hat{y} = 0.5$ đối với hàm Sigmoid), đó chính là ranh giới nơi mô hình thay đổi quyết định từ loại này sang loại khác.
- **Phân loại Tuyến tính:** Nếu không có hàm kích hoạt phi tuyến, mô hình chỉ có thể tạo ra các ranh giới là đường thẳng (trong không gian 2D) hoặc mặt phẳng.
- **Tính toán:** Các cặp giá trị đầu vào $($x_1$, $x_2$)$ nằm về một phía của ranh giới sẽ được gán cho Lớp 1, và phía ngược lại là Lớp 2.

---

## 3. Các hàm Kích hoạt Phổ biến

Nghiên cứu nhấn mạnh ba hàm kích hoạt nền tảng trong học sâu:
1. **Sigmoid:** Nén đầu ra vào khoảng $[0, 1]$, thường được dùng ở tầng cuối cùng để dự đoán xác suất.
2. **Tanh (Tangent Hyperbolic):** Nén đầu ra vào khoảng $[-1, 1]$, giúp điều chỉnh dữ liệu quanh giá trị 0.
3. **ReLU (Rectified Linear Unit):** Trả về 0 cho các giá trị âm và giữ nguyên giá trị dương. Đây là hàm phổ biến nhất ở các tầng ẩn (hidden layers) do tính đơn giản và hiệu quả tính toán.

Điểm quan trọng: Các nút nơ-ron luôn là **tuyến tính khi đi vào (input)** và trở thành **phi tuyến khi đi ra (output)**.

---

## 4. Chuyển đổi giá trị thành Xác suất

Khi sử dụng hàm kích hoạt Sigmoid, các giá trị số thô từ phép tính tuyến tính được chuyển đổi thành xác suất:
- Các giá trị dương lớn tiến gần đến xác suất tuyệt đối bằng 1.
- Các giá trị âm lớn tiến gần đến xác suất bằng 0.
- Tại ranh giới quyết định, xác suất là 0.5, thể hiện sự không chắc chắn cao nhất của mô hình.
Hàm kích hoạt không làm thay đổi vị trí của ranh giới phân tách (do trọng số quyết định), nhưng nó làm thay đổi cách chúng ta giải thích độ tin cậy của dự đoán khi càng xa ranh giới đó.

---

## 5. Kết luận
Lan truyền xuôi là "hơi thở" của mạng nơ-ron, nơi dữ liệu thô được nhào nặn qua các ma trận trọng số và các bộ lọc phi tuyến để tạo ra trí tuệ. Toàn bộ kiến trúc học sâu (Deep Learning) thực chất là sự lặp lại của quy trình đơn giản này hàng triệu lần qua nhiều tầng nơ-ron liên kết với nhau. Tuy nhiên, để mô hình thực sự học được, chúng ta cần một cơ chế để điều chỉnh các trọng số này dựa trên sai số dự đoán – tiền đề cho các nghiên cứu về hàm mất mát và lan truyền ngược trong các phần tiếp theo.

---

## Tài liệu tham khảo (Citations)
1. Quy trình lan truyền xuôi và phân tích hàm kích hoạt dựa trên `aero_LL_03_ANN math part 1 (forward prop).md`. Thuyết minh về việc hấp thụ bias vào tích vô hướng và sự chuyển đổi giá trị tuyến tính thành xác suất qua hàm Sigmoid. village.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Học sâu: Perceptron và Kiến trúc Mạng Nơ-ron Nhân tạo (ANN)](aero_llm_01_the_perceptron_and_ann_architecture.md) | [Xem bài viết →](aero_llm_01_the_perceptron_and_ann_architecture.md) |
| [Học sâu: Góc nhìn Hình học về Mạng Nơ-ron Nhân tạo (ANN)](aero_llm_02_a_geometric_view_of_anns.md) | [Xem bài viết →](aero_llm_02_a_geometric_view_of_anns.md) |
| 📌 **[Học sâu: Giải tích ANN Phần 1 – Lan truyền xuôi (Forward Propagation)](aero_llm_03_ann_math_part_1_forward_prop_.md)** | [Xem bài viết →](aero_llm_03_ann_math_part_1_forward_prop_.md) |
| [Học sâu: Giải tích ANN Phần 2 – Sai số, Mất mát và Chi phí (Errors, Loss, Cost)](aero_llm_04_ann_math_part_2_errors_loss_cost_.md) | [Xem bài viết →](aero_llm_04_ann_math_part_2_errors_loss_cost_.md) |
| [Học sâu: Giải tích ANN Phần 3 – Lan truyền ngược (Backpropagation)](aero_llm_05_ann_math_part_3_backprop_.md) | [Xem bài viết →](aero_llm_05_ann_math_part_3_backprop_.md) |
| [Học sâu: Thực thi Lan truyền xuôi trong PyTorch](aero_llm_06_forward_pass_in_pytorch.md) | [Xem bài viết →](aero_llm_06_forward_pass_in_pytorch.md) |
| [Học sâu: Thực thi Lan truyền ngược trong PyTorch](aero_llm_07_backprop_in_pytorch.md) | [Xem bài viết →](aero_llm_07_backprop_in_pytorch.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
