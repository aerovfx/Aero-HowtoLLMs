
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
# Học sâu: Góc nhìn Hình học về Mạng Nơ-ron Nhân tạo (ANN)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về cách tiếp cận hình học để thấu hiểu cơ chế vận hành của mạng nơ-ron nhân tạo (ANN) và mô hình Perceptron. chúng ta phân tích các khái niệm về không gian đặc trưng (feature space), nơi mỗi quan sát được đại diện như một điểm tọa độ, và các siêu phẳng phân tách (separating hyperplanes) đóng vai trò là ranh giới quyết định (decision boundary). Nghiên cứu cũng thực hiện phân biệt giữa các loại dự đoán rời rạc $categorical/binary$ và dự đoán liên tục $numeric/continuous$, đồng thời giải mã cách mô hình chuyển đổi từ các đầu vào đa chiều sang các kết quả dự đoán có ý nghĩa trong thế giới thực.

---

## 1. Không gian Đặc trưng (Feature Space)

Mọi bài toán học sâu đều bắt đầu bằng việc chuyển đổi dữ liệu thực tế thành các con số trong một không gian hình học:
- **Định nghĩa:** Không gian đặc trưng là một hệ trục tọa độ nơi mỗi trục đại diện cho một tính chất (feature) của dữ liệu.
- **Ví dụ thực tiễn:** Để dự đoán kết quả thi của sinh viên, chúng ta có hai trục: số giờ học ($x_1$) và số giờ ngủ ($x_2$). Mỗi sinh viên sẽ là một điểm tọa độ $(x_1, x_2)$ trong không gian 2 chiều này.
- **Tính đa chiều:** Trong các bài toán phức tạp, không gian này có thể lên đến hàng nghìn hoặc hàng triệu chiều, nơi mỗi chiều là một đặc trưng riêng biệt mà mô hình cần xử lý.

---

## 2. Siêu phẳng Phân tách và Ranh giới Quyết định

Mục tiêu của việc huấn luyện mạng nơ-ron là tìm ra một ranh giới tối ưu để phân loại dữ liệu trong không gian đặc trưng:
- **Separating Hyperplane (Siêu phẳng phân tách):** 
    - Trong không gian 2D, nó là một đường thẳng.
    - Trong không gian 3D, nó là một mặt phẳng.
    - Trong không gian n-chiều ($n > 3$), nó được gọi là siêu phẳng.
- **Decision Boundary (Ranh giới quyết định):** Đây là "lãnh giới" mà mô hình dựa vào để đưa ra kết luận. Ví dụ: những sinh viên nằm phía trên đường ranh giới được dự đoán là "Đỗ", và những người nằm phía dưới là "Trượt".

---

## 3. Các loại Hình thái Dự đoán

Mạng nơ-ron có thể được thiết kế để đưa ra hai loại kết quả chính tùy thuộc vào bản chất của bài toán:

### 3.1. Dự đoán Rời rạc $Discrete/Categorical$
- **Đặc điểm:** Kết quả thuộc về các nhóm cố định (ví dụ: Đỗ/Trượt, Chó/Mèo, Tích cực/Tiêu cực).
- **Hình học:** Được đại diện bởi việc băm nhỏ không gian đặc trưng thành các vùng riêng biệt bởi các siêu phẳng.

### 3.2. Dự đoán Liên tục $Numeric/Continuous$
- **Đặc điểm:** Kết quả là một con số thực trên một dải giá trị (ví dụ: điểm thi từ 0-100%, giá nhà, nhiệt độ).
- **Hình học:** Đòi hỏi thêm một trục tọa độ thứ ba (hoặc n+1) để biểu diễn giá trị dự đoán. Thay vì chỉ phân tách không gian, mô hình lúc này cố gắng tìm một "bề mặt" (surface) sao cho khoảng cách từ các điểm dữ liệu thực tế đến bề mặt đó là nhỏ nhất.

---

## 4. Ý nghĩa của việc Học Trọng số

Trong góc nhìn hình học, việc điều chỉnh các trọng số ($w_1, w_2, ...$) và bias ($b$) thực chất là các thao tác:
- **Xoay:** Thay đổi trọng số khiến đường ranh giới xoay quanh không gian để tìm hướng phân tách tốt nhất.
- **Dịch chuyển:** Thay đổi bias giúp dịch chuyển ranh giới ra khỏi gốc tọa độ để khớp với vị trí thực của các cụm dữ liệu.
Quá trình Gradient Descent chính là "người dẫn đường" giúp mô hình thực hiện các thao tác xoay và dịch chuyển này cho đến khi ranh giới phân tách được dữ liệu một cách chính xác nhất.

---

## 5. Kết luận
Góc nhìn hình học giúp chúng ta thoát khỏi những con số khô khan để thấy được bản chất của học sâu là quá trình phân cắt và biến đổi không gian. Dù là dự đoán kết quả thi đơn giản hay xử lý ngôn ngữ tự nhiên phức tạp trong các hệ thống LLM, mọi thứ đều có thể được quy về việc tìm kiếm các siêu phẳng tối ưu trong không gian đặc trưng cao chiều. Thấu hiểu hình học của ANN là chìa khóa để thiết kế các kiến trúc mạng hiệu quả và giải thích được tại sao mô hình lại đưa ra những quyết định nhất định.

---

## Tài liệu tham khảo (Citations)
1. Phân tích không gian đặc trưng và ranh giới quyết định dựa trên `aero_LL_02_A geometric view of ANNs.md`. Thuyết minh về sự khác biệt giữa dự đoán rời rạc và liên tục trong không gian đa chiều của mạng nơ-ron. village.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Học sâu: Perceptron và Kiến trúc Mạng Nơ-ron Nhân tạo (ANN)](aero_llm_01_the_perceptron_and_ann_architecture.md) | [Xem bài viết →](aero_llm_01_the_perceptron_and_ann_architecture.md) |
| 📌 **[Học sâu: Góc nhìn Hình học về Mạng Nơ-ron Nhân tạo (ANN)](aero_llm_02_a_geometric_view_of_anns.md)** | [Xem bài viết →](aero_llm_02_a_geometric_view_of_anns.md) |
| [Học sâu: Giải tích ANN Phần 1 – Lan truyền xuôi (Forward Propagation)](aero_llm_03_ann_math_part_1_forward_prop_.md) | [Xem bài viết →](aero_llm_03_ann_math_part_1_forward_prop_.md) |
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
