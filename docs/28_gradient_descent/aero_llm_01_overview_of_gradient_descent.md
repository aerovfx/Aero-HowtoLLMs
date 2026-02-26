
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
# Học sâu: Tổng quan về Thuật toán Hạ giang (Gradient Descent)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về Gradient Descent, thuật toán nền tảng nhất trong lĩnh vực học sâu và tối ưu hóa hiện đại. chúng ta phân tích cơ chế học tập của mô hình thông qua ba bước: dự đoán ngẫu nhiên, tính toán sai số và điều chỉnh tham số dựa trên đạo hàm. Nghiên cứu giải mã trực giác hình học của việc "đi xuống" trên bề mặt hàm mất mát (loss landscape), vai trò then chốt của tốc độ học (learning rate), và sự biến đổi từ khái niệm đạo hàm một chiều sang gradient đa chiều. Kết quả thực nghiệm minh chứng rằng mặc dù thuật toán có tính hội tụ cao, nó vẫn đối mặt với các thách thức về độ chính xác tuyệt đối và các bẫy cực trị địa phương.

---

## 1. Cơ chế Học tập của Mạng Nơ-ron

Quá trình "học" của một mô hình AI thực chất là một chuỗi các bước lặp toán học:
1. **Khởi tạo:** Mô hình đưa ra một dự đoán hoàn toàn ngẫu nhiên về dữ liệu (ví dụ: nhầm lẫn giữa ảnh con mèo và bánh sandwich).
2. **Tính toán Sai số:** Sử dụng một hàm mất mát (loss function) để đo lường khoảng cách giữa dự đoán và thực tế.
3. **Điều chỉnh (Gradient Descent):** Đây là bước quan trọng nhất, nơi mô hình sử dụng đạo hàm để biết cần thay đổi các tham số (weights) theo hướng nào và bao nhiêu để sai số nhỏ hơn ở lần lặp sau.

---

## 2. Định nghĩa và Trực quan hóa Gradient Descent

- **Gradient (Độ dốc):** Là đạo hàm mở rộng trong không gian đa chiều. Nó chỉ ra hướng mà hàm số tăng nhanh nhất.
- **Descent (Hạ giang):** Có nghĩa là đi xuống. Thuật toán sẽ di chuyển ngược hướng với gradient để tìm kiếm điểm thấp nhất của hàm mất mát.
- **Trực quan:** Hãy tưởng tượng bạn đang ở trên một đỉnh núi đầy sương mù và muốn tìm đường xuống thung lũng (nơi có lỗi thấp nhất). Bạn sẽ cảm nhận độ dốc dưới chân và bước theo hướng dốc xuống. Mỗi bước đi chính là một lượt cập nhật tham số của mô hình.

---

## 3. Quy tắc cập nhật và Tốc độ học (Learning Rate)

Công thức cốt lõi của việc cập nhật tham số là:

$$

W_{mới} = W_{cũ} - \eta \cdot \frac{df}{dw}

$$

Trong đó:
- **$\frac{df}{dw}$**: Đạo hàm của hàm mất mát tại vị trí hiện tại.
- **$\eta$ (Learning Rate):** Một hệ số nhỏ (ví dụ 0.01) dùng để kiểm soát kích thước bước đi. Nếu bước đi quá lớn, bạn có thể nhảy qua khỏi thung lũng; nếu quá nhỏ, quá trình học sẽ diễn ra cực kỳ chậm chạp.

---

## 4. Những giới hạn và Thách thức

Mặc dù mạnh mẽ, Gradient Descent không phải là một công cụ hoàn hảo:
1. **Độ chính xác:** Thuật toán thường hội tụ về một giá trị rất gần nhưng không nhất thiết trùng khớp tuyệt đối với nghiệm thực tế sau một số lượng vòng lặp hữu hạn.
2. **Cực trị địa phương (Local Minima):** Mô hình có thể bị kẹt ở một "hố nhỏ" trên sườn núi thay vì xuống được thung lũng sâu nhất (Global Minimum).
3. **Vấn đề Gradient:** Các hiện tượng gradient biến mất (vanishing) hoặc bùng nổ (exploding) có thể làm tê liệt quá trình huấn luyện.

---

## 5. Kết luận
Gradient Descent là "trái tim" của mọi kiến trúc LLM hiện đại. Thấu hiểu thuật toán này không chỉ giúp chúng ta giải thích được cách thức máy tính học tập mà còn cung cấp nền tảng để tùy chỉnh các siêu tham số trong quá trình huấn luyện. Dù vẫn tồn tại những rào cản toán học, nhưng sự phát triển của các kỹ thuật bổ trợ trong hai thập kỷ qua đã biến Gradient Descent thành một công cụ vận hành ổn định và hiệu quả, cho phép chúng ta xây dựng những hệ thống trí tuệ nhân tạo có khả năng xử lý những bài toán có hàng tỷ biến số.

---

## Tài liệu tham khảo (Citations)
1. Nguyên lý vận hành và công thức cập nhật của Gradient Descent dựa trên `aero_LL_01_Overview of gradient descent.md`. Phân tích bước lặp tối ưu, vai trò của learning rate và các hạn chế về hội tụ trong học sâu. village.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[Học sâu: Tổng quan về Thuật toán Hạ giang (Gradient Descent)](aero_llm_01_overview_of_gradient_descent.md)** | [Xem bài viết →](aero_llm_01_overview_of_gradient_descent.md) |
| [Học sâu: Vấn đề Cực trị Địa phương (Local Minima)](aero_llm_02_what_about_local_minima.md) | [Xem bài viết →](aero_llm_02_what_about_local_minima.md) |
| [Học sâu: Thực thi Hạ giang trong Không gian 1 Chiều (1D Gradient Descent)](aero_llm_03_gradient_descent_in_1d.md) | [Xem bài viết →](aero_llm_03_gradient_descent_in_1d.md) |
| [Học sâu: Hạ giang trong Không gian 2 Chiều (2D Gradient Descent)](aero_llm_04_gradient_descent_in_2d.md) | [Xem bài viết →](aero_llm_04_gradient_descent_in_2d.md) |
| [Học sâu: Thử thách Lập trình – Tốc độ học Cố định vs. Động (Fixed vs. Dynamic Learning Rate)](aero_llm_05_codechallenge_fixed_vs_dynamic_learning_rate.md) | [Xem bài viết →](aero_llm_05_codechallenge_fixed_vs_dynamic_learning_rate.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
