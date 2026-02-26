
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [27 math deep learning](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Toán học trong Học sâu: Tìm Cực trị bằng Đạo hàm (Minima and Maxima)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về phương pháp xác định các điểm cực trị (local minima và local maxima) của hàm số bằng công cụ đạo hàm, một thành phần cốt lõi của thuật toán Hạ giang (Gradient Descent) trong học sâu. chúng ta phân tích khái niệm các "điểm tới hạn" (critical points) nơi đạo hàm bằng 0, đồng thời thiết lập các tiêu chí toán học để phân biệt giữa cực tiểu và cực đại dựa trên dấu của đạo hàm ở các vùng lân cận. Nghiên cứu cũng thảo luận về hiện tượng "biến mất gradient" (vanishing gradient) tại các vùng hàm số không đổi, một thách thức lớn trong việc huấn luyện các mạng nơ-ron đa tầng.

---

## 1. Điểm tới hạn: Nơi Đạo hàm bằng 0

Trong giải tích, các điểm mà tại đó hàm số ngừng tăng hoặc ngừng giảm và bắt đầu đổi hướng được gọi là điểm tới hạn:
- **Nguyên lý:** Tại các đỉnh (cực đại) hoặc đáy (cực tiểu) của một đường cong, tiếp tuyến của đồ thị nằm ngang, nghĩa là độ dốc hay đạo hàm tại đó bằng chính xác 0.
- **Quy trình tìm kiếm:** Để tìm các điểm này, chúng ta tính đạo hàm của hàm mất mát, cho đạo hàm bằng 0 và giải phương trình tìm biến số $x$. Kết quả trả về là tập hợp tất cả các vị trí có tiềm năng là cực trị.

---

## 2. Phân biệt Cực tiểu (Minima) và Cực đại (Maxima)

Mặc dù cả cực tiểu và cực đại đều có đạo hàm bằng 0, chúng có đặc điểm thay đổi độ dốc khác nhau ở hai phía:
- **Cực tiểu (Minima):** Là mục tiêu của học sâu (cực tiểu hóa sai số).
    - Bên trái điểm cực tiểu: Hàm số đang giảm (đạo hàm âm).
    - Bên phải điểm cực tiểu: Hàm số đang tăng (đạo hàm dương).
- **Cực đại (Maxima):**
    - Bên trái điểm cực đại: Hàm số đang tăng (đạo hàm dương).
    - Bên phải điểm cực đại: Hàm số đang giảm (đạo hàm âm).
Việc thấu hiểu sự khác biệt này giúp thuật toán Gradient Descent biết cách điều chỉnh trọng số để luôn hướng về phía "thung lũng" của hàm mất mát thay vì leo lên các "đỉnh núi".

---

## 3. Thách thức từ Vùng phẳng và Vanishing Gradient

Ngoài cực tiểu và cực đại, còn có trường hợp thứ ba nơi đạo hàm bằng 0: **Vùng phẳng (Plateaus)**.
- **Đặc điểm:** Hàm số không đổi hoặc thay đổi cực kỳ chậm trong một khoảng rộng. Tại đây, đạo hàm biến mất (về 0) nhưng chúng ta chưa đạt được điểm tối ưu.
- **Hệ quả trong Deep Learning:** Khi gradient biến mất, mô hình ngừng học vì đạo hàm không còn cung cấp thông tin về hướng cần di chuyển. Đây là vấn đề phổ biến khi sử dụng các hàm kích hoạt như Sigmoid trong các mạng quá sâu.

---

## 4. Ứng dụng trong Thuật toán Hạ giang (Gradient Descent)

Thuật toán Gradient Descent tận dụng thông tin từ đạo hàm để thực hiện các bước di chuyển:
1. Nếu đạo hàm âm: Nghĩa là chúng ta đang ở sườn dốc bên trái cực tiểu, cần tăng $x$ để tiến về đáy.
2. Nếu đạo hàm dương: Nghĩa là chúng ta đang ở sườn dốc bên phải cực tiểu, cần giảm $x$ để lùi về đáy.
Sự tương tác liên tục giữa giá trị đạo hàm và vị trí giúp mô hình dần hội tụ về điểm có sai số thấp nhất có thể.

---

## 5. Kết luận
Tìm kiếm cực đại và cực tiểu không chỉ là bài toán tìm ẩn số, mà là hành trình tìm kiếm sự tối ưu cho trí tuệ nhân tạo. Khả năng phân tích các điểm tới hạn bằng đạo hàm giúp chúng ta định vị được các cấu hình trọng số tốt nhất cho mô hình. Việc nhận diện được các bẫy vùng phẳng và hiểu rõ cơ chế chuyển đổi dấu của đạo hàm là nền tảng để nắm vững các kỹ thuật tối ưu hóa tiên tiến, đảm bảo mô hình LLM có thể học tập hiệu quả từ những dữ liệu phức tạp nhất.

---

## Tài liệu tham khảo (Citations)
1. Phương pháp xác định cực trị và phân tích điểm tới hạn dựa trên `aero_LL_14_Derivatives find minima.md`. Phân tích dấu đạo hàm lân cận, phân biệt cực tiểu/cực đại và thảo luận về hiện tượng vanishing gradient.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Toán học trong Học sâu: Hệ thuật ngữ và Kiểu dữ liệu trong Điện toán (Terms and Datatypes)](aero_llm_01_terms_and_datatypes_in_math_and_computers.md) | [Xem bài viết →](aero_llm_01_terms_and_datatypes_in_math_and_computers.md) |
| [Toán học trong Học sâu: Phép Chuyển vị Vectơ và Ma trận (Transpose)](aero_llm_02_vector_and_matrix_transpose.md) | [Xem bài viết →](aero_llm_02_vector_and_matrix_transpose.md) |
| [Toán học trong Học sâu: Tổ hợp Tuyến tính có Trọng số (Linear Weighted Combinations)](aero_llm_03_linear_weighted_combinations.md) | [Xem bài viết →](aero_llm_03_linear_weighted_combinations.md) |
| [Toán học trong Học sâu: Tích vô hướng (The Dot Product)](aero_llm_04_the_dot_product.md) | [Xem bài viết →](aero_llm_04_the_dot_product.md) |
| [Toán học trong Học sâu: Phép Nhân Ma trận (Matrix Multiplication)](aero_llm_05_matrix_multiplication.md) | [Xem bài viết →](aero_llm_05_matrix_multiplication.md) |
| [Toán học trong Học sâu: Hàm Softmax và Diễn giải Xác suất (Softmax)](aero_llm_06_softmax.md) | [Xem bài viết →](aero_llm_06_softmax.md) |
| [Toán học trong Học sâu: Hàm Logarit và Ứng dụng trong Tối ưu hóa (Logarithms)](aero_llm_07_logarithms.md) | [Xem bài viết →](aero_llm_07_logarithms.md) |
| [Toán học trong Học sâu: Entropy và Cross-Entropy (Entropy)](aero_llm_08_entropy_and_cross_entropy.md) | [Xem bài viết →](aero_llm_08_entropy_and_cross_entropy.md) |
| [Toán học trong Học sâu: Cực trị và Chỉ số Cực trị (Min/Max & Argmin/Argmax)](aero_llm_09_minmax_and_argminargmax.md) | [Xem bài viết →](aero_llm_09_minmax_and_argminargmax.md) |
| [Toán học trong Học sâu: Giá trị Trung bình và Phương sai (Mean and Variance)](aero_llm_10_mean_and_variance.md) | [Xem bài viết →](aero_llm_10_mean_and_variance.md) |
| [Toán học trong Học sâu: Lấy mẫu Ngẫu nhiên và Biến thiên Mẫu (Sampling Variability)](aero_llm_11_random_sampling_and_sampling_variability.md) | [Xem bài viết →](aero_llm_11_random_sampling_and_sampling_variability.md) |
| [Toán học trong Học sâu: Kiểm định T (The T-Test)](aero_llm_12_the_t_test.md) | [Xem bài viết →](aero_llm_12_the_t_test.md) |
| [Toán học trong Học sâu: Trực giác về Đạo hàm và Đa thức (Derivatives)](aero_llm_13_derivatives_intuition_and_polynomials.md) | [Xem bài viết →](aero_llm_13_derivatives_intuition_and_polynomials.md) |
| 📌 **[Toán học trong Học sâu: Tìm Cực trị bằng Đạo hàm (Minima and Maxima)](aero_llm_14_derivatives_find_minima.md)** | [Xem bài viết →](aero_llm_14_derivatives_find_minima.md) |
| [Toán học trong Học sâu: Quy tắc Nhân và Quy tắc Chuỗi (Product & Chain Rules)](aero_llm_15_derivatives_product_and_chain_rules.md) | [Xem bài viết →](aero_llm_15_derivatives_product_and_chain_rules.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
