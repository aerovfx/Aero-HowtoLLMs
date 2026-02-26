
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [27 math deep learning](index.md)

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
# Toán học trong Học sâu: Giá trị Trung bình và Phương sai (Mean and Variance)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về các thước đo xu hướng tập trung và độ phân tán của dữ liệu, hai khái niệm nền tảng trong việc chuẩn hóa (normalization) và điều chỉnh (regularization) các mô hình học sâu. chúng ta phân tích công thức toán học của giá trị trung bình (mean) và phương sai (variance), giải mã lý do tại sao các khoảng cách đến số trung bình cần được bình phương, và sự khác biệt giữa phương sai hiệu chỉnh (unbiased) và không hiệu chỉnh (biased). Nghiên cứu thực hiện các thực nghiệm trên NumPy để làm rõ tham số `ddof` (degrees of freedom), đồng thời thiết lập mối liên hệ giữa các thước đo này với các kỹ thuật chính quy hóa L1 và L2 trong huấn luyện mạng nơ-ron.

---

## 1. Giá trị Trung bình: Thước đo Xu hướng Tập trung

Giá trị trung bình ($\bar{x}$ hoặc $\mu$) là con số đại diện cho "điểm tựa" của một phân phối dữ liệu.
- **Tính toán:** Tổng các giá trị chia cho số lượng phần tử ($n$).
- **Phạm vi ứng dụng:** Hiệu quả nhất đối với dữ liệu có phân phối chuẩn (Gaussian). Đối với các phân phối lệch (như thu nhập dân cư) hoặc phân phối hai đỉnh (bimodal), giá trị trung bình có thể không phản ánh chính xác thực tế, khi đó các thước đo như trung vị (median) sẽ được xem xét.

---

## 2. Phương sai: Thước đo Độ phân tán

Phương sai ($\sigma^2$) đo lường mức độ "trải rộng" của dữ liệu xung quanh giá trị trung bình.
- **Cơ chế Bình phương:** Việc bình phương các hiệu số $($x_i$ - \bar{x})^2$ phục vụ hai mục đích:
    1. Triệt tiêu các giá trị âm (đảm bảo phương sai luôn dương).
    2. Khuếch đại tầm ảnh hưởng của các giá trị ngoại lai (outliers), giúp mô hình nhạy bén hơn với các sai số lớn.
- **So sánh với MAD:** Xu hướng sử dụng giá trị tuyệt đối thay vì bình phương dẫn đến thước đo **Mean Absolute Difference (MAD)**. Trong học sâu, MAD là cơ sở của chính quy hóa L1, trong khi Phương sai là cơ sở của chính quy hóa L2.

---

## 3. Độ lệch chuẩn (Standard Deviation)

Độ lệch chuẩn ($\sigma$) đơn giản là căn bậc hai của phương sai. Lợi thế của nó là có cùng đơn vị đo lường với dữ liệu gốc, giúp việc diễn giải và trực quan hóa trở nên trực quan hơn trên các biểu đồ phân phối.

---

## 4. Thực thi Kỹ thuật và Bẫy lập trình trong NumPy

### 4.1. Vấn đề bậc tự do (Degrees of Freedom)
Trong thống kê, phương sai hiệu chỉnh (không chệch) yêu cầu chia cho $n-1$ thay vì $n$. Điều này giúp loại bỏ sai số hệ thống khi ta ước lượng phương sai của quần thể từ một mẫu nhỏ.

### 4.2. Tham số `ddof` trong NumPy

$$
Mặc định, hàm `np.var()` chia cho n (`ddof=0`). Để có kết quả thống kê chuẩn xác (unbiased), lập trình viên phải chỉ định `ddof=1`.
$$

- **Lưu ý thực tiễn:** Trong học sâu, do kích thước tập dữ liệu (batch size) thường rất lớn, sự khác biệt giữa việc chia cho $n$ hay $n-1$ trở nên không đáng kể. Tuy nhiên, việc hiểu rõ tham số này là dấu hiệu của một kỹ sư AI có nền tảng toán học vững chắc.

---

## 5. Kết luận
Giá trị trung bình và phương sai không chỉ là các khái niệm thống kê mô tả mà là công cụ để "thuần hóa" dữ liệu. Việc đưa dữ liệu về trạng thái có trung bình bằng 0 và phương sai bằng 1 (Standardization) là bước đi tiên quyết giúp các thuật toán tối ưu hóa như Gradient Descent hội tụ nhanh hơn. Thấu hiểu bản chất của bình phương và giá trị tuyệt đối trong các thước đo này sẽ giúp nhà nghiên cứu lựa chọn đúng phương pháp chính quy hóa để ngăn chặn hiện tượng quá khớp (overfitting) trong các mô hình LLM.

---

## Tài liệu tham khảo (Citations)
1. Cơ sở thống kê về xu hướng tập trung và độ phân tán dựa trên `aero_LL_10_Mean and variance.md`. Phân tích công thức $\mu$ và $\sigma^2$, vai trò của bình phương trong tối ưu hóa và thực thi `ddof` trong NumPy.
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
| 📌 **[Toán học trong Học sâu: Giá trị Trung bình và Phương sai (Mean and Variance)](aero_llm_10_mean_and_variance.md)** | [Xem bài viết →](aero_llm_10_mean_and_variance.md) |
| [Toán học trong Học sâu: Lấy mẫu Ngẫu nhiên và Biến thiên Mẫu (Sampling Variability)](aero_llm_11_random_sampling_and_sampling_variability.md) | [Xem bài viết →](aero_llm_11_random_sampling_and_sampling_variability.md) |
| [Toán học trong Học sâu: Kiểm định T (The T-Test)](aero_llm_12_the_t_test.md) | [Xem bài viết →](aero_llm_12_the_t_test.md) |
| [Toán học trong Học sâu: Trực giác về Đạo hàm và Đa thức (Derivatives)](aero_llm_13_derivatives_intuition_and_polynomials.md) | [Xem bài viết →](aero_llm_13_derivatives_intuition_and_polynomials.md) |
| [Toán học trong Học sâu: Tìm Cực trị bằng Đạo hàm (Minima and Maxima)](aero_llm_14_derivatives_find_minima.md) | [Xem bài viết →](aero_llm_14_derivatives_find_minima.md) |
| [Toán học trong Học sâu: Quy tắc Nhân và Quy tắc Chuỗi (Product & Chain Rules)](aero_llm_15_derivatives_product_and_chain_rules.md) | [Xem bài viết →](aero_llm_15_derivatives_product_and_chain_rules.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
