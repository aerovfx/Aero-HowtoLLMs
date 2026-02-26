
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
# Toán học trong Học sâu: Kiểm định T (The T-Test)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về phương pháp kiểm định giả thuyết thống kê (T-test), một công cụ quan trọng để đánh giá tính hiệu quả của các kiến trúc và tham số trong học sâu. chúng ta phân tích cơ chế so sánh giữa giả thuyết không ($H_0$) và giả thuyết đối ($H_a$), công thức toán học dựa trên hiệu số trung bình chuẩn hóa theo độ lệch chuẩn, và ý nghĩa của giá trị $p$ trong việc xác định sự khác biệt có ý nghĩa thống kê. Nghiên cứu thực hiện các thực nghiệm trên thư viện SciPy để minh chứng cách thức quy trình kiểm định T giúp nhà nghiên cứu đưa ra các quyết định có cơ sở khoa học khi lựa chọn giữa các mô hình AI khác nhau.

---

## 1. Mục tiêu của Kiểm định T trong Deep Learning

Trong quá trình phát triển AI, chúng ta thường đặt câu hỏi: "Kiến trúc mô hình A có thực sự tốt hơn kiến trúc B?". Việc chỉ nhìn vào độ chính xác (accuracy) cao hơn ở một vài lượt chạy là chưa đủ để kết luận.
- **Kiểm định T:** Cho phép xác định xem sự khác biệt về hiệu năng giữa hai phân phối dữ liệu (ví dụ: độ chính xác của 20 lượt chạy mô hình A vs 20 lượt chạy mô hình B) là thực tế hay chỉ là kết quả ngẫu nhiên của biến thiên mẫu.
- **Giả thuyết Không ($H_0$):** Giả định rằng hai mô hình có hiệu năng như nhau. Mọi khác biệt quan sát được chỉ là do ngẫu nhiên.
- **Giả thuyết Đối ($H_a$):** Khẳng định có sự khác biệt thực sự và có ý nghĩa giữa hai mô hình.

---

## 2. Công thức và Cơ chế vận hành

Giá trị $t$ được tính toán dựa trên một nguyên lý đơn giản:

t = \frac{\bar{x} - \bar{y}}{s / \sqrt{n}}

Trong đó:
- **Tử số:** Khoảng cách giữa hai giá trị trung bình.
- **Mẫu số:** Độ lệch chuẩn được chuẩn hóa theo kích thước mẫu (nhiễu).
- **Nguyên lý cốt lõi:** Giá trị $t$ càng lớn khi sự khác biệt giữa các giá trị trung bình càng cao và độ biến thiên (nhiễu) bên trong mỗi nhóm mẫu càng thấp.

---

## 3. Diễn giải Kết quả: Ngưỡng ý nghĩa và Giá trị $p$

Sau khi có giá trị $t$, chúng ta quy đổi nó sang giá trị $p$ (p-value):
- **Ngưỡng 0.05:** Đây là ngưỡng phổ biến nhất trong khoa học. Nếu $p < 0.05$, có ít hơn 5% khả năng sự khác biệt này xảy ra do ngẫu nhiên. Chúng ta bác bỏ $H_0$ và kết luận mô hình có sự cải tiến thực sự.
- **Trường hợp $p $\ge$q 0.05$:** Không đủ bằng chứng để kết luận sự khác biệt. Trong ngữ cảnh học sâu, điều này có nghĩa là kiến trúc mới không mang lại lợi ích thực chất so với kiến trúc cũ, mặc dù con số trung bình có thể trông cao hơn một chút.

---

## 4. Thực thi Kỹ thuật với SciPy

Nghiên cứu sử dụng hàm `stats.ttest_ind()` (Independent Samples T-test) từ thư viện SciPy:
- **Tính độc lập:** Hàm này phù hợp để so sánh hai nhóm dữ liệu không phụ thuộc vào nhau (ví dụ: hai mô hình được huấn luyện hoàn toàn tách biệt).
- **Tính đối xứng:** Dấu của giá trị $t$ (âm hay dương) chỉ phụ thuộc vào thứ tự đưa dữ liệu vào hàm, không ảnh hưởng đến giá trị $p$ và kết luận cuối cùng.
- **Trực quan hóa Dữ liệu:** Sử dụng kỹ thuật "jittering" (thêm nhiễu ngẫu nhiên vào trục X) giúp tách các điểm dữ liệu bị chồng lấp, cho phép quan sát phân phối thực tế một cách trực quan hơn trước khi thực hiện kiểm định.

---

## 5. Kết luận
Kiểm định T là "thanh bảo kiếm" giúp các kỹ sư AI tránh được bẫy của những cải tiến ảo do ngẫu nhiên. Trong thế giới của LLM, nơi mà chi phí huấn luyện cực kỳ đắt đỏ, việc sử dụng các công cụ thống kê như T-test để xác nhận tính hiệu quả của các siêu tham số (hyperparameters) trước khi triển khai quy mô lớn là vô cùng cần thiết. Thấu hiểu T-test là bước đệm để tiến tới những phương pháp so sánh phức tạp hơn như ANOVA hay tính toán kích thước hiệu ứng (effect size).

---

## Tài liệu tham khảo (Citations)
1. Ứng dụng kiểm định T trong so sánh hiệu năng mô hình dựa trên `aero_LL_12_The t-test.md`. Phân tích giả thuyết không, giá trị $p$, công thức thống kê và thực thi kiểm định độc lập trong SciPy.
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
| 📌 **[Toán học trong Học sâu: Kiểm định T (The T-Test)](aero_llm_12_the_t_test.md)** | [Xem bài viết →](aero_llm_12_the_t_test.md) |
| [Toán học trong Học sâu: Trực giác về Đạo hàm và Đa thức (Derivatives)](aero_llm_13_derivatives_intuition_and_polynomials.md) | [Xem bài viết →](aero_llm_13_derivatives_intuition_and_polynomials.md) |
| [Toán học trong Học sâu: Tìm Cực trị bằng Đạo hàm (Minima and Maxima)](aero_llm_14_derivatives_find_minima.md) | [Xem bài viết →](aero_llm_14_derivatives_find_minima.md) |
| [Toán học trong Học sâu: Quy tắc Nhân và Quy tắc Chuỗi (Product & Chain Rules)](aero_llm_15_derivatives_product_and_chain_rules.md) | [Xem bài viết →](aero_llm_15_derivatives_product_and_chain_rules.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
