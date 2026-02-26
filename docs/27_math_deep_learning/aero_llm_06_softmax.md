
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
# Toán học trong Học sâu: Hàm Softmax và Diễn giải Xác suất (Softmax)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về hàm Softmax, một phép biến đổi phi tuyến quan trọng trong các bài toán phân loại đa lớp của học sâu. chúng ta phân tích vai trò của số mũ tự nhiên ($e$) trong việc tạo ra các đầu ra không âm, cơ chế chuẩn hóa dữ liệu về dạng phân phối xác suất (tổng bằng 1), và ý nghĩa của việc chuyển đổi các giá trị thô (logits) thành các mức độ tin cậy có thể diễn giải được. Nghiên cứu thực hiện thực nghiệm so sánh phương pháp tính toán thủ công trong NumPy và sử dụng module `torch.nn` trong PyTorch, qua đó làm rõ tính chất co giãn phi tuyến của hàm số đối với các giá trị đầu vào cực biên.

---

## 1. Cơ sở Toán học: Số mũ Tự nhiên ($e$)

Hàm Softmax dựa trên hằng số Euler $e \approx 2.718$. Hai đặc tính của hàm số mũ $e^x$ quyết định tính khả thi của Softmax:
- **Tính Dương tuyệt đối:** $e^x$ luôn lớn hơn 0 với mọi giá trị $x$ (ngay cả khi $x$ âm). Điều này đảm bảo xác suất đầu ra không bao giờ bị âm.
- **Tốc độ Tăng trưởng:** Hàm số mũ khuếch đại các giá trị lớn và thu nhỏ các giá trị nhỏ một cách nhanh chóng, tạo ra sự phân tách rõ rệt giữa các lớp đối tượng.

---

## 2. Công thức và Cơ chế Chuẩn hóa

Giả sử có một tập hợp các số thực $z$, hàm Softmax cho phần tử thứ $i$ được định nghĩa là:

$$

\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}

$$


- **Tử số:** Chuyển đổi giá trị thô sang không gian số mũ.
- **Mẫu số:** Tổng của toàn bộ các giá trị sau khi lấy số mũ, đóng vai trò là hệ số chuẩn hóa.
- **Hệ quả:** Tập hợp đầu ra luôn nằm trong khoảng $(0, 1)$ và có tổng bằng chính xác $1.0$. Đặc tính này cho phép chúng ta coi đầu ra của mạng nơ-ron như một phân phối xác suất.

---

## 3. Diễn giải trong context Học sâu (Logits to Probs)

Các mô hình AI thường xuất ra các con số tùy ý (gọi là logits) không có ý nghĩa trực tiếp. Hàm Softmax đóng vai trò là một "bộ thông dịch":
- **Gán nhãn xác suất:** Chuyển đổi các số điểm thô thành xác suất cho từng danh mục (ví dụ: 0.9 xác suất là mèo, 0.05 là chó).
- **Tính phi tuyến:** Trong thực nghiệm, sự khác biệt nhỏ ở đầu vào (ví dụ từ 1 lên 2) tạo ra sự khác biệt rất lớn ở đầu ra sau khi qua Softmax. Ngược lại, các giá trị âm đều bị ép về gần 0, giúp mô hình tập trung vào các giả thuyết có khả năng cao nhất.

---

## 4. Thực thi Kỹ thuật: NumPy vs PyTorch

### 4.1. NumPy (Tiếp cận Thủ công)
Phép toán có thể thực hiện chỉ với một dòng mã: `np.exp(z) / np.sum(np.exp(z))`. Cách tiếp cận này giúp nhà nghiên cứu nắm vững bản chất toán học nhưng thiếu tối ưu hóa cho các tensor đa chiều phức tạp.

### 4.2. PyTorch (Tiếp cận Hướng đối tượng)
PyTorch cung cấp lớp `nn.Softmax(dim=...)`. Điểm lưu ý quan trọng là tham số `dim`:
- Phải chỉ định rõ chiều nào sẽ được chuẩn hóa (ví dụ `dim=0` cho vectơ hàng).
- PyTorch yêu cầu dữ liệu đầu vào phải là `torch.Tensor`, việc đưa vào một danh sách thông thường (`list`) sẽ dẫn đến lỗi logic.

---

## 5. Kết luận
Hàm Softmax là cầu nối giữa các phép toán đại số thô và ngôn ngữ xác suất của con người. Khả năng biến các tín hiệu điện toán phức tạp thành các phân phối xác suất chuẩn mực giúp các mô hình ngôn ngữ như GPT đưa ra các dự đoán từ kế tiếp một cách logic và có độ tin cậy cao. Việc làm chủ cả công thức toán học và kỹ thuật thực thi trong PyTorch là yêu cầu bắt buộc đối với bất kỳ kỹ sư AI nào.

---

## Tài liệu tham khảo (Citations)
1. Cơ sở toán học và ứng dụng của hàm Softmax trong mạng nơ-ron dựa trên `aero_LL_06_Softmax.md`. Phân tích hàm số mũ tự nhiên, cơ chế chuẩn hóa xác suất và thực nghiệm so sánh NumPy/PyTorch.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Toán học trong Học sâu: Hệ thuật ngữ và Kiểu dữ liệu trong Điện toán (Terms and Datatypes)](aero_llm_01_terms_and_datatypes_in_math_and_computers.md) | [Xem bài viết →](aero_llm_01_terms_and_datatypes_in_math_and_computers.md) |
| [Toán học trong Học sâu: Phép Chuyển vị Vectơ và Ma trận (Transpose)](aero_llm_02_vector_and_matrix_transpose.md) | [Xem bài viết →](aero_llm_02_vector_and_matrix_transpose.md) |
| [Toán học trong Học sâu: Tổ hợp Tuyến tính có Trọng số (Linear Weighted Combinations)](aero_llm_03_linear_weighted_combinations.md) | [Xem bài viết →](aero_llm_03_linear_weighted_combinations.md) |
| [Toán học trong Học sâu: Tích vô hướng (The Dot Product)](aero_llm_04_the_dot_product.md) | [Xem bài viết →](aero_llm_04_the_dot_product.md) |
| [Toán học trong Học sâu: Phép Nhân Ma trận (Matrix Multiplication)](aero_llm_05_matrix_multiplication.md) | [Xem bài viết →](aero_llm_05_matrix_multiplication.md) |
| 📌 **[Toán học trong Học sâu: Hàm Softmax và Diễn giải Xác suất (Softmax)](aero_llm_06_softmax.md)** | [Xem bài viết →](aero_llm_06_softmax.md) |
| [Toán học trong Học sâu: Hàm Logarit và Ứng dụng trong Tối ưu hóa (Logarithms)](aero_llm_07_logarithms.md) | [Xem bài viết →](aero_llm_07_logarithms.md) |
| [Toán học trong Học sâu: Entropy và Cross-Entropy (Entropy)](aero_llm_08_entropy_and_cross_entropy.md) | [Xem bài viết →](aero_llm_08_entropy_and_cross_entropy.md) |
| [Toán học trong Học sâu: Cực trị và Chỉ số Cực trị (Min/Max & Argmin/Argmax)](aero_llm_09_minmax_and_argminargmax.md) | [Xem bài viết →](aero_llm_09_minmax_and_argminargmax.md) |
| [Toán học trong Học sâu: Giá trị Trung bình và Phương sai (Mean and Variance)](aero_llm_10_mean_and_variance.md) | [Xem bài viết →](aero_llm_10_mean_and_variance.md) |
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
