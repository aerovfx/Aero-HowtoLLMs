
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
# Toán học trong Học sâu: Entropy và Cross-Entropy (Entropy)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về lý thuyết thông tin trong học sâu, tập trung vào hai khái niệm cốt lõi: Entropy và Cross-Entropy. chúng ta phân tích Entropy Shannon như một thước đo của sự "bất ngờ" (surprise) hoặc độ bất định trong một hệ thống dữ liệu. Nghiên cứu đi sâu vào cơ chế của Cross-Entropy trong việc đo lường khoảng cách giữa phân phối xác suất thực tế (labels) và phân phối dự đoán của mô hình (predictions). Bằng các thực nghiệm trên NumPy và PyTorch, chúng ta minh chứng cách thức biến đổi các bài toán phân loại thành các bài toán tối ưu hóa thông qua hàm mất mát Binary Cross Entropy (BCE), đồng thời làm rõ các yêu cầu kỹ thuật về định dạng tensor và thứ tự tham biến trong lập trình thực tiễn.

---

## 1. Entropy Shannon: Thước đo Độ bất định

Trong lý thuyết thông tin, Entropy không đại diện cho sự hỗn loạn vật lý mà đại diện cho lượng thông tin hoặc độ khó dự đoán của một biến ngẫu nhiên.
- **Nguyên lý cực đại:** Entropy đạt giá trị cao nhất khi xác suất các sự kiện là tương đương nhau (ví dụ $p=0.5$ trong tung đồng xu), vì khi đó chúng ta hoàn toàn không biết kết quả nào sẽ xảy ra.
- **Nguyên lý cực tiểu:** Khi một sự kiện trở nên chắc chắn ($p=0$ hoặc $p=1$), sự bất ngờ biến mất và Entropy tiến về 0.
- **Công thức:** $H(x) = -\sum p(x) \log p(x)$. Dấu âm giúp đảm bảo giá trị Entropy luôn dương vì logarit của xác suất (từ 0 đến 1) luôn âm.

---

## 2. Cross-Entropy trong Huấn luyện Mô hình

Cross-Entropy là công cụ để so sánh hai phân phối xác suất khác nhau:
- **Phân phối thực tế ($p$):** Thường là các nhãn (labels) dạng "one-hot" (ví dụ: [1, 0] cho mèo).
- **Phân phối dự đoán ($q$):** Là đầu ra của hàm Softmax từ mô hình (ví dụ: [0.9, 0.1]).
- **Mục tiêu tối ưu:** Cực tiểu hóa Cross-Entropy đồng nghĩa với việc đẩy dự đoán của mô hình ($q$) tiến sát về phía sự thật khách quan ($p$). Khi mô hình dự đoán chính xác tuyệt đối, Cross-Entropy sẽ đạt giá trị tối thiểu.

---

## 3. Binary Cross Entropy (BCE) và Sự đơn giản hóa

Đối với các bài toán phân loại nhị phân (có/không, mèo/chó), công thức Cross-Entropy được đơn giản hóa thành:

$$

BCE = -[p \log(q) + (1-p) \log(1-q)]

$$


Trong thực tế học sâu, vì $p$ thường chỉ bằng 0 hoặc 1, công thức này lại càng đơn giản hơn: nó chỉ đơn thuần là giá trị âm logarit của xác suất mà mô hình gán cho lớp đúng. Nếu mô hình càng tự tin vào lớp đúng, giá trị mất mát (loss) càng nhỏ.

---

## 4. Thực thi Kỹ thuật trên PyTorch

Việc sử dụng PyTorch yêu cầu sự chính xác cao về cú pháp:
- **Hàm `F.binary_cross_entropy`:** Yêu cầu tham số đầu tiên là dự đoán từ mô hình và tham số thứ hai là nhãn thực tế. Việc đảo ngược thứ tự này sẽ dẫn đến kết quả sai lệch nghiêm trọng.
- **Quản lý Tensor:** PyTorch không chấp nhận danh sách Python (`list`) thông thường cho các phép toán này. Dữ liệu phải được chuyển đổi thành `torch.Tensor` trước khi tính toán.
- **Tính ổn định số học:** PyTorch thường tích hợp sẵn các kỹ thuật xử lý để tránh lỗi khi $\log(0)$ (giá trị không xác định), giúp quá trình huấn luyện diễn ra trơn tru ngay cả khi mô hình đưa ra dự đoán cực đoan.

---

## 5. Kết luận
Entropy và Cross-Entropy là "ngôn ngữ" để đo lường sự thông minh của một mô hình. Một mô hình càng học tốt thì Cross-Entropy giữa dự đoán của nó và thực tế càng thấp. Thấu hiểu các khái niệm này giúp nhà nghiên cứu không chỉ nắm vững cơ chế của các hàm mất mát (loss functions) mà còn có cái nhìn sâu sắc về cách thức mà thông tin được luân chuyển và định lượng bên trong các kiến trúc LLM hiện đại.

---

## Tài liệu tham khảo (Citations)
1. Lý thuyết thông tin Shannon và Cross-Entropy trong học sâu dựa trên `aero_LL_08_Entropy and cross-entropy.md`. Phân tích độ bất định, công thức BCE và thực thi hàm mất mát trong PyTorch.
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
| 📌 **[Toán học trong Học sâu: Entropy và Cross-Entropy (Entropy)](aero_llm_08_entropy_and_cross_entropy.md)** | [Xem bài viết →](aero_llm_08_entropy_and_cross_entropy.md) |
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
