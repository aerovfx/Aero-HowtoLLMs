
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [10 identifying circuits](index.md)

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
# Rạch Ròi Giới Tính (Him vs Her) Bằng Generalized Eigendecomposition Trong MLP

## Tóm tắt (Abstract)
Thách thức lớn nhất khi áp dụng Generalized Eigendecomposition (GED) trên Mạng ngôn ngữ Lớn (LLMs) nằm ở Vấn đề Khủng hoảng Không gian Đa chiều: Quá nhiều Biến số (Neurons) nhưng Cấp bậc thứ hạng dữ liệu (Rank) lại quá thấp, khiến hàm ma trận không thể tự nghịch đảo. Báo cáo này đưa ra phương thức "Nén rồi Tách" (Two-stage Compression-Separation Procedure) bằng cách ép phẳng Không gian $3072$ chiều thành $63$ chiều thông qua PCA, sau đó mới dùng thuật GED có Shrinkage Regularization. Kết quả trên bộ test-set Câu Điều kiện Đại Từ (Pronouns dataset) cho thấy thuật toán tách đôi và khoanh vùng độc lập thành công Lưới kích hoạt dành riêng cho từ 'Him' so diện với Không gian dành riêng cho 'Her' ngay tại MLP Expansion. 

---

## 1. Mở Đầu (Introduction)
Từ Demo mô phỏng ở chương trước, chúng ta đã nắm được GED hoạt động bằng cách làm Phép chia Tỉ Lệ (SNR) giữa Khối Cầu Tín hiệu và Khối Cầu Đối Chiếu. Khi áp dụng thẳng vào Không gian Thực của Mô hình Transformers, tức $\sim 3072$ chiều Nơ-ron trên hàng ngàn Lớp Token, máy tính sẽ báo lỗi Vô Hiệu Lệnh (Rank Deficient) do $R^{-1}$ tiến tới vô cực. 
Nhiệm vụ của bài thí nghiệm là phải Cắt Bỏ Mỡ Thừa (Data Compression) cho bộ Data trước khi đưa vào Máy vắt GED: Thay vì thao tác trên 3072 tham số rỗng, ta thu bé nó lại về một Nhóm Nhỏ Đại Diện nhưng vẫn chứa $99\%$ năng lượng biến thiên của dữ liệu. 

---

## 2. Tiết Thiết Lập Cấu Trúc Khối Nén Kép (Methodology)

### 2.1. Giải Thuật Hai Giai Đoạn (Two-stage Separation Procedure)
Khi $\text{Rank} \ll \text{Size}$, Phép Tính Eigendecomposition trở nên bất ổn tột độ. Ta thi hành "PCA lọc Nền":
1. Trích xuất Activations kích thước `[N_Mẫu_câu, 3072_Neurons]`. 
2. Chạy **PCA** trên Ma Trận Trung Bình (Average Covariance Matrix) của cả Hai Dữ Kho (Cả HIM và HER gộp chung). Tại sao? Để PCA đi lùng sục **"Toàn bộ vùng không gian chung mà Cả hai đối tượng này cùng kích hoạt"**, lọc lấy các Phân mảnh Chính mang tính sống còn.
3. Cắt Lát (Scree Plot Cut-off): Chỉ giữ lại các PC gộp đủ $99\%$ lượng Variance (Lệch chuẩn) cùa toàn đồ thị. Ví dụ ở đây ta thu về Nhóm Tinh Túy $63$ Mạch $PC$.
4. **Chiếu Rút Chiều:** Phóng (Project) khối Dữ liệu Gốc lên không gian 63 chiều mới này để "Xóa sổ 3000 chiều Rác".

### 2.2. Trực Khán Với Shrinkage (Shrinkage Regularized GED)

$$
Tuyển 63-Dimension Matrix mới có vẻ bé, nhưng bản thân nó vẫn bị Vướng Rank Zero! Nghĩa là \text{Rank}(Cov) = 52 < 63.
$$

$$
Áp dụng cơ chế Covariance Shrinking 1\% (\gamma = 0.01):
$$

$$
\tilde{\mathbf{R}} = (1 - 0.01)\mathbf{R} + 0.01 \alpha \mathbf{I}
$$

$$
Phép toán này biến hóa Rank 52 \xrightarrow{Inflate} 63 (Full Rank). Lúc này hàm vi phân của SciPy (`scipy.linalg.eigh`) có thể tiêu hóa ma trận R_{her\_shrunk}^{-1} \cdot S_{him} hoàn toàn trơn tru. --- ## 3. Khảo Sát Tách Mạch Căn Giới (Analysis) ### 3.1. Sự Trỗi Dậy Của Thành Phần Phân Cực Tuyệt Đối (Top Eigenvector) Khi GED hoàn tất, hệ số Trị Riêng (Eigenvalues) được sắp xếp từ cao xuống thấp. Top 1 Eigenvalue cho thấy có một Vectơ đặc biệt (Eigenvector) mà khi dữ liệu chiếu vào: - Nó Tràn Đầy Năng lượng (Tạo Max Variance) khi dữ liệu mang chữ HIM. - Nó Triệt Tiêu Năng lượng (Chìm nghỉm thành Zero Variance) khi dữ liệu mang chữ HER.
$$

$$
(Và khi đảo \mathbf{S=Her}, \mathbf{R=Him}, ta lại thấy điều ngược lại hoạt động song song).
$$
