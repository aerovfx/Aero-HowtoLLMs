
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [10 Identifying circuits](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
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
2. Chạy **PCA** trên Ma Trận Trung Bình (Ave18-RAGe Covariance Matrix) của cả Hai Dữ Kho (Cả HIM và HER gộp chung). Tại sao? Để PCA đi lùng sục **"Toàn bộ vùng không gian chung mà Cả hai đối tượng này cùng kích hoạt"**, lọc lấy các Phân mảnh Chính mang tính sống còn.
3. Cắt Lát (Scree Plot Cut-off): Chỉ giữ lại các PC gộp đủ $99\%$ lượng Variance (Lệch chuẩn) cùa toàn đồ thị. Ví dụ ở đây ta thu về Nhóm Tinh Túy $63$ Mạch $PC$.
4. **Chiếu Rút Chiều:** Phóng (Project) khối Dữ liệu Gốc lên không gian 63 chiều mới này để "Xóa sổ 3000 chiều Rác".

### 2.2. Trực Khán Với Shrinkage (Shrinkage Regularized GED)
Tuyển 63-Dimension Matrix mới có vẻ bé, nhưng bản thân nó vẫn bị Vướng Rank Zero! Nghĩa là $\text{Rank}(Cov) = 52 < 63$. 
Áp dụng cơ chế Covariance Shrinking $1\%$ ($\gamma = 0.01$):
$$ \tilde{\mathbf{R}} = (1 - 0.01)\mathbf{R} + 0.01 \alpha \mathbf{I} $$
Phép toán này biến hóa Rank $52 \xrightarrow{Inflate} 63$ (Full Rank). Lúc này hàm vi phân của SciPy (`scipy.linalg.eigh`) có thể tiêu hóa ma trận $R_{her\_shrunk}^{-1} \cdot S_{him}$ hoàn toàn trơn tru.

---

## 3. Khảo Sát Tách Mạch Căn Giới (Analysis)

### 3.1. Sự Trỗi Dậy Của Thành Phần Phân Cực Tuyệt Đối (Top Eigenvector)
Khi GED hoàn tất, hệ số Trị Riêng (Eigenvalues) được sắp xếp từ cao xuống thấp. Top 1 Eigenvalue cho thấy có một Vectơ đặc biệt (Eigenvector) mà khi dữ liệu chiếu vào:
- Nó Tràn Đầy Năng lượng (Tạo Max Variance) khi dữ liệu mang chữ $HIM$.
- Nó Triệt Tiêu Năng lượng (Chìm nghỉm thành Zero Variance) khi dữ liệu mang chữ $HER$.
(Và khi đảo $\mathbf{S=Her}, \mathbf{R=Him}$, ta lại thấy điều ngược lại hoạt động song song).
Do không có điều kiện ràng buộc Trực Giao (Orthogonality), Vectơ tìm thấy đã "thẩm thấu lách mình" một cách uyển chuyển theo dọc chiều Phân Lớp Giới Tính chứ không bị ép xoay 90 độ cứng ngắc như PCA.

### 3.2. Hiệu Ứng Loại Cừu Khỏi Bầy Xói (Sentence Contrast Validation)
Kiểm chứng tính "Chuyên biệt" (Selectivity) của Vector này: Ta dùng bộ Vectơ Tách HIM ném chồng lên toàn bộ Trục Kích Hoạt của MỘT câu chữ dài (Bao gồm các từ không liên quan: The, dog, was...).
Kết quả đáng kinh ngạc:
- Đối với hầu hết các từ như The, Dog: Năng lượng Kích Hoạt $\dots$ Tương đương nhau (Rất thấp do bị lọc Nhiễu).
- Chỉ riêng tại Ngưỡng cắt Token vị trí Đại Từ ("Him" hoặc "Her"), đồ thị có cú xé toạc thẳng đứng: Cùng một bộ lọc, Chữ HIM văng đỉnh cực đại, Chữ HER thụt đáy tận cùng. Hiện tượng lật Mặt Phiến (Flip Activation) này xác nhận ta đã Tách thành công Mạch Nội Suy Giới tính Cô lập độc lập hoàn toàn khỏi hệ thống cấu trúc cú pháp nền (Grammar Base).

---

## 4. Kết Luận
Việc áp đặt thẳng thuật toán GED lên Dữ liệu Khổng Lồ LLM là tự sát mô hình. Nhưng thông qua Chiến thuật Nén Không gian $\to$ Áp dụng Điều tiết Shrinkage, bài toán chia cắt Cấu Tính Mạch Từ Vựng (Ví dụ Tách biệt giới tính Đại Từ) giữa hàng Ngàn Nơ-ron MLP trở thành hiện thực rực rỡ và dễ dàng truy vết. Phương pháp GED bộc lộ tính Vượt Trội so với SAEs hay Logistic Regression ở điểm nó "Phân Rã Hai Đối Thủ Không Gian" ra một cách Trực Quan cực đại mà không cần một Biến Đích Lable Label khắt khe. 

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm "Two-stage compression + GED" bóc tách Đại từ Giới tính từ file `aero_LLM_12_Generalized eigendecomposition separates him from her in MLP.md`. Giải phẫu hiện tượng Tràn Rank-Deficient và cách hồi sinh thành Full-Rank bằng $\gamma=0.01$ Shrinkage parameter.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Mạng Mạch Thuật Toán (Circuits) Trong Mô Hình Học Sâu](aero_LLM_01_What is a circuit in a DL model.md) | [Xem bài viết →](aero_LLM_01_What is a circuit in a DL model.md) |
| [Cô Lập Và Thăm Dò Khối Chú Ý (Attention Heads)](aero_LLM_02_Isolating and investigating attention heads.md) | [Xem bài viết →](aero_LLM_02_Isolating and investigating attention heads.md) |
| [Thử Thách Lập Trình: Biểu Diễn Phân Bố Nhiệt Laminar Của Trọng Số Chú Ý](aero_LLM_03_CodeChallenge Laminar profile of attention head weights.md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Laminar profile of attention head weights.md) |
| [Khảo Sát Tương Quan Cụm (Clustering) Vi Mạch (Circuits) Trong Không Gian Giảm Chiều](aero_LLM_04_Are circuits clustered in low-dimensional space.md) | [Xem bài viết →](aero_LLM_04_Are circuits clustered in low-dimensional space.md) |
| [Lý Thuyết Và Ứng Dụng Của Kỹ Thuật Dò Thưa (Sparse Probing)](aero_LLM_05_Sparse probing theory and code.md) | [Xem bài viết →](aero_LLM_05_Sparse probing theory and code.md) |
| [Thách Thức Của Tín Hiệu Thưa Trong Dữ Liệu Tập Lớn (Statistical Suppression)](aero_LLM_06_Challenges with sparse logistic regression in large datasets.md) | [Xem bài viết →](aero_LLM_06_Challenges with sparse logistic regression in large datasets.md) |
| [Biến Tiềm Ẩn (Latent) Và Biến Hiển Ngôn (Manifest) Trong Giải Diễn AI](aero_LLM_07_Latent vs. manifest variables.md) | [Xem bài viết →](aero_LLM_07_Latent vs. manifest variables.md) |
| [Mô Hình Sparse Autoencoders (SAEs): Lý Thuyết Và Kiến Trúc Khôi Phục Vi Mạch Tiềm Ẩn](aero_LLM_08_Sparse autoencoders theory and code.md) | [Xem bài viết →](aero_LLM_08_Sparse autoencoders theory and code.md) |
| [Huấn Luyện Sparse Autoencoder Trích Xuất Khái Niệm Ngữ Cảnh Palinka Trên GPT-2](aero_LLM_09_SAE in GPT2 learns about Hungarian Palinka.md) | [Xem bài viết →](aero_LLM_09_SAE in GPT2 learns about Hungarian Palinka.md) |
| [Khảo Sát Phân Tầng Kích Hoạt (Laminar Profile) Qua Sparse Autoencoder](aero_LLM_10_CodeChallenge Laminar profile of autoencoder sparsity.md) | [Xem bài viết →](aero_LLM_10_CodeChallenge Laminar profile of autoencoder sparsity.md) |
| [Nhận Diện Khái Niệm Xuyên Tâm Với Phân Rã Giá Trị Riêng Suy Rộng (Generalized Eigendecomposition - GED)](aero_LLM_11_Non-orthogonal latent components via eigendecomposition (theory and demo).md) | [Xem bài viết →](aero_LLM_11_Non-orthogonal latent components via eigendecomposition (theory and demo).md) |
| 📌 **[Rạch Ròi Giới Tính (Him vs Her) Bằng Generalized Eigendecomposition Trong MLP](aero_LLM_12_Generalized eigendecomposition separates him from her in MLP.md)** | [Xem bài viết →](aero_LLM_12_Generalized eigendecomposition separates him from her in MLP.md) |
| [Thử Thách Lập Trình (Code Challenge): Tách Nhóm GED Đa Tầng (Phần 1)](aero_LLM_13_CodeChallenge GED for category isolation across layers (part 1).md) | [Xem bài viết →](aero_LLM_13_CodeChallenge GED for category isolation across layers (part 1).md) |
| [Thử Thách Lập Trình (Code Challenge): Tách Nhóm GED Đa Tầng (Phần 2) & Kiểm Chứng Chéo](aero_LLM_14_CodeChallenge GED for category isolation across layers (part 2).md) | [Xem bài viết →](aero_LLM_14_CodeChallenge GED for category isolation across layers (part 2).md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
