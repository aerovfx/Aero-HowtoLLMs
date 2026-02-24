# Nhận Diện Khái Niệm Xuyên Tâm Với Phân Rã Giá Trị Riêng Suy Rộng (Generalized Eigendecomposition - GED)

## Tóm tắt (Abstract)
Báo cáo này chuyển đổi phương pháp luận Giải Diễn Cơ Học của Mạng nơ-ron (Mechanistic Interpretability) từ mô hình Autoencoders tuyến tính đa tầng sang Phương pháp Đại số Tuyến Tính Giải Tích - Cụ thể là Phân rã Trị Riêng Suy Rộng (Generalized Eigen-Decomposition, GED). Khác biệt sâu sắc đối với Phân tích Thành phần Chính (PCA - vốn bị gò bó bởi tính Trực Giao - Orthogonality), **GED cho phép các vectors thành phần phi-trực giao đan chéo**, nhằm mục đích tìm ra một Trọng số (Weight Matrix) phân ly tối đa tỉ số Phương Sai (Variance Ratio) giữa Quần thể Dữ liệu Tín hiệu (Signal) so với Dữ liệu Tạp Âm (Reference/Noise). Thông qua thao tác thu hẹp hệ số Covariance Matrix (Shrinkage Regularization), GED loại bỏ thành công các ảo giác Nhiễu Tương Quan, mở ra tiềm năng lọc Khái Niệm Rời rạc (VD: Dò tìm Mạch vi phân tách Nouns vs. Verbs).

---

## 1. Mở Đầu (Introduction)
Trong Không gian Hệ số (Weight Space) khổng lồ của một Language Model, các Khái Niệm Tiềm Ẩn (Latent constructs) như Ngữ Nghĩa Vạn Vật không nằm độc lập tại một góc tọa độ mà bị Phối Trộn (Mixed together).
Bình thường, phương pháp tiếp cận kinh điển nhất của Khoa học Dữ liệu là **PCA (Principal Component Analysis)**. Tuy nhiên PCA bản chất chỉ là Phép Cắt và Xoay Trục Tọa độ Trực giao: Nó tối đa hóa Phương Sai Toàn Khối (Variance) nhưng Không có khái niệm về "Sự khác biệt Nhãn Loại". Nó sẽ cho bạn biết Trục nào dữ liệu tản mạn nhất, nhưng Không hề có khả năng cắt đôi Khối Dữ liệu Màu Xanh (Dành cho danh từ) ra khỏi Khối Dữ Liệu Màu Tím (Dành cho động từ), vì nó không được lập trình để "Tối Đa Hóa Khoảng Cách Phân Ly Tín Hiệu".
GED xuất hiện như một "Cỗ máy siêu việt" để tróc lớp hai khái niệm đồng vị.

---

## 2. Tiết Lập Trình Toán Học (Methodology)

### 2.1. Đạo Hàm Tỉ Số Tín Hiệu - Nhiễu (Signal-to-Noise Ratio Optimization)
Ta đặt 2 Ma trận Hiệp Phương Sai (Covariance Matrix): 
- Cấu trúc Mạch Tín Hiệu cần làm nổi bật $\mathbf{S}$. (Ví dụ: Từ vựng xe Ô tô)
- Cấu trúc Mạch Đối Chiếu cần bị triệt tiêu $\mathbf{R}$. (Ví dụ: Từ vựng xe Tải) 
Mục đích là đi tìm một vector trọng số $W$ sao cho nó khuếch đại tối đa Ma trận $S$ và nén nhỏ tối đa Ma trận $R$. Hay định nghĩa bằng công thức Tỉ số Rayleigh Quotient:
$$ \Lambda = \frac{W^T \mathbf{S} W}{W^T \mathbf{R} W} $$
Khi ta cần tìm Đạo hàm vi phân Lagrange (Bằng cách trói Buộc $W^T \mathbf{R} W = 1$), toàn bộ Biểu thức Toán học kinh điển này hóa giải dưới dạng Biểu thức Eigendecomposition trên Tích hiệp:
$$ \mathbf{R}^{-1} \mathbf{S} \ W = \Lambda W $$ 
Tuy nhiên, nghịch lý là ở đây: Trong khi $\mathbf{R}^{-1}$ và $\mathbf{S}$ đều là ma trận Đối Xứng Phẳng (Symmetric), khi chúng cấu thành Phép nhân $\mathbf{R}^{-1} \mathbf{S}$, nó tạo thành Thể Đa Hình (Non-Symmetric). Hệ quả cực quan trọng của lý thuyết Tuyến tính: **Eigenvectors ($W$) tìm được sẽ mất tính Trực Giao (Orthogonal).** Thay vì các Vector xoay góc 90 độ Vuông vức, nó có thể nhọn hơn, xòe hơn, tự điều chỉnh linh động để men theo Dải Phân Tách dữ liệu thực thụ.

### 2.2. Phẫu Thuật Covariance Matrix Với Điều Hòa Thu Hẹp (Shrinkage Regularization)
Vì cấu trúc Language Model sở hữu Feature Khổng Lồ, Ma trận $R$ sẽ dễ bị rơi vào dạng Dẹt Phẳng Siêu Hình (Flattened ellipse with Zero-Rank determinant) - tức Determinant $=0$, dồn ép phép Kịch Đảo $\mathbf{R}^{-1}$ thành vô cực.
Phương pháp "Shrinkage Regularized" ép phồng khối Ellipse xẹp lép này bằng cách độn lên một chút năng lượng vào Đường Chéo (Identity Matrix), mô phỏng bằng công thức:
$$ \tilde{\mathbf{R}} = (1 - \gamma)\mathbf{R} + \gamma \alpha \mathbf{I} $$
**(Trong đó $\alpha$ là Trung bình dãy Giá trị riêng Eigenvalues).**
Nếu $\gamma \to 0$, Không có điều hòa áp dụng. Nếu $\gamma \to 1$, Phân rã GED tan rã trở về lại hình thái chắp vá PCA thô ban đầu. Việc chọn biến số Gamma (Thường $\approx 0.01$) đóng vai trò then chốt cho sự sinh tồn của mô hình GED.

---

## 3. Khảo Sát Thử Nghiệm Phân Phối (Analysis)

Sử dụng thư viện `scipy.linalg.eigh` (Định mức Toán Hàm Hermitian) trên bộ Dữ liệu 2-Stream Mô phỏng:
1. **Qua PCA:** Trục tọa độ Component quay vuông góc, gộp chung hai dòng Dữ liệu (Xanh-Tím) làm một. Khả năng tìm Vi Mạch Độc Lập $\to 0$.
2. **Qua Thuần GED:** Đồ thị Phép Chiếu (Projected Space) trả về 2 Luồng Vector Phân lập Độc lập rõ rệt Không Đè Nhau, dù các hướng góc Eigenvectors tự do xiên xẹo phi-trực giao. Nó bóc trần đúng nghĩa hai dòng dữ liệu ra hai hệ tham chiếu khác nhau.
3. **Hiệu Ứng Shrinkage:** 
  Khi ta cố tình cấp phép thử $\gamma = 0.4 \to 0.9$ rất lớn. Cặp Eigenvectors Không trực giao bị bẻ dãn góc lồi dần, bị lực ép cưỡng chế chuyển hóa dần thành Đồ thị vuông góc PCA như cũ. Nó tái khẳng định lại nguyên lý Toán: Việc Bơm quá tay Ma trận Đường chéo để vá lỗi Rank Null có thể đánh đổi bằng sự mù lòa của Toán Lý Phân Ly.

---

## 4. Kết Luận
Việc áp dụng Phân tích Generalized Eigen-Decomposition (Hàm Tỉ Số Cạnh Tranh Giữa 2 Covariances) là phép tịnh tiến thay thế ưu việt cho thuật giải rập khuôn của PCA khi làm việc với Multi-variates Source. Bằng việc cởi trói Không Gian Trực Giao để tự do luồn qua các Vector phân mảnh, và dùng phẫu thuật Cấy Đường chéo Shrinkage để vượt rào Cấm Zero-Inverse, GED không chỉ ứng dụng cực thịnh trong Thần kinh học mà còn mở toang cánh cửa để Cắt Mạch (Circuit Cutting) một cách tường minh cho những Khái Điểm Ẩn (Gender, Logic) giấu kín bên trong LLMs. 

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm ứng dụng Khoa học Giải Tích Trực Giao tại `aero_LLM_11_Non-orthogonal latent components via eigendecomposition (theory and demo).md`. Minh định hóa công thức Khối Co Giãn Hệ Số Covariance $\tilde{\mathbf{R}}$ và sự suy thoái PCA theo bước của biến $\gamma$.
