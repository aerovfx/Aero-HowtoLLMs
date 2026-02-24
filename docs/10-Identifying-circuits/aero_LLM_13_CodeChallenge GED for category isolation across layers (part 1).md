# Thử Thách Lập Trình (Code Challenge): Tách Nhóm GED Đa Tầng (Phần 1)

## Tóm tắt (Abstract)
Thử thách lập trình (Code Challenge) này là một bước tiến xa hơn trên hành trình diễn giải Mạng thần kinh, nơi Học viên được hướng dẫn xây dựng Bộ máy Kiểm định Chéo (Cross-validation) trên thuật phân tích GED (Generalized Eigendecomposition). Ta tách biệt "Data Huấn luyện" (Câu Điều kiện tự thiết kế) khỏi "Data Đánh giá" (FineWeb Dataset ngẫu nhiên) nhằm phòng chống Overfitting. Cấu trúc gồm ba Module lõi: 1) Hệ thống Khai thác Ngữ Cảnh của Từ khóa Từ FineWeb. 2) Cỗ Máy Ép Chiều Tức Thời (Per-layer PCA) và 3) Hàm Đóng gói Phân ly GED, tính toán Pattern Vectors và Eigenvalues tịnh tiến trên các tầng.

---

## 1. Mở Đầu (Introduction)
Với sức mạnh kinh hoàng của GED, thật dễ bị cuốn vào cạm bẫy "Mù mờ Overfit" - Nơi Ma trận Eigenvectors học quá kỹ những đặc thù vô nghĩa từ một File văn bản bé xíu rồi tạo ảo giác thành công. 
Trong bài tập này, sinh viên được rèn luyện quy trình thiết lập một đường ống Mạch (Pipeline) Khoa Học Dữ liệu Tiêu chuẩn:
- Mảnh ghép Cơ bản $\mathbf{Train\_Set}$: Sử dụng bộ dữ liệu kinh điển HIM/HER mà ta thiết kế trong lab trước (54 Cặp Tuyên thệ tương đồng). Eigenvectors sẽ được huấn luyện TẠI ĐÂY.
- Thử thác Thực địa $\mathbf{Test\_Set}$: Tự tay cào 2000 Văn bản Mạng (`fineweb`), trích chọn 50 mẩu câu chứa HIM, và 50 mẩu chứa HER trong vô vàn bối cảnh (Contexts) hỗn loạn. GED Filter Model phải lọc thành công dữ liệu chưa hề được thấy này.

---

## 2. Tiết Lập Trình Thử Thách (Code Challenges)

### Giai Đoạn 1: Tuyển mộ Tập Đánh Giá Độc lập (The Blind Test Set)
Để lấy Token Trọng tâm (Target Tokens) từ Fineweb, thuật toán quét lùng Token `him` & `her`. Để đảm bảo cấu trúc Đồng đều Input Model `[100, 10]`, ta cắt đúng Mảnh ngữ cảnh dài 10 Token, với Keyword trọng tâm luôn đậu ở Vị trí Index $6$.
> **Bí quyết Thiết Kế Tách Biệt**: Để ngăn mô hình ăn gian (Ví dụ 1 câu chứa cả "him and her" lọt vào cả 2 list), mảng Token Her được cộng thêm chỉ số Lệch (Offset) $1/2$ quãng đường, tống chúng thẳng xuống Nửa sau của Dataset. Sự phân tách này là cơ chế bảo vệ Sinh học (Contamination Free).

### Giai Đoạn 2: Trạm Ép Dữ Liệu Chuyên Sâu (Dimension Reduction Factory)
Tạo Hàm `dim_red(layer)` tái diễn quá trình Ép không gian $3072D\ \to 99\%\ Variance$ mà ta từng làm, nhưng phải bọc trong một Hàm độc lập để lặp tự động n-Layers:
- Xử lý Nút thắt Cổ Chai Hiệu Năng: Ở bài trước ta Tự tay giải Eigen PCA. Tại bài lặp Vòng Tầng này, ta BẮT BUỘC gọi mảng `sklearn.decomposition.PCA` để tận dụng hệ tăng tốc nhân $C++$.
- Kẻ thù Của Đại số Học (`sklearn Components`): Chú ý cực điểm rằng `scikit-learn` cố tình xoay ngang Eigenvector theo trục Hàng (Rows) thay vì chuẩn Mực Cột (Columns). Khi tính toán, một phép Chuyển Vị Ma Trận (`PCA.components_.T`) là bức tường phòng thủ cuối cùng chống lại sụp đổ Lõi Hệ thống.

### Giai Đoạn 3: Cấu Trúc Khối Tổng Hợp GED (The GED Pipeline)
Tạo hàm tự động giải GED: `run_ged(train_data, pca_evecs)`
Một điểm tinh ý trong việc Lý giải Hệ Trọng Số: Eigenvector (`W`) KHÔNG PHẢI là Mẫu Kích Hoạt (Activation Pattern). Eigenvector là Lưới Lọc (Filter). 
Để thấy rõ Dấu Ấn Vật Lý của khái niệm "Giới tính" phủ lên bề mặt $3072$ Nơ-ron (Mọi Tòa nhà của Mạng Lưới), hàm toán học Đốc chiếu Phải lội ngược dòng: 
$$ Pattern = W_{GED} \cdot Covariance(S) \cdot PCA\_Evecs_{T} $$
Khối Cột Pattern Cuối Cùng đó được Correlate chéo giữa phân lớp HIM và lớp HER. Giá trị Correlation ($R^{2}$) và Trị Số Tách Lớp ($Max\ Eigenvalue$) được tống xuất phục vụ cho biểu đồ Diễn tiến Xuyên Tầng ở video Phần 2.

---

## 3. Khảo Sát & Trả Lời Vắn Tắt
- Tại sao phải `test_activations.copy()` nhưng Dictionary `train_activations` thì không? 
$\to$ Đáp án: Khi ta cắm Pipeline vào Memory Pytorch. Hàm Gán Không (Assignment) trỏ thẳng 2 Data tới 1 vùng vật lý. Chèn `.copy()` để bẻ gãy Con Trỏ, niêm phong Test_Set trở thành vùng biển Đóng Băng miễn nhiễm với mọi sửa đổi biến tấu diễn ra ở khối Train_Set.

---

## Tài Liệu Tham Khảo (Citations)
1. Thử thách Tái Lập và Viết Code tự động Đa Hệ (Automated Cross-validation Framework) tại `aero_LLM_13_CodeChallenge GED for category isolation across layers (part 1).md`. Giới thiệu cơ chế Lọc Cổ chai bằng `sklearn` tốc độ cao.
