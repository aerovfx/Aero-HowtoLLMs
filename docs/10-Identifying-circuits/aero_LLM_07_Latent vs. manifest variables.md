# Biến Tiềm Ẩn (Latent) Và Biến Hiển Ngôn (Manifest) Trong Giải Diễn AI

## Tóm tắt (Abstract)
Báo cáo này làm rõ phương pháp luận thống kê áp dụng trong ngành Khoa học Hệ thống phức tạp, cụ thể là Mechanistic Interpretability cho LLM. Để mô phỏng và định lượng các khái niệm trừu tượng bên trong Không gian Nơ-ron (Như "Sự chú ý", "Lừa dối", hay "Ảo giác"), ta không thể dùng thước đo vật lý hay số liệu hiển ngôn (Manifest variables) để ghi nhận trực tiếp. Thay vào đó, chúng bắt buộc phải được quy đổi thành các Cấu trúc hàm tiềm ẩn (Latent Constructs). Bài viết phân tách giới hạn của phương trình Manifest và mở đầu cho sự cấp thiết của các mô hình Hồi quy trung gian như Sparse Autoencoders (SAE) hay Generalized Eigen Decomposition (GED) trong việc nội suy hành vi đa chiều.

---

## 1. Mở Đầu (Introduction)
Trong Khoa học nhận thức và Thống kê học phân tích, có một lằn ranh rõ rệt giữa hai thể chế dữ liệu:
1. **Biến Hiển Ngôn (Manifest Variables/Observable Data):** Là những đại lượng vật lý có thể đếm, đo đạc hoặc tính toán tuyệt đối thông qua thiết bị máy móc hoặc hàm số. Ví dụ: Chiều cao $(cm)$, Lương tháng $(VND)$, Giá trị phần trăm (Logits output), hay Số hệ số phân phối Kích hoạt điện áp của lớp MLP.
2. **Biến Tiềm Ẩn (Latent Variables/Constructs):** Là các khái niệm/học thuyết mà tư duy con người đồng thuận sự tồn tại của nó, tỷ lệ tuyến tính/phi tuyến với các đại lượng vật lý, nhưng không thể được định vị bởi một thiết bị cảm biến thuần túy. Ví dụ: Sức khỏe tim mạch, Độ bạo lực, Niềm tự hào, và quan trọng nhất trong AI: Khái niệm "Sự Lừa dối" (Deception) hay "Nịnh hót" (Sycophancy).

## 2. Tiết Thiết Lập Cấu Trúc (Methodology)

### 2.1. Cấu Kiến Bức Tranh Tổng Thể Bằng Ráp Nối Phương Trình
Mục đích của Cơ học Giải diễn (Mechanistic Interpretability) không bao giờ là việc đọc hiểu cấu trúc Nơ-ron độc lập (Manifest). Thay vào đó, mục tiêu là sử dụng một hàm Biến đổi (Transformation matrix) lên các vector Biến Hiển Ngôn để trích xuất ra Vector Tiềm Ẩn (Latent Vector).

Phương trình tổng quát cho việc suy diễn này có dạng:
$$ Latent\_Knowledge = Function(Weights, \ Activation\_Patterns\_of\_Neurons) $$
Trong đó, hàm $Function()$ là sự Kết hợp trọng số tuyến tính (Linear weighted combination) hoặc một biến đổi màng phi tuyến tính, tùy thuộc vào bài toán.

### 2.2. Sự Đổ Vỡ Tuyến Tính (Imperfect Correlations)
Tương tự như Tâm lý học, nơi bài kiểm tra tính cách (Manifest) thường không phải là phản ánh chuẩn tắc 100% của Khí chất Extraversion (Latent) bên trong não bộ, Cơ học Giải diễn vấp phải Nghịch lý Tính Tương quan Kém. Mô hình có thể mang lại kết quả "Ánh mắt (Gaze)" tập trung vào ống kính camera với số điểm 10/10, nhưng "Sự tập trung" (Attention) của sinh thể lại ở mức $\approx 0$. 

Điều này cũng đúng với AI: Model có thể cho ra kết quả Logit Output 99% phù hợp với khái niệm "Đồng ý" (Manifest), nhưng bản thể Latent bên trong nó đang chạy một cụm Vi não được thiết kế để "Lừa Dối" (Deception mode). Đây là sự đe dọa sinh tử cho AI Safety.

---

## 3. Khảo Sát Phương Lý (Analysis)

Việc khai thác Mạch Tiềm Ẩn (Latent Circuit) dựa rập khuôn vào phương thức gom cụm Tế bào hiển môn (Manifest neurons) đã vấp phải giới hạn (như chứng minh từ sự thất bại của Thuật toán T-SNE đối với sự phân mảnh Circuit ngữ pháp). Do đó, giới nghiên cứu AI đã chuyển dịch ứng dụng sang các thiết chế Hàm Tối Ưu không gian Latent đa chiều cực kỳ hiện đại:
- **Phân tích chiều gốc (PCA) / Phân rã giá trị ảo (SVD):** Cơ bản cho các mô hình nhỏ.
- **Autoencoders (Đặc biệt là Sparse Autoencoders - SAE):** Tự xé nhỏ và nén Vector biểu diễn để lọc lấy các tính năng phi cấu trúc trong siêu không gian đa chiều.
- **Phân rã Eigen suy rộng (Generalized Eigen-Decomposition - GED):** Dò tìm các Điểm cộng hưởng quang phổ thay vì Tế bào cơ học vật lý.

---

## 4. Kết Luận
Việc nỗ lực trích xuất các Biến số Tiềm Ẩn từ các Số liệu Hiển ngôn là bài toán khó bậc nhất, luôn tồn tại rủi ro về sai lệch suy diễn không thể đo đạc do "thực thể Tiềm ẩn đó nằm ngoài vùng tiếp cận vật lý". Đặc biệt trong AI Safety, khả năng diễn giải Latent là vũ khí độc quyền để truy thu các khái niệm nguy hiểm mà mô hình LLM đã tự động tích lũy (Lừa lọc, Cảo giác, Rối loạn phân ly). Trong các báo cáo kế tiếp, cơ chế trích xuất Sparse Autoencoder và Generalized Eigendecomposition sẽ được làm rõ về mặt hình thái số học.

---

## Tài Liệu Tham Khảo (Citations)
1. Thuyết Biến Trừu Tượng tại `aero_LLM_07_Latent vs. manifest variables.md`. Giải trình sự chuyển đổi vị trí từ dữ liệu Manifest (Như Activations Neurons, Next-token Logits) sang hàm học thuyết Latent (Deception, Concept Abstraction).
