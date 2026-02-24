# Quỹ Đạo Không Gian Trạng Thái (State-Space Trajectories) Của Hệ Vector Ngôn Ngữ

## Tóm tắt (Abstract)
Báo cáo trình bày phương pháp phân tích Quỹ đạo không gian trạng thái (State-space trajectories), một kỹ thuật phân rã chiều dữ liệu áp dụng cho khối Embeddings của mô hình ngôn ngữ. Thực nghiệm trên GPT-2 Medium tích hợp thuật toán PCA (Principal Component Analysis) để ngưng tụ không gian ngàn chiều (1024 dimensions) của Token vectors xuống mặt phẳng tọa độ 2D. Bằng cách quan sát chuyển động tịnh tiến của ba biến thể đại từ (`him`, `her`, và lỗi ngữ pháp `round`) băng qua 24 khối Transformer, nghiên cứu đo đạc mức độ phân kỳ khoảng cách không gian (Euclidean distance). Dù giải thích được 80% tổng phương sai, phương pháp PCA vẫn tiềm ẩn rủi ro diễn giải sai lệch do đánh đồng phương sai với sự liên quan ngữ nghĩa (variance vs relevance).

---

## 1. Mở Đầu (Introduction)
Sự phức tạp của hộp đen mổ xẻ LLM chủ yếu đến từ rào cản đa chiều (High-dimensionality). Việc phải theo dõi sự chuyển động của Vector Embeddings trong không gian 1000 chiều là điều bất khả thi với nhận thức trực quan.
Để giải phẫu quá trình định hướng, ta có thể dùng kỹ thuật Giảm chiều dữ liệu (Dimensionality Reduction) - đặc trưng là Phân tích thành phần chính (PCA) để dập các vector về chung một bình diện không gian 2 trục/3 trục (Principal Components). Sự tịnh tiến của Vector đi từ Trạm sinh từ đầu tiên (Embeddings Matrix) đến chốt xả lũ cuối (Final Transformer block) sẽ vẽ nên một "Quỹ đạo Không gian trạng thái".

---

## 2. Thiết Lập Thí Nghiệm & Phương Pháp Chuyển Đổi (Methodology)

### 2.1. Cấu Trúc Biến Thể Văn Bản
Lựa chọn một tập hợp 54 câu văn bản tĩnh. Sau đó ngụy tạo 3 hệ dữ liệu (Datasets) độc lập dựa trên việc đánh tráo Token Mục Tiêu (Target Token):
1. Nhóm 1: Đại từ nhân xưng `"her"` (Chuẩn ngữ pháp).
2. Nhóm 2: Đại từ nhân xưng `"him"` (Chuẩn ngữ pháp).
3. Nhóm 3: Tính từ `"round"` (Lỗi sai ngữ pháp bẻ gãy cấu trúc câu, ví dụ: "we invited *round* to dinner").

### 2.2. Nhúng Tọa Độ Hệ Tham Chiếu Chung (Common Projection Space)
Một lỗi sai nghiêm trọng khi làm PCA là chạy hàm Fit tách biệt cho từng lớp layer. Nếu làm vậy, mỗi lớp layer sẽ sinh ra một tập tọa độ cơ sở (Basis Vectors) hướng khác nhau, làm gãy đứt sự liên kết so sánh trực tiếp.
Giải pháp: Ghép nối toàn bộ dữ liệu (Concatenation) dọc theo trục Token từ tất cả các câu văn và tất cả các Transformer Layers ($4050 \text{ vectors} \times 1024 \text{ dimensions}$). Sau đó nạp vào bộ hàm `sklearn.decomposition.PCA` để lấy về trục Tọa độ cốt lõi duy nhất chứa đựng quy luật chung nhất cho hệ dữ liệu.

---

## 3. Khảo Sát Đánh Giá: Quỹ Đạo & Điểm Mù (Analysis)

### 3.1. Sự Trương Nở Của Quỹ Đạo Chuyển Động (Trajectory Distances)
Khi rải đường viền theo Layer:
- Ở các điểm đầu (lúc mới nhúng Embedding), vạch tọa độ chung của cả 3 chữ `"him", "her", "round"` nằm chen chúc chật chội lấy nhau.
- Càng đi sâu vào các Transform block nội đĩa, mạng Neural bắt đầu cày xới ngữ cảnh. Khoảng cách (Euclidean Distance) vạch ra trên bình diện Trục PC1 và PC2 bắt đầu phân nhánh tẽ xa nhau. 
- *Hiện tượng bất đồng thuận:* Từ sai ngữ pháp (`"round"`) bộc lộ sự nhiễu loạn xa cách so với trục bình quân của hai đại từ chuẩn. Mô hình ý thức được rủi ro ngữ nghĩa của `"round"` và xoay xở xử lý nó bằng sự vặn xoắn phức tạp hơn (Additional rotations). Sự phân rã ở Final Layer cực kỳ dị dạng, bung rộng mãnh liệt như một lưới sao rải rác.

### 3.2. Giới Hạn Của Thành Phần Chính (The Scree Plot Trade-off)
Đồ thị phân bổ Scree Plot thể hiện: 3 tháp Thành phần đầu tiên nắm giữ khống chế $\approx 80\%$ tín hiệu cốt lõi (PC1 cắn 62%, PC2 10%, PC3 7.5%).
Tuy tỷ lệ nén vượt mốc đa số, 20% phương sai rải rác ở đằng sau đã bị cắt cụt không thương tiếc. Mặc định của PCA coi: "Sự phân tán càng rộng thì càng chứa vựa thông tin mạnh nhất" (Variance = Relevance). Thực tế vận hành LLMs phủ quyết điều này: Những quy luật ý nghĩa mỏng như tờ giấy (chiếm 0.1% phương sai) có thể định hình toàn bộ tư duy logic của mô hình đối với Token. Nên việc giản lược State-space trajectories chứa đựng một rủi ro Diễn dịch Quá Mức (Overinterpreting).

---

## 4. Kết Luận
Quỹ đạo không gian trạng thái biến thuật toán vĩ mô thành một thước phim có thể theo dõi. Nhờ giảm chiều, ta chứng kiến tận mắt cách một mô hình tách dần một từ sai ngữ pháp tách xa khỏi các từ vựng hợp chuẩn khi nó đi qua từng trạm kiểm duyệt nội vi của Transformer. Dẫu vậy, phương pháp này chỉ đóng vai trò là cột đèn dẫn lối trực quan (Insightful starting points), không được phép dùng làm kết luận tối thượng cuối cùng cho kiến trúc LLM vì hệ lụy đánh mất thông tin của PCA.

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm đo đạc khoảng cách và biểu diễn tọa độ quỹ đạo hai chiều PCA trong `aero_LLM_06_State-space trajectories through embedding space.md` (Cách sử dụng hệ chiếu chung Common Space thay vì tách lẻ cho Transformer Layers).
