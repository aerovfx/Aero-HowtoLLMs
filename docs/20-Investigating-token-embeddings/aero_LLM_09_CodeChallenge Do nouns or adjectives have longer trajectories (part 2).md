# Thử Thách Lập Trình: So Sánh Độ Dài Quỹ Đạo Của Danh Từ Và Tính Từ (Phần 2)

## Tóm tắt (Abstract)
Tiếp nối chặng thiết lập PCA, phần hai đi sâu vào phác họa quỹ đạo biểu diễn của Danh từ so với Tính từ trên bình diện Thành phần chính (PC1, PC2). Khảo sát ghi nhận một sự tách biệt quỹ đạo cực lớn giữa hai loại từ này, chứng tỏ mô hình sinh ra những phép nhúng phân mảnh (divergence) cho các tín hiệu ngữ pháp khác nhau. Hơn nữa, bằng việc sử dụng phép tính khoảng cách Euclide chéo lớp (Layer-to-layer Euclidean distance) và biến đổi Logarit, báo cáo định lượng tốc độ dịch chuyển (Path length trajectory) của từng vector. Thử nghiệm trên mô hình siêu lớn (GPT-2 XL) chứng thực giả thuyết lõi: Từ loại Đóng vai trò trục xoay cấu trúc (Danh Từ) bắt buộc hệ thống phải kéo thông tin từ khoảng context rộng dài hơn, nên nó sinh ra quỹ đạo dịch chuyển dài và kịch liệt hơn so với từ loại Mô tả vệ tinh cục bộ (Tính Từ).

---

## 1. Mở Đầu (Introduction)
Có tọa độ PCA trong tay, giờ là lúc truy vết lại chuyển động của các điểm hạt (Particles) theo tuyến tính thời gian của mạng Transformer (Layer $0 \to 36$). 
Dự phóng ban đầu (Hypothesis): "Độ uốn nắn" đối với Tính từ sẽ nhẹ nhàng hơn so với Danh từ. Bởi trong Tiếng Anh, tính từ thường chỉ quan tâm đến danh từ đứng ngay sát vách nó (ví dụ "a *happy* sentence"), nên cửa sổ tập trung (Context Window Demands) rất hẹp. Nhưng một Danh từ (Noun) mang sức nặng cấu trúc, nó cần phải liên kết với các đại từ bổ nghĩa phía trước, động từ theo sau, và đối tượng hệ quả ở tuốt cuối câu. Sự hấp thụ ngữ cảnh phong phú này chắc chắn sẽ làm quỹ đạo tính toán Vector của Danh từ di chuyển quãng đường dài hơn.

---

## 2. Trình Bày Quỹ Đạo Động (Trajectory Projections)

### 2.1. Sự Phân Cực Trên Tòa Độ PC2 (Divergence in PC Space)
Khi chiếu bóng Token Vectors lên hệ trục PC1-PC2:
- Ở trục PC1, quỹ đạo của Nouns và Adjectives vận động chạy song song giằng co về cùng một phương hướng, hấp thụ năng lượng tổng quan.
- Ở trục PC2, xuất hiện một cuộc "Đại phân ly" (Striking Divergence). Danh từ và Tính từ bay thẳng về hai cực chóp trái ngược nhau. Điều này xác thực sự tinh tế của GPT-2: Mô hình từ nội tại đã hình thành một lưới phân định rõ ràng cách hành xử chuyên biệt cho hai vùng trời từ loại (Grammar part of speech paths).

### 2.2. Nghịch Lý Ở Tầng Xả Cuối (Final-layer Drop-off)
Tại trạm Transformer cuối cùng trước khi đưa vào Vocab Output, khoảng cách của chúng tụt dốc thảm hại, cả hai Quỹ đạo hợp nhất và vón cục lại với nhau. Đây là đặc tính lặp lại (Consistent feature) đặc trưng của việc xả áp suất (Unembedding projection compression) thường thấy trong mạng Transformer. 

---

## 3. Khoảng Cách Quỹ Đạo Vượt Tầng (Layer Inter-Distance Analysis)

### 3.1. Phương Pháp Luận Logarit Euclid
Để định lượng giả thuyết đầu bài, ta đo quãng đường vector di chuyển bằng khoảng cách Euclide từ Layer I sang Layer I-1. Code được Vector hóa (Vectorization Arrays) để trừ triệt tiêu từng khối mà không cần các vòng lặp For tốn kém phần cứng `numpy.sqrt(numpy.sum(numpy.diff(points)^2))`.
Mặc dù vậy, sự vận động ở các chặng Layer sâu là cực kỳ bạo liệt (Big Leaps), nó đè bẹp xẹp lép thông số ở các chặng đầu nếu để nguyên đồ thị tỷ lệ tuyến tính (Linear scaling). Do đó, phép logarit cơ số e (Log-scaling) được sử đụng để dàn đều sắc thái động học. Mọi sai số được xử lý qua công thức đại số $Log(A) - Log(B) = Log(A/B)$.

### 3.2. Kiểm Chứng Cross-model (Thử Thử Với GPT-2 XL)
Khi thi hành bộ quy tắc trên bản thể có trọng lượng cao nhất (GPT-2 XL - $1.5$ Billion Parameters):
1. **Độ ổn định PCA tăng:** Component Scree Plot nắm giữ tốt hơn, 2 Trục PC1, PC2 nay thâu tóm trên $\approx 55\%$ Variance (so với dưới 40% của GPT-2 Large). 
2. **Xác nhận giả thuyết:** Đồ thị đường chênh lệch $\Delta = Log(Path_{Nouns}) - Log(Path_{Adjs})$ nằm cắm rễ hoàn toàn ở lãnh địa Dương (Positive Zone) trên toàn bộ mọi tuyến Layer. 
Nghĩa là: **Vector Danh từ luôn phải di chuyển nhiều hơn, tự thay đổi mãnh liệt hơn Vector Tính từ ở từng nấc của mô hình**. Lượng bồi đắp thông tin (Integration tokens context) đổ dồn vào Danh từ cao hơn hẳn Tính từ như đúng Logic ngôn ngữ học con người.

---

## 4. Kết Luận
Bằng việc hợp rèn hai thế võ: Bóc tách tự động (spaCy POS) và Biểu diễn quỹ đạo nén (PCA Path length), kỹ thuật này là một kính quang phổ hiệu năng cao giúp chứng minh LLM hiểu cấu trúc ngữ pháp ngôn ngữ người sâu sắc hơn việc chỉ đoán bừa chữ tiếp theo. Lực chú ý context window (Attention heads) cấp phát thông lực cực kỳ uyển chuyển tùy thuộc vào gánh nặng vai vế của từ. Cảnh báo nhỏ duy nhất là mẫu thử Frankenstein là một tệp tiểu thuyết cổ ngữ, phân tích thực địa (Research application) cần mở rộng trên các thư viện mạng Modern Text để tránh thiên vị (Bias text limit).

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm mã hóa Log-distance và kỹ thuật vector hóa theo mảng numpy trong `aero_LLM_09_CodeChallenge Do nouns or adjectives have longer trajectories (part 2).md`. Tái lập sự bùng nổ hiệu năng bằng mô hình GPT-2 XL.
