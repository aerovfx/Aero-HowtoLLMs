# Phân Tích Độ Lệch RSA (Part 1): So Sánh Sự Bất Đồng Giữa Không Gian GloVe 50D và 300D

## Tóm tắt

Phân tích Độ tương tự Biểu diễn (Representational Similarity Analysis - **RSA**) là công cụ thống kê cốt lõi cho phép chẩn đoán chéo mức độ "đồng điệu" giữa hai mạng hệ không gian nhúng có chiều không gian bất cân xứng. Bài nghiên cứu học thuật này sử dụng cấu trúc RSA để khai phá sự dị biệt và chọn lọc từ vựng khi nâng cấp mạng *GloVe* cơ sở từ 50-Chiều (50D) lên kiến trúc nặng 300-Chiều (300D). Bên cạnh đó, chúng ta sẽ giới thiệu khái niệm Chỉ số Chọn lọc Hạng mục (Category Selectivity Index) nhằm lượng hóa sự cải thiện cấu trúc ý niệm do độ sâu chiều kích mang tới.

---

## 1. Thiết Lập 2 Trạm Không Gian Đấu Thầu (Embeddings Extractor)

Cả hai mô hình được chọn đều xuất thân chung từ gốc đào tạo Dữ liệu Wikipedia (`glove-wiki-gigaword-50` và `glove-wiki-gigaword-300`). Tuy nhiên, sức chứa (capacity) của chúng lại khác biệt hoàn toàn: một bên bị o ép trong khung $50$ tọa độ, một bên nở rộng đến $300$ tọa độ.

Mẫu phân tích không được để ngẫu nhiên. Chúng ta tiến hành một bộ thử nghiệm tiền tri thức (A priori controlled group) với 20 Từ vựng đại diện cho ba khối nghĩa khác biệt:
1. Nhóm **Vũ Trụ (Space):** *spaceship, satellite, galaxy, asteroid,...*
2. Nhóm **Nội thất (Furniture):** *chair, sofa, couch, desk,...*
3. Nhóm **Hoa Quả (Fruit):** *apple, banana, kiwi, peach,...*

Hệ ma trận cục bộ (Sub-matrices) được tạo ra cho cả 2 phía sẽ là $M_{50D} \in \mathbb{R}^{20 \times 50}$ và $M_{300D} \in \mathbb{R}^{20 \times 300}$. Mặc dù hai ma trận này không có cùng một hệ giải tích cơ bản, tuy nhiên, ma trận Tương quan Cosine giữa 20 từ ghép cặp (tương tác tự thân) lại luôn luôn trả về chung một kích thước là $20 \times 20$. Đây chính là "Cây cầu nối RSA".

---

## 2. Đo Lường Bằng Chỉ Số Chọn Lọc Hạng Mục (Category Selectivity Index)

Trước khi thực hiện đồng bộ RSA, mỗi phương trình Cosine Similarity Map sẽ được đánh giá mức độ sạch nhiễu nội bộ. 

### Chỉ Số Kháng Nhiễu Category Selectivity Index (CSI) 
Ý tưởng của CSI là so sánh: **Liệu độ gắn kết cấu trúc CÙNG một mạng (Wihtin-category) có áp đảo lực gắn kết độ lêch GIỮA các mạng sai lệch (Between-category) hay không.**
Phương trình tạo Mask $S_{idx}$ là nhân chéo Vector các ID nhãn nhóm. Sau đó, công thức CSI được xác định:
$$
CSI = \frac{\text{Mean}(S_{\text{within-categories}})}{\text{Mean}(S_{\text{between-categories}})}
$$
Trong đó:
- Dữ liệu thuộc **Within-category** (Tự thân trong nhóm) = Trích xuất các Block vuông nằm trên đường chéo Heatmap. 
- Dữ liệu thuộc **Between-category** (Xiên chéo giữa 2 nhóm, vd: Bàn ghế so với Vũ trụ) = Trích xuất các dải tọa độ Background của Heatmap.

### So Sánh 50D và 300D:
Thực nghiệm cho thấy CSI của GloVe 50D chỉ đạt $\mathbf{3.27}$ trong khi GloVe 300D đạt sức mạnh phân giải $\mathbf{5.62}$. 
Phương sai độ lệch vi phân của 50D cũng tản mác dữ dội hơn, trong khi tập ma trận 300D có xu hướng ép chặt sự sai lệch, tạo thành các hố rỗng nhiễu phân rã hoàn toàn những từ ngữ không chung rễ nội hàm. Kết luận: Chiều hướng ẩn lớn hơn (More dimensions) tạo ra một khoảng không đủ sâu để hệ thần kinh cất giữ các tinh chỉnh vi mô, thay vì cọ xát chồng chéo như ma trận hẹp.

---

## 3. Quần Thể Hóa Thuật Toán T-SNE và Cụm Mật Độ DBSCAN

Để kiểm chứng tính xác đáng của luận điểm CSI, phân cụm mật độ phi tuyến được bổ sung. Đồ thị chuyển hóa mô hình tọa độ từ Không gian Euclid $N$-chiều xuống mặt phẳng hiển thị vi mô (2D mapping).

Sử dụng chuỗi hàm liên hợp:
1. `t-SNE(perplexity=5...)` làm bứt gãy sự liên kế giả để hình thành hạt.
2. `DBSCAN(epsilon=0.5, min_samples=2)` khóa hạt nhân vi mô tạo chuỗi liên hợp ranh giới.

Ngạc nhiên thay, dù cho 300D có chỉ số kháng nhiễu CSI đỉnh cao hơn, nhưng thuật toán cấu trúc DBSCAN trên **cả 50D và 300D đều chia ra đúng 3 mảng cụm nội thất - không gian - trái cây giống hệt nhau.**
Tùy vào hạt giống ngẫu nhiên (Random Initializations), đôi khi từ "*Kiwi*" lại bị văng khỏi mảng trái cây và rơi vào lõi Vũ trụ, hoặc chìm vào Độc lập phân lập (Ungrouped Outliers). Sự hỗn loạn nhẹ này chứng thực một chân lý: Các thuật toán t-SNE hoạt động dựa theo quy luật Láng giềng t-Student không quan tâm tới chuẩn khoảng cách xa tuyến tính, do đó không bị lung lay bởi kích thước đa chiều mà dựa vào sức mạnh quy hội cấu trúc cục bộ.

*(Mời xem tiếp CodeChallenge Compare embeddings with RSA part 2 để đi vào ma trận Correlation chéo)*.

---

## Tài liệu tham khảo

1. **Pennington, J., et al. (2014).** *GloVe: Global Vectors for Word Representation.* EMNLP (Thông tin về Vector 50D và 300D).
2. **Kriegeskorte, N., et al. (2008).** *Representational similarity analysis.* 
3. Giảng nghĩa kỹ thuật khoa học dữ liệu *Compare embeddings with RSA (part 1)*.
