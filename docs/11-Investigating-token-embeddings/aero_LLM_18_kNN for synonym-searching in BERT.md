
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [11 Investigating token embeddings](../index.md)

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
# Khai Thác Thuật Toán k-NN Cho Tìm Kiếm Từ Đồng Nghĩa Trên BERT

## Tóm tắt

K-Nearest Neighbors (k-NN - k Láng giềng gần nhất) là một thuật toán cốt lõi trong phân loại cụm học máy cổ điển (Machine Learning Classification). Tuy nhiên, khi kết hợp cùng Vector nhúng (Embeddings) của họ mô hình ngôn ngữ lớn (LLMs) như BERT, thuật toán này thể hiện khả năng tra cứu chéo từ đồng nghĩa (synonym search) ở mức độ đáng kinh ngạc. Bài viết dưới đây trình bày nguyên lý không gian của k-NN, phân biệt hai định chuẩn tính điểm Euclidean và Cosine Similarity, cũng như cách triển khai cho các chiều ẩn trong từ điển số hóa tự nhiên.

---

## 1. Nguyên Lý Của k-Nearest Neighbors (k-NN)

Trong mô hình k-NN, dữ liệu mới không hề được gắn nhãn trước theo một hệ số chặn hàm (Linear threshold). Cách tiếp cận này tuân theo một cơ chế định vị hình học đơn giản: **Dữ liệu thuộc về cộng đồng nào đang áp đảo xung quanh nó.**

1. Đầu vào là một Vector truy vấn vô danh (Unlabeled data point) $x$.
2. Hệ thống quét đếm khoảng cách từ $x$ đến toàn bộ những Vector $\vec{v}_i$ đang mang nhãn dữ liệu có sẵn trong bộ nhớ.
3. Tham số $k$ quy định sẽ lấy $k$ điểm có khoảng cách không gian sát với $x$ nhất (Nearest Neighbors). Lời khuyên là thiết lập $k$ thành số lẻ (VD: $k=3, 5, 7$) để chặn trường hợp hòa/cân bằng.
4. Lựa chọn ưu tiên theo nguyên lý Bầu chọn theo khuynh hướng (Majority Voting): nhãn hiệu chiếm đa số trong $k$ phần tử cận kề sẽ được gán cho $x$.

Trường hợp sử dụng để khám phá từ loại trong BERT, việc dự đoán theo lớp sẽ được thay thế bằng liệt kê $k$-vector gần nhất với từ gốc nhằm tìm ra các từ đồng nghĩa (vd: "Beauty" sẽ gọi ra "Gorgeous", "Elegance"). 

---

## 2. Số Học Khoảng Cách: Euclidean và Cosine Similarity

Không gian tọa độ của mảng Embeddings ma trận BERT sở hữu $D=768$ chiều. Bài toán tìm Láng giềng (Distance calculations) yêu cầu một thước đo chuẩn. Hai thước đo thông dụng đem lại hai góc nhìn dị biệt:

### 2.1. Chuẩn Khoảng Cách Hình Học (Euclidean Distance)
Lấy gốc từ định lý tam giác vuông trong không gian $N$-chiều, Euclidean đo đạc chiều dài thật sự của sợi dây nối giữa mũi tên vector token $\vec{v}$ và token mục tiêu $\vec{w}$:
$$ 
\delta(\vec{v}, \vec{w}) = \sqrt{\sum_{i=1}^{D} (v_i - w_i)^2} 
$$
Chuẩn Euclidean thể hiện tính tách biệt tuyệt đối (absolute spatial magnitude) của thông tin.

### 2.2. Chuẩn Tương Quan Góc (Cosine Similarity)
Trọng tâm đo lường sự đồng dạng không nằm ở lực độ dài, mà vứt bỏ tất cả giới hạn véc-tơ để tìm độ chênh góc giữa hai ngọn vector:
$$ 
\text{CosineSim}(\vec{v}, \vec{w}) = \frac{\vec{v} \cdot \vec{w}}{\|\vec{v}\| \|\vec{w}\|} \in [-1, 1] 
$$

**Sự Lệch Pha Đáng Lưu Ý:** Các vector có chung hướng nội hàm (Cosine Similarity hướng về 1) nhưng hoàn toàn có thể sở hữu Khoảng cách Euclidean kéo dãn ra khổng lồ nếu độ phủ vector (Norm of vector) bị đẩy cực xa gốc tọa độ. Do đó, việc tìm Láng giềng gần nhất k-NN trong cấu trúc BERT đòi hỏi nhà nghiên cứu phải xác định thuộc tính đang săn tìm là khoảng cách hay góc lệch nhạy cảm biểu diễn song song.

---

## 3. Khai Thác Tiền Xử Lý Giảm Chiều Bằng PCA/t-SNE

Khi ứng dụng tệp $k=5$ cho cụm từ "Beauty" xuyên thấu toàn bộ $30.000$ từ điển của BERT, gánh nặng toán học (Tính $30.000$ phép tính hàm mũ $L2-Norm$) có thể sẽ làm đình trệ bộ vi xử lý nếu hệ vector lớn như định dạng GPT hiện đại (với số token trên 1 triệu). 
Theo lý thuyết thông luật của Học Máy (Machine Learning), để tránh "Lời nguyền đa chiều" (Curse of Dimensionality), ma trận nên được phân rã bằng Principal Component Analysis (PCA) triệt tiêu quang phổ yếu (SVD variance noise) sinh ra một ma trận giảm chiều $D = 100$ trước khi hàm k-NN khởi chạy, đảm bảo chi phí thấp mà không đánh tụt độ nhạy tương quan ngữ nghĩa.

---

## 4. Kết luận

Mô hình k-Nearest Neighbors là khối hạt nhân trong mọi bộ truy vấn tìm điểm dữ liệu (Search Engines) ứng dụng vào Mạng Nơ-ron. Việc lạm dụng tính chất khoảng cách ở vùng Embeddings của BERT cho phép k-NN bứt phá khỏi cơ chế nhãn mác nhị phân, trở thành công cụ đắc lực giải phẫu hiện tượng đa nghĩa từ vựng cũng như khai thác vùng giao thoa khái niệm (Concept boundary overlapping).

---

## Tài liệu tham khảo

1. **Cover, T., & Hart, P. (1967).** *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory.
2. **Devlin, J., et al. (2018).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
3. Tài liệu đào tạo *Investigating token embeddings - kNN for synonym-searching in BERT*.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [aero_LLM_01_CodeChallenge Cosine similarity (advanced) (part 1).md](aero_LLM_01_CodeChallenge Cosine similarity (advanced) (part 1).md) | [Xem bài viết →](aero_LLM_01_CodeChallenge Cosine similarity (advanced) (part 1).md) |
| [aero_LLM_02_CodeChallenge Cosine similarity (advanced) (part 2).md](aero_LLM_02_CodeChallenge Cosine similarity (advanced) (part 2).md) | [Xem bài viết →](aero_LLM_02_CodeChallenge Cosine similarity (advanced) (part 2).md) |
| [Theo Dõi Dòng Chảy Cosine Similarity Trên Trục Văn Bản Chuyên Tuần Tự (Word Sequences)](aero_LLM_03_CodeChallenge Cosine similarity in word sequences.md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Cosine similarity in word sequences.md) |
| [Nghệ Thuật Vẽ Bản Đồ Nhiệt Ma Trận Nhúng Bằng Cường Độ Từ (Coloring Cosine Similarity)](aero_LLM_04_CodeChallenge Coloring cosine similarity.md) | [Xem bài viết →](aero_LLM_04_CodeChallenge Coloring cosine similarity.md) |
| [Ảo Ảnh Của Trí Tuệ Toán Học Trong Ngôn Ngữ: Sức Mạnh Của Random Embeddings](aero_LLM_05_CodeChallenge Can random embeddings be interpreted.md) | [Xem bài viết →](aero_LLM_05_CodeChallenge Can random embeddings be interpreted.md) |
| [Phương Pháp T-SNE Và Thuật Toán Phân Cụm DBSCAN: Chiếu Không Gian Đa Chiều Cho LLMs](aero_LLM_06_T-SNE projection and DBSCAN clustering (theory).md) | [Xem bài viết →](aero_LLM_06_T-SNE projection and DBSCAN clustering (theory).md) |
| [Phân Cụm Ngữ Nghĩa Qua Phép Chiếu t-SNE & Mật Độ DBSCAN (Python)](aero_LLM_07_T-SNE projection and DBSCAN clustering (Python).md) | [Xem bài viết →](aero_LLM_07_T-SNE projection and DBSCAN clustering (Python).md) |
| [Thách Thức Code: Tìm Lỗ Hổng Phân Cụm Bằng Bộ Lọc Bảng Chữ Cái Chữ X](aero_LLM_08_CodeChallenge cluster the x terms.md) | [Xem bài viết →](aero_LLM_08_CodeChallenge cluster the x terms.md) |
| [Phân Rã Token, Nhúng Và Phân Cụm Biểu Tượng Emojis Bằng Đồ Thị Mật Độ](aero_LLM_09_CodeChallenge Tokenize, embed, and cluster happy emojis.md) | [Xem bài viết →](aero_LLM_09_CodeChallenge Tokenize, embed, and cluster happy emojis.md) |
| [Phân Tích RSA (Representational Similarity Analysis) Giữa Các Mô Hình Ngôn Ngữ](aero_LLM_10_RSA (representational similarity analysis).md) | [Xem bài viết →](aero_LLM_10_RSA (representational similarity analysis).md) |
| [Phân Tích Độ Lệch RSA (Part 1): So Sánh Sự Bất Đồng Giữa Không Gian GloVe 50D và 300D](aero_LLM_11_CodeChallenge Compare embeddings with RSA (part 1).md) | [Xem bài viết →](aero_LLM_11_CodeChallenge Compare embeddings with RSA (part 1).md) |
| [Phân Tích Độ Lệch RSA (Part 2): Đối Chiếu Tương Quan Pearson Cho Khoảng Cách Cosine](aero_LLM_12_CodeChallenge Compare embeddings with RSA (part 2).md) | [Xem bài viết →](aero_LLM_12_CodeChallenge Compare embeddings with RSA (part 2).md) |
| [So Sánh Không Gian Nhúng: Word2Vec Và GPT-2 Qua Phân Tích RSA](aero_LLM_13_CodeChallenge Word2vec vs. GPT2.md) | [Xem bài viết →](aero_LLM_13_CodeChallenge Word2vec vs. GPT2.md) |
| [Bố Cục Đồ Thị Mạng (Network Graph) Thông Qua Ma Trận Cosine Similarity](aero_LLM_14_CodeChallenge Graph representation of cosine similarities.md) | [Xem bài viết →](aero_LLM_14_CodeChallenge Graph representation of cosine similarities.md) |
| [Số Học Tuyến Tính và Rút Trích Tương Đồng Giữa Các Từ Nhúng (Word Embeddings Analogies)](aero_LLM_15_Embeddings arithmetic and analogies.md) | [Xem bài viết →](aero_LLM_15_Embeddings arithmetic and analogies.md) |
| [Vỡ Mộng Về Số Học Vector Tương Đương (Soft-Coded Analogies) Trên Word2Vec](aero_LLM_16_CodeChallenge soft-coded analogies in word2vec.md) | [Xem bài viết →](aero_LLM_16_CodeChallenge soft-coded analogies in word2vec.md) |
| [Thiết Lập Và Diễn Giải Trục Ngữ Nghĩa Tuyến Tính (Linear Semantic Axes)](aero_LLM_17_Creating and interpreting linear semantic axes.md) | [Xem bài viết →](aero_LLM_17_Creating and interpreting linear semantic axes.md) |
| 📌 **[Khai Thác Thuật Toán k-NN Cho Tìm Kiếm Từ Đồng Nghĩa Trên BERT](aero_LLM_18_kNN for synonym-searching in BERT.md)** | [Xem bài viết →](aero_LLM_18_kNN for synonym-searching in BERT.md) |
| [Cạnh Tranh Tìm Từ Đồng Nghĩa BERT vs GPT: Cơ Chế Tokenization Đa Ký Tự](aero_LLM_19_CodeChallenge BERT v GPT kNN kompetition.md) | [Xem bài viết →](aero_LLM_19_CodeChallenge BERT v GPT kNN kompetition.md) |
| [Sự Dịch Chuyển Và Đồng Tồn Biểu Diễn Giữa Các Không Gian Nhúng](aero_LLM_20_Research on translating embeddings spaces.md) | [Xem bài viết →](aero_LLM_20_Research on translating embeddings spaces.md) |
| [Phân Tích Chùm Quang Phổ Suy Biến (Singular Value Spectrum) Của Không Gian Nhúng](aero_LLM_21_Singular value spectrum of embeddings submatrices.md) | [Xem bài viết →](aero_LLM_21_Singular value spectrum of embeddings submatrices.md) |
| [Ánh Xạ SVD Các Dải Điểm Nhúng Có Quan Hệ Chéo](aero_LLM_22_CodeChallenge SVD projections of related embeddings.md) | [Xem bài viết →](aero_LLM_22_CodeChallenge SVD projections of related embeddings.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
