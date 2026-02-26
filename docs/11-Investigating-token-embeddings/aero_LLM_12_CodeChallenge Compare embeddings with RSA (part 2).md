
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
# Phân Tích Độ Lệch RSA (Part 2): Đối Chiếu Tương Quan Pearson Cho Khoảng Cách Cosine

## Tóm tắt

Nối tiếp nghiên cứu tiền đề ở cấu hình GloVe 50D và GloVe 300D (trong bài *CodeChallenge Compare embeddings with RSA part 1*). Với hai ma trận Tương quan Góc $S_{50D}$ và $S_{300D}$ có cùng kích thước mặt phẳng tọa độ ($20 \times 20$), chúng ta bắt đầu kích hoạt cơ chế đánh giá độ giống nhau thông qua thuật toán Phân tích Tương tự Biểu diễn (RSA - Representational Similarity Analysis). Điểm đặc biệt của báo cáo này là cách lý giải toán học để vượt qua định luật thông thường, lý giải tại sao phép tính tương quan lại được thực hiện bởi Thống kê độ lệch (Pearson Correlation) thay vì Góc khoảng cách (Cosine Similarity).

---

## 1. Thiết Lập Điểm Giao Cắt và So Sánh Tam Giác Phẳng

Cả 2 ma trận Cosine Similarity $S_{50D}$ và $S_{300D}$ hoàn toàn song song nhau về mặt Token Indices (Các vị trí từ vựng như "apple", "galaxy", ghép với "couch" nằm ở y hệt các tọa độ $(i, j)$).
Theo chuẩn mực RSA, chúng ta không được phép dùng trực tiếp góc ma trận $20 \times 20$, vì tính đối xứng hình học sẽ tạo ra Dư thừa Thống kê (Redundant duplicates) và Đường chéo chính tự tương quan $\equiv 1.0$ sẽ bóp méo kết quả.
Do đó, thuật toán sẽ phẳng hóa (Flattening) các phần tử thuộc tam giác trên (Upper Triangular Data):
$$ 
\mathbf{v}_{50D} = \text{Upper}(S_{50D}) \in \mathbb{R}^{\frac{20 \times 19}{2}}
$$
$$ 
\mathbf{v}_{300D} = \text{Upper}(S_{300D}) \in \mathbb{R}^{190} 
$$

---

## 2. Hệ Số RSA: Tại Sao Pearson Là Bằng Chứng Tranh Tụng Hoàn Hảo?

Bây giờ ta có 2 vec-tơ mảng một chiều đại diện cho "hệ thống lưới tọa độ khái niệm". Chúng ta sẽ tính RSA bằng Hệ Số Thống Kê Pearson ($\rho$).

### Điểm Mù Của Cosine Nhúng
Mặc dù $S_{50D}$ và $S_{300D}$ được tạo ra bởi **Cosine Similarity**, bước đối chiếu RSA lại tuyệt đối cấm kỵ áp dụng Cosine Similarity một lần nữa. 
Lý do nằm ở hiện tượng Tịnh tiến điểm trung vị (Mean Offset Shifts).
Thông qua các kiểm định đồ thị Scatter Plots, GloVe 50D có băng thông phổ rộng, biến thiên từ $[-0.4, 0.8]$.
GloVe 300D lại bị siết chặt vào dải băng thông hẹp hơn từ $[-0.2, 0.6]$.
Sự khác biệt trung bình cộng (Mean variance) này sẽ làm điểm Cosine rớt xuống đáy vì Cosine lấy mốc 0 làm gốc Vector, cho rằng 2 phổ dữ liệu này bị lệch phương hướng nội hàm.

### Sức Mạnh Tuyệt Đối Của Pearson Thống Kê
Hệ số r Pearson:
$$ 
\text{RSA} = \rho(\mathbf{v}_{50D}, \mathbf{v}_{300D}) = \frac{\text{Cov}(\mathbf{v}_{50D}, \mathbf{v}_{300D})}{\sigma_{50D} \sigma_{300D}} 
$$
Thuật toán này **trừ đi chính điểm trung bình tâm** (mean-centering data) mỗi bên, tước bỏ và cạo sạch yếu tố "Global offsets". 
Hệ số Pearson chỉ xét hỏi một tính chất duy nhất của sự liên kết: *"Khi lực kết nối ở 50D nhích lên cao hơn, thì điểm tương quan 300D có nhích theo một nhịp điệu tương khắc hay không?"*

### Kết Luận Từ Chỉ Số
Thực nghiệm rà quét đồ thị Scatter 190 cặp so sánh cho thấy $\text{RSA Score} \approx \mathbf{0.90}$ (Cực kỳ mạnh). 
Một trục đường phân phối tuyến tính hẹp được nối kết chắc chắn, minh chứng cho một học thuyết quan trọng trong không gian Embeddings: **Bản ngã của một mạng lưới từ vựng không nằm ở trị số tuyệt đối của Không gian chiều, mà nằm ở Tỷ Lệ Khoảng Cách Tương Đối theo hệ quy chiếu.** Dù là 50D hay 300D, thứ tự logic (Semantic structures) của chúng là một bản chụp sao chép gần như đồng bộ vô cực.

---

## 3. Hệ Quả Khóa Của Bài Toán So Sánh Kiến Trúc

Hiệu suất hoạt động của 50D mang theo một sự phân phối tản mác, cho thấy tính nhạy cảm của nó ở mức thấp. Tuy nhiên, nó vẫn giữ trọn đạo hàm từ với mã hóa 300D. 

Trong các thực tiễn về Machine Learning ứng dụng, nếu bài toán đòi hỏi một nguồn tài nguyên eo hẹp (VD: Chạy Local LLMs trên Mobile App) và chỉ cần nhóm rã các khối khái niệm to lớn (Topic clustering), sự thu gọn xuống kích thước cực tiểu (VD như mô hình thu gọn) hoàn toàn cung cấp một bức tranh toàn cảnh không quá chênh lệch nhờ tính đồng nhất của hệ điểm nhúng mạng nơ-ron chia sẻ sức mạnh cấu trúc lõi. 

---

## Tài liệu tham khảo

1. **Abnar, S., et al. (2019).** *Blackbox meets blackbox: Representational Similarity and Stability Analysis of Neural Language Models.* BlackboxNLP.
2. Tài liệu kỹ thuật nâng cao *Compare embeddings with RSA (part 2)*.
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
| 📌 **[Phân Tích Độ Lệch RSA (Part 2): Đối Chiếu Tương Quan Pearson Cho Khoảng Cách Cosine](aero_LLM_12_CodeChallenge Compare embeddings with RSA (part 2).md)** | [Xem bài viết →](aero_LLM_12_CodeChallenge Compare embeddings with RSA (part 2).md) |
| [So Sánh Không Gian Nhúng: Word2Vec Và GPT-2 Qua Phân Tích RSA](aero_LLM_13_CodeChallenge Word2vec vs. GPT2.md) | [Xem bài viết →](aero_LLM_13_CodeChallenge Word2vec vs. GPT2.md) |
| [Bố Cục Đồ Thị Mạng (Network Graph) Thông Qua Ma Trận Cosine Similarity](aero_LLM_14_CodeChallenge Graph representation of cosine similarities.md) | [Xem bài viết →](aero_LLM_14_CodeChallenge Graph representation of cosine similarities.md) |
| [Số Học Tuyến Tính và Rút Trích Tương Đồng Giữa Các Từ Nhúng (Word Embeddings Analogies)](aero_LLM_15_Embeddings arithmetic and analogies.md) | [Xem bài viết →](aero_LLM_15_Embeddings arithmetic and analogies.md) |
| [Vỡ Mộng Về Số Học Vector Tương Đương (Soft-Coded Analogies) Trên Word2Vec](aero_LLM_16_CodeChallenge soft-coded analogies in word2vec.md) | [Xem bài viết →](aero_LLM_16_CodeChallenge soft-coded analogies in word2vec.md) |
| [Thiết Lập Và Diễn Giải Trục Ngữ Nghĩa Tuyến Tính (Linear Semantic Axes)](aero_LLM_17_Creating and interpreting linear semantic axes.md) | [Xem bài viết →](aero_LLM_17_Creating and interpreting linear semantic axes.md) |
| [Khai Thác Thuật Toán k-NN Cho Tìm Kiếm Từ Đồng Nghĩa Trên BERT](aero_LLM_18_kNN for synonym-searching in BERT.md) | [Xem bài viết →](aero_LLM_18_kNN for synonym-searching in BERT.md) |
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
