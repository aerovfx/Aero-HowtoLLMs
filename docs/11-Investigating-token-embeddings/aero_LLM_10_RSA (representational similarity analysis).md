
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
# Phân Tích RSA (Representational Similarity Analysis) Giữa Các Mô Hình Ngôn Ngữ

## Tóm tắt

Representational Similarity Analysis (RSA) là một phương pháp luận ban đầu được phát triển trong Khoa học Thần kinh (Neuroscience) nhằm so sánh phổ điện não đồ với mô hình tính toán. Ngày nay, thuật toán này trở thành một trong những mũi nhọn của lĩnh vực Phân tích Biểu diễn Ngôn ngữ (Representational Analysis) giúp chúng ta đối chiếu, so sánh và định lượng sự tương đồng giữa các ma trận nhúng (Embeddings matrices) vốn có không gian chiều (dimensionality) hoàn toàn lệch nhau (Ví dụ: So sánh Word2Vec 300 chiều với GPT-2 768 chiều). Bài viết dưới đây trình bày nguyên lý toán học và quy trình thực hiện cấu trúc RSA trong ngữ cảnh xử lý ngôn ngữ học máy.

---

## 1. Giới thiệu

Với sự bùng nổ của các mô hình nhúng (Embeddings) như GloVe, Word2Vec, BERT hay GPT, một câu hỏi lớn được đặt ra: *Làm sao để biết liệu hai mô hình này có chung một cách hiểu về mặt vật lý không gian cho một bộ từ vựng hay không?* 

Sự lệch pha về chiều không gian vector của các ma trận khiến cho chúng ta không thể sử dụng các phép trừ trực tiếp (direct subtraction) hay khoảng cách Euclidean giữa hai mạng mô hình. RSA giải quyết vấn đề này bằng cách chắt lọc các đặc trưng tương quan khoảng cách *bên trong* vùng dữ liệu của mỗi mô hình trước, sau đó mới so sánh khối đặc trưng tương quan (Similarity structures) *giữa* hai mô hình.

---

## 2. Nguyên Lý Toán Học Của RSA

Khung toán học của RSA trải qua 3 bước cốt lõi: 

### 2.1 Ma trận Khoảng Cách / Tương Quan Cục Bộ (Similarity Matrices)

Cho ma trận nhúng $E_1 \in \mathbb{R}^{N \times D_1}$ từ mô hình 1 (Ví dụ Word2Vec kích thước $D_1 = 300$) và $E_2 \in \mathbb{R}^{N \times D_2}$ từ mô hình 2 (GPT, kích thước $D_2 = 768$), với $N$ là số lượng token ngôn ngữ chung giữa hai mô hình (phải đồng nhất thứ tự token).

Bước đầu tiên, RSA tính toán các Ma trận Tương quan nội bộ (viết tắt là Representational Similarity Matrix - RSM) cho từng không gian chiều:
$$ S_1 = \text{CosineSimilarity}(E_1) $$
$$ S_2 = \text{CosineSimilarity}(E_2) $$

Trong đó, mỗi phần tử $S(i, j)$ được cho bằng công thức nội tích ma trận Gram đã chuẩn hóa:
$$
S(i,j) = \frac{e_i \cdot e_j}{\|e_i\| \|e_j\|}
$$
Kết quả thu được là 2 ma trận vuông đối xứng kích thước $N \times N$, độc lập hoàn toàn với chiều không gian ban đầu $D_1$ hay $D_2$.

### 2.2 Trích Xuất Vector Tam Giác Thượng (Upper Triangular Unrolling)

Vì các ma trận $S_1$ và $S_2$ là đối xứng qua đường chéo $S(i,j) = S(j,i)$, và các giá trị trên đường chéo luôn bằng 1 ($S(i,i) = 1$), việc tính toán trên toàn bộ ma trận sẽ dẫn đến hiện tượng bơm phồng tương quan (inflation artifact). Do đó, ta chỉ trích xuất các thành phần không bị trùng lặp ở nửa trên tam giác (upper triangular part):
$$ 
\vec{v}_1 = \{ S_1(i, j) \mid i < j \}
$$
$$ 
\vec{v}_2 = \{ S_2(i, j) \mid i < j \}
$$
Số lượng các phần tử duy nhất sau khi bung ra là $\frac{N(N-1)}{2}$.

### 2.3 Phân Tích Pearson Correlation Giữa RSA

Bước cuối cùng là áp dụng hệ số Tương quan bình phương Pearson (hoặc Spearman rank correlation) giữa hai vector $\vec{v}_1$ và $\vec{v}_2$:

$$
\rho = \frac{\sum (\vec{v}_1 - \mu_{\vec{v}_1})(\vec{v}_2 - \mu_{\vec{v}_2})}{\sigma_{\vec{v}_1} \sigma_{\vec{v}_2}}
$$

Nếu $\rho$ tiến sát tới 1, ta kết luận rằng bất chấp việc được huấn luyện ở những nguồn dữ liệu khác nhau với số lượng lớp nơ-ron khác nhau, hai mô hình này sử dụng cùng một cấu trúc hình học tương quan để bảo toàn ngữ nghĩa từ vựng.

---

## 3. Ứng Dụng Khai Thác Độ Dư Thừa Của Neural Network

Trong tài liệu đính kèm, RSA được khai thác ở một biến thể thú vị: thay vì so sánh hai mô hình độc lập, ta so sánh nội bộ hai ma trận chia cắt từ một cụm nhúng đơn điệu. Bằng cách tách một ma trận 300 chiều thành hai khối 150 chiều D-chẵn (Even dimensions) và D-lẻ (Odd dimensions), chúng ta thu được sự tương đồng mã hóa $\rho \approx 0.8$. Sự lệch pha còn lại ($\sim 20\%$) tạo nên một lượng thông tin không đối xứng (Unique internal coding) bên cạnh phần dư thừa đặc trưng.

Việc đánh giá sự tương quan dư thừa (representational redundancy) giúp tối ưu bài toán nén và cắt bớt mô hình (Model Pruning) nhằm tăng tốc quá trình suy luận mà không giảm hiệu suất diễn giải của hệ thống trí tuệ.

---

## 4. Kết luận

Representational Similarity Analysis (RSA) được coi là một ống kính trung gian hoàn hảo để thu phóng và đối chiếu hai hộp đen AI độc lập bằng cách so sánh các đặc tính mối quan hệ thay vì giá trị vector thô. Khả năng loại bỏ tính không biểu diễn (Dimension elimination constraint) là nền tảng giúp phương pháp này trở thành một phép tính chuẩn trong lĩnh vực Alignment và Định lượng Khả năng Diễn giải (Interpretability).

---

## Tài liệu tham khảo

1. **Kriegeskorte, N., et al. (2008).** *Representational similarity analysis - connecting the branches of systems neuroscience.* Frontiers in Systems Neuroscience, 2. (Khoa học hệ thần kinh gốc của RSA).
2. **Abnar, S., et al. (2019).** *Blackbox meets blackbox: Representational Similarity and Stability Analysis of Neural Language Models.* Proceedings of the 2019 ACL Workshop BlackboxNLP.
3. **Chrupała, G., & Alishahi, A. (2019).** *Correlating neural and symbolic representations of language.* ACL.
4. Tài liệu bài giảng *Investigating token embeddings - RSA*.
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
| 📌 **[Phân Tích RSA (Representational Similarity Analysis) Giữa Các Mô Hình Ngôn Ngữ](aero_LLM_10_RSA (representational similarity analysis).md)** | [Xem bài viết →](aero_LLM_10_RSA (representational similarity analysis).md) |
| [Phân Tích Độ Lệch RSA (Part 1): So Sánh Sự Bất Đồng Giữa Không Gian GloVe 50D và 300D](aero_LLM_11_CodeChallenge Compare embeddings with RSA (part 1).md) | [Xem bài viết →](aero_LLM_11_CodeChallenge Compare embeddings with RSA (part 1).md) |
| [Phân Tích Độ Lệch RSA (Part 2): Đối Chiếu Tương Quan Pearson Cho Khoảng Cách Cosine](aero_LLM_12_CodeChallenge Compare embeddings with RSA (part 2).md) | [Xem bài viết →](aero_LLM_12_CodeChallenge Compare embeddings with RSA (part 2).md) |
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
