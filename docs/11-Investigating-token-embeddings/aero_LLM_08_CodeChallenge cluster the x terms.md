
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
# Thách Thức Code: Tìm Lỗ Hổng Phân Cụm Bằng Bộ Lọc Bảng Chữ Cái Chữ X

## Tóm tắt

Giáo trình thực nghiệm tiếp nối sự kết hợp của Không Gian Kéo Chiều t-SNE và Thuật Toán Bọc Mật Độ DBScan từ GPT-2. Trong chuyên đề giới hạn này (CodeChallenge), một rào cản thử nghiệm về sức chịu đựng của thuật toán được tạo ra bằng cách thiết lập một lồng ép Lọc Nhiễu Thô (Regex Filtering): Việc chọn lọc ngẫu nhiên các từ vựng phải chứa chữ "x" và có độ dài chuỗi ký tự hạn chế. Phép thử đếm cụm (Cluster Counts) biến thiên qua ma trận quét rào Epsilon và Min Samples chứng tỏ cho sự bất ổn trong phân rã dữ liệu máy học bằng tham số định trước.

---

## 1. Màng Lọc Tần Số Ký Tự Ngẫu Hình (Character Threshold Filters)

Thử thách khởi đầu không phải thuật toán không gian mà ở cơ chế khai lọc từ vựng. Trí thông minh của ngôn ngữ lớn luôn bị rào cản bởi rác (Noises). Bộ lọc lưới chài được đặt mục tiêu: 
- Token chỉ được sở hữu bộ gõ chứa kí tự `"x"`.
- Token bị chặn ép độ dài chữ trong khoảng tự do từ $4 \leq L \leq 8$, cấm không tính cả khoảng cách đệm đầu Prefix. 

Ví dụ với lệnh chặn lùi không gian: `token = " exp"`. Do khoảng cách trắng đầu, Token này có tổng chiều dài String Lenght $= 4$, nhưng bỏ đi khoảng trắng sẽ thành `exp` (Độ dài bằng $3$). Buộc từ này rớt khỏi thang đếm.
Hay token `" axis"` (5 chữ cái, gồm khoảng trắng) bị lọc thành `axis` (Độ dài $4$) vượt cửa môn lưới lọt vào mảng dữ liệu phân tích.

Sự thanh lọc tạo thành mảng dữ liệu thô cô đặc $\mathbf{N} = 514$ điểm tokens chứa "x" rải rác. Vốn dĩ những chữ cái này vô nghĩa về mặt liên kết học sâu, chúng bị ép gượng ép đứng chung để đối chiếu kết tủa không gian.

---

## 2. Ranh Giới Giữa Giải Thích Nhị Phân Và Phân Cụm Ngữ Nghĩa

Khi đem $514$ token này qua máy nghiền Dimension Reduction của T-SNE với tham số `perplexity = 5.0` để áp về tọa độ lưới không gian màn hình phẳng 2D. 
Bằng mắt thường, ta thấy vô vàn các hạt cô đặc bị nén lại. Nhưng con số phân tách của DBScan (`Epsilon = 6.0`, `Min_samples = 3.0`) lại trút xuống tận $65$ Cluster phân tách nhỏ cho $514$ điểm số.  (Sau này chỉ cần giảm epsilon xuống mức ranh giới hẹp, tỷ lệ này sụp hoàn toàn).

Thế nhưng, điểm gây nghiện nhất của công trình là tính **Dị Chủng Ma Trận**:
Về mặt tích cực, DBScan đã nhóm được một chuỗi đồng nghĩa cấu trúc cực chất như `texture` với `textures` và `Texture`; hoặc `exile`, `exiled`, `expel` quy hội thành khối cụm Trừng trị Đày đọa. 

Nhưng, nếu t-SNE mắc lỗi gom sai khoảng cách, DBScan sẽ bị dắt mũi nhốt luôn các khái niệm tương quan thảm họa lại thành 1 rọ (Siêu Cụm Bất Quy Tắc):
Từ vựng `galaxy`, `galaxies` (Ngân hà) lại bị t-SNE và DBscan đóng nhốt thành một hệ siêu vòng với những chữ viễn tượng vi tính như `syntax`, `regex`, `codex`. Lỗi xuất phát bởi yếu tố Epsilon có đường kính bán kính (Reachability Range) đủ dài để nuốt trọn cả các cụm lân bang thành một "Thành phố rác" khổng lồ, khiến thuật toán AI diễn dịch "Ngân Hà" là một cấu trúc "Ngôn Ngữ Lập Trình Hệ Thống" - Xuyên thủng kiến trúc giải phẫu logic (Mechanistic error).

---

## 3. Khảo Sát Tính Mong Manh Của Siêu Tham Số (Hyper-parameters Grid Search)

Hậu quả của hiện tượng gom chéo siêu cụm dẫn đến thực nghiệm thứ 3: Xây dựng một móng ma trận Quét lưới vòng lặp For (Grid Search).

1. **Khóa `min_samples = 3`**: Thử nghiệm Epsilon trượt biên từ $2 \to 20$. 
Đồ thị vạch lộ ra cấu trúc đồ thị đổ đèo tiêu chuẩn (Curve descending). Khi $Epsilon = 16$, độ bạo của mảng nối vòng làm 514 điểm nuốt nhau, báo cáo thuật toán có số lượng cụm $k$ trượt thẳng về mức $1 \to 3$ Cụm Mega. Không còn chi tiết nhỏ vi mô (Micro details collapsed).
2. **Khóa `Epsilon = fixed`**: Khảo nghiệm tham số thứ 2, thay đổi Min-Samples từ $2 \to 20$. 
3. **Hiệu Ứng Bảng Nóng Xoay Chiều (Heatmap Matrix Search)**: Chạy hai vòng lặp lồng chéo, tạo thành khung $19 \times 15$ giao diện Heatmaps để tìm ra "vành đai vàng - Goldilocks zone" nơi mà số cụm không nằm mấp mé 0 và cũng không vụn vỡ quá mức.

Tính "Chỉnh đốn thủ công - Cherry picking" của giới học viện Machine Learning bộc lộ rõ: Kỹ sư có thể nhìn biểu đồ Scatter, và nắn gân các hệ số `Epsilon` / `Min_Samples` cho tới khi máy móc rặn ra đúng thành quả mà trực giác mong đợi. Cách gọt giũa này tạo ra sự bẻ cong công lý trong khoa học dữ liệu, minh định cho một kết luận: "Kế hoạch phân rã Unsupervised Clustering" không thực tế mang hàm lượng khách quan trừ phi đi kèm một bằng chứng lưới tham số mở để minh bạch sai số.

---

## Tài liệu tham khảo

1. **Ester, M., et al. (1996).** *A density-based algorithm for discovering clusters in large spatial databases with noise.* KDD (Nền móng toán học siêu cụm Epsilon).
2. Tài liệu giải mã và bài tập thực chiến bộ dữ liệu - *CodeChallenge cluster the x terms.*
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
| 📌 **[Thách Thức Code: Tìm Lỗ Hổng Phân Cụm Bằng Bộ Lọc Bảng Chữ Cái Chữ X](aero_LLM_08_CodeChallenge cluster the x terms.md)** | [Xem bài viết →](aero_LLM_08_CodeChallenge cluster the x terms.md) |
| [Phân Rã Token, Nhúng Và Phân Cụm Biểu Tượng Emojis Bằng Đồ Thị Mật Độ](aero_LLM_09_CodeChallenge Tokenize, embed, and cluster happy emojis.md) | [Xem bài viết →](aero_LLM_09_CodeChallenge Tokenize, embed, and cluster happy emojis.md) |
| [Phân Tích RSA (Representational Similarity Analysis) Giữa Các Mô Hình Ngôn Ngữ](aero_LLM_10_RSA (representational similarity analysis).md) | [Xem bài viết →](aero_LLM_10_RSA (representational similarity analysis).md) |
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
