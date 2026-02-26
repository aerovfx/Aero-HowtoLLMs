
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
# Nghệ Thuật Vẽ Bản Đồ Nhiệt Ma Trận Nhúng Bằng Cường Độ Từ (Coloring Cosine Similarity)

## Tóm tắt

Trí não con người là một cỗ máy nhận diện phổ thị giác (Visual pattern recognition), nó bế tắc hoàn toàn trước các bức tường ma trận số nguyên. Đóng vai trò làm cầu nối giữa hệ thống kỹ thuật tuyến tính và cảm thụ sinh lý của kỹ sư học máy, bài mô phỏng này dùng phương pháp Min-Max Scaling của Cường Độ Vector và Góc Tọa Độ để bóc tách một dạng Bản Đồ Nhiệt Văn Bản (Heatmap Overlays Text) trực tiếp trên các đoạn tài liệu Wikipedia (VD: Georgia/Algae Fuel/Purple). Kỹ thuật này giúp phát quang được sự lười biếng phân loại của hệ thống Tokenizer LLMs.

---

## 1. Công Cụ Khuếch Đại Khoảng Cách Mạng Bằng Độ Lớn Vector Hình Học

Khác với khoảng cách hai chiều, Độ Lớn Kích Thước (Vector Magnitude / L2-Norm) của một Embeddings vector (Khoảng cách điểm đó tính từ lõi $0$ của Không gian học) được tính bằng hàm Sum of Squares:
$$ 
\|v\| = \sqrt{\sum_{i=1}^{D} v_i^2} 
$$
Với BERT, sự biến vi mô phân tử chỉ nằm tản mác từ dải $[0.8, 1.6]$.

Để dùng thước đo này gán vào thang Gradients Màu RGB (Heatmap Red color map), ta phải nén ép khoảng biến thiên dị biệt trên bằng hàm Cân Kế Tuyến Tính:
$$ 
\text{Scaled } \|v\| = \frac{\|v\| - \text{Min}}{\text{Max} - \text{Min}} 
$$
Kỹ thuật này bảo lưu trọn vẹn điểm đồ thị tỉ lệ (Dữ liệu Scale tịnh tiến), nhưng đóng khung kết quả cứng vào $[0.0, 1.0]$. 
Khi nhuộm sắc lên văn bản, kết quả thị giác hóa mang lại điều kinh ngạc:
- **Pale Trắng Bạc (Min Length):** Toàn bộ giới từ ngữ pháp, dấu câu yếu như: *of, it's, comma (,), period (.), a, the, because, at*. Chúng chỉ nằm cách lõi 0 một quãng ngắn (chìm dưới đáy xã hội học sâu).
- **Red Sẫm Máu (Max Length):** Các từ ngữ mang tính khái niệm độc bảng dày đặc: *neoclassical, crossroads, contention, nouveau, various*. 

Các hạt từ vựng mang đặc tính tần suất học thấp (Rare vocab / High specialized), xuất hiện lẻ tẻ trên tập đào tạo bị mạng Lõi hệ thần kinh phóng đẩy văng mạnh thành những "tọa độ trôi dạt" ra xa Origin. 

---

## 2. Truy Vết Cosine Gốc Trực Tiếp Lên Phổ Vệ Tinh Trực Quan

Ứng dụng bản đồ nhiệt thứ hai được tiến hành qua cơ chế Bóc tách Cosine chuỗi: Nhuộm nền một Tokenized Document theo cường độ Cosine Similarity so với từ liền trước nó (Ngoại trừ phần tử thứ 0 trả kết quả `NaN`, buộc phải dùng hàm `np.nanmin` để triệt tiêu lỗi sập thuật toán Zero-division).

Với thuật toán gán Color Overlay lên đoạn văn *Algae fuel*, những từ ngữ bị chìm đỏ gắt bộc lộ ra các bộ đôi bài trùng cố hữu trong ngôn ngữ người như:
- `practical` + `significance` 
- `algae` + `fuel` 
- `fossil` + `fuels`

Máy học không hiểu sinh học hữu cơ, nó chỉ là một con chíp lặp chuỗi thống kê khi thấy Algae & Fuel cọ sát nhau lặp đi lặp lại tạo thành một vệt dính kết không gian. 

---

## 3. Khóa Target Tìm Điểm Gây Mòn Sự Tính Toán

Ngoài dạng tìm đồng bộ tiếp nối, Bản Đồ Nhiệt có khả năng ghim chết (Pinning) mục tiêu thành tâm đối tượng. Trong tài liệu Wikipedia nói về MÀU TÍM, chúng ta khóa Token `purple` làm tâm ($V_{\text{target}}$) .
Lệnh quét chổi tạo Heatmap quét toàn bộ đại lục văn bản, tất cả các từ trong văn bản đều bị làm Scale Cosine đối chiếu tới duy nhất tâm `purple`.
- Lúc này cường độ đỏ dâng lên ở các cụm từ liên kết địa đồ với sắc thái tím.
- Các vệ tinh mang tên `purple` nếu xuất hiện lặp lại trong văn bản, thuật toán ép điểm Normalize Cosine max $= 1.0 \to \text{Red}_{100\%}$. Hệ thống nhận dạng đây đích xác là hiện tượng Gương phản chiếu tự thân trong mạng Vector Space (Autocorrelation).

Phép toán màu hóa không dùng để vẽ đồ án mỹ thuật, mà trang bị cho các kỹ sư Explainable AI (XAI) khả năng đọc lướt nhanh cơ chế tập trung ngầm của Attention, phơi bày ra cách trí thông minh sinh học được định dạng lại dưới lớp mặt nạ Tensor thần kinh.

---

## Tài liệu tham khảo

1. **Karpathy, A., et al. (2015).** *Visualizing and Understanding Recurrent Networks.* ICLR (Phương trình đánh giá lớp vỏ nhiệt XAI).
2. Tài liệu thực hành lập trình số liệu XAI - *Coloring cosine similarity visualization.*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [aero_LLM_01_CodeChallenge Cosine similarity (advanced) (part 1).md](aero_LLM_01_CodeChallenge Cosine similarity (advanced) (part 1).md) | [Xem bài viết →](aero_LLM_01_CodeChallenge Cosine similarity (advanced) (part 1).md) |
| [aero_LLM_02_CodeChallenge Cosine similarity (advanced) (part 2).md](aero_LLM_02_CodeChallenge Cosine similarity (advanced) (part 2).md) | [Xem bài viết →](aero_LLM_02_CodeChallenge Cosine similarity (advanced) (part 2).md) |
| [Theo Dõi Dòng Chảy Cosine Similarity Trên Trục Văn Bản Chuyên Tuần Tự (Word Sequences)](aero_LLM_03_CodeChallenge Cosine similarity in word sequences.md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Cosine similarity in word sequences.md) |
| 📌 **[Nghệ Thuật Vẽ Bản Đồ Nhiệt Ma Trận Nhúng Bằng Cường Độ Từ (Coloring Cosine Similarity)](aero_LLM_04_CodeChallenge Coloring cosine similarity.md)** | [Xem bài viết →](aero_LLM_04_CodeChallenge Coloring cosine similarity.md) |
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
