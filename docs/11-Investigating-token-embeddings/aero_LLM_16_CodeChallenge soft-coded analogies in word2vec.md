
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
# Vỡ Mộng Về Số Học Vector Tương Đương (Soft-Coded Analogies) Trên Word2Vec

## Tóm tắt

Các tiêu đề báo chí khoa học đại chúng thường sử dụng một công thức vàng gây ấn tượng của mô hình Word Embeddings: `King - Man + Woman = Queen`. Phương trình vector học này tạo ra niềm tin rằng Mạng mô hình ngôn ngữ lớn hoạt động thuần túy trên công thức toán học khái niệm. Báo cáo đánh giá độc lập này mổ sẻ sự chắp vá và tính bất toàn của thuật toán học phép loại suy khoảng cách (Word analogies), phân tích sự hụt hẫng khi vận dụng "Soft-coding" trên Word2Vec so với thực tại của lý thuyết hình học.

---

## 1. Giới Tuyến Của GloVe Và Tính Sắc Bén Của Word2Vec

Hai kỳ phùng địch thủ thời tiền-Transformer là *GloVe* và *Word2Vec* nắm giữ hai cơ chế trích xuất ma trận (Factorization) khác biệt. 
- **GloVe (Global Vectors):** Thiết lập mạng lưới phân giải ma trận đếm số lần quy tẩm cận kề tần suất từ vựng (Co-occurrence text mapping). Nó nắm trong tay cấu trúc vĩ mô toàn thể tài liệu.
- **Word2Vec (CBoW / Skip-gram):** Thiết lập mô hình hồi quy trọng số nhắm vào việc điền từ còn thiếu giữa bộ vi mô khung cửa lưới (Context windows prediction). Việc mô phỏng chuỗi học tương tự quy luật Neural Networks hiện đại giúp Word2Vec bén nhạy triệt để với các quy luật giao thoa ngữ nghĩa học (Semantic relationships). 

Theo luận thuyết trên, khả năng thao túng phép Tương đồng Loại suy Toán học (Math analogies) của Word2Vec 300D được kỳ vọng phá vỡ ngưỡng cực hạn mà công cụ GloVe 50D để lại.

---

## 2. Kiểm Định Thất Bại Với Hàm Khai Khai Khái Niệm Tự Động (Soft-Coded Function)

Bằng việc gói gém cấu hình hàm Soft-coded nhận vào đầu vào linh hoạt:
$$ 
\mathbf{V}_{\text{Analogy}} = \mathbf{V}_{\text{Word1}} - \mathbf{V}_{\text{Word2}} + \mathbf{V}_{\text{Word3}} 
$$
Thuật toán phóng chiếu mũi tên $V_{\text{Analogy}}$ rà quét qua tập 400.000 lượng từ điển của Word2Vec thông qua Cosine Similarity để xuất kho Top 10 ứng cử viên gần nhất.

**Kiểm định 1 - Sự thần thánh hóa:**
Lệnh: `Tree` so với `Leaf`  $\approx$ `?` so với `Petal`. Trực giác sinh học con người dễ dàng xuất kho từ `Flower`.
Đội ngũ máy học trả về kết quả mờ mịt: Top ứng cử viên lộn xộn các từ `Willow Tree` (Cây Liễu).

**Kiểm định 2 - Đảo chiều trục:**
Lệnh: `Leaf` so với `Tree` $\approx$ `Petal` so với `Flower`.
Biên độ dự báo của mạng lưới từ vựng trượt dốc. Không có bất kỳ bóng dáng một đại lượng từ vựng nào nằm trong Top 10 chạm tới logic ý niệm. 

**Kiểm định 3 - Logic Giải Phẫu Người:**
Lệnh: `Finger` so với `Hand` $\approx$ `?` so với `Foot`. Đáp án chuẩn hóa là `Toe` (Ngón chân).
Mô hình toán học mớm lại từ `Pinky` (Ngón út) trôi nổi trong không gian nhiễu vector.

---

## 3. Bản Chất Của Kỹ Thuật Cộng Trừ Nhúng

Sự rạn nứt giữa huyền thoại `King-Man+Woman` và sự tàn bạo của các phép thử tự do ngoài lề đè bẹp kỳ vọng của giới nghiên cứu XAI về khả năng suy diễn quy nạp của Machine Learning chỉ dựa trên một Vector đơn hướng.
Các phép phân tích trừ - cộng Vector Analogies thực chất là một sự lãng mạn hóa học thuật. Sự diệu kỳ toán học này thường chỉ vận hành nhịp nhàng đối với những tập từ ngữ phổ quát cực mạnh (VD: Giới tính, vương quyền, quốc gia - thủ đô) đã được cọ xát hàng trăm triệu lần trong quá trình huấn luyện tạo thành một "Dòng chảy trọng tâm" cứng vững chắc ở ma trận $E$. Với những hệ thống cấu trúc tương quan nhỏ và hốc búa hơn, các ma trận Vector thường bị xé rão (Vector entanglement) và không tuân theo luật chơi Tịnh tiến độ dài tam giác.

Tuy vậy, những phép tính Vector căn nguyên nhất này không hề vứt đi. Chúng là bản nguyên nền móng để phát triển lên hệ quy chiếu siêu tinh vi Transformer. Tại kiến trúc ChatGPT hiện đại, những phép ma trận nhúng cộng trừ (Vector adjustments) không xảy ra một lần, mà bị giằng xé nhào nặn qua 96 vòng quy hồi Attention phi tuyến nhằm đúc ra một luồng suy nghĩ sắc lẹm thay vì chỉ là bề mặt của Vector Tĩnh.

---

## Tài liệu tham khảo

1. **Mikolov, T., et al. (2013).** *Distributed Representations of Words and Phrases and their Compositionality.* NIPS. (Khai sinh kỹ thuật Word2Vec và phép loại suy King-Queen).
2. **Levy, O., & Goldberg, Y. (2014).** *Linguistic Regularities in Sparse and Explicit Word Representations.* CoNLL. (Chỉ trích lỗ hổng toán học vector truyền thống).
3. Tài liệu thực hành lập trình *CodeChallenge soft-coded analogies in word2vec*.
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
| 📌 **[Vỡ Mộng Về Số Học Vector Tương Đương (Soft-Coded Analogies) Trên Word2Vec](aero_LLM_16_CodeChallenge soft-coded analogies in word2vec.md)** | [Xem bài viết →](aero_LLM_16_CodeChallenge soft-coded analogies in word2vec.md) |
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
