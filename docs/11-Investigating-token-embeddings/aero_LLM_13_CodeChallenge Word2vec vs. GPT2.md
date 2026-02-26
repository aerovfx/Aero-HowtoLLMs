
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
# So Sánh Không Gian Nhúng: Word2Vec Và GPT-2 Qua Phân Tích RSA

## Tóm tắt

Một rào cản chí mạng trong Nghiên cứu Dữ liệu Văn bản (NLP) là việc xác nhận chất lượng tương quan giữa hai cỗ máy sở hữu chiều kích nhúng (Embeddings dimension size) đôi đũa lệch. Ma trận của mô hình đại cương Word2Vec có 300 chiều ẩn (300D), trong khi GPT-2 nặng nề sở hữu 768 chiều nhúng (768D). Làm thế nào để giải phẫu và chẩn đoán liệu GPT-2 và Word2Vec có chia sẻ chung một "triết học toán học" ngôn từ hay không? Bài báo này sẽ vận dụng một đường vòng bằng giải tích không gian thông qua thủ thuật kết chiếu Hệ số Tương Quan Pearson nội hàm, nền tảng của phương thức **RSA (Representational Similarity Analysis)**.

---

## 1. Thiết Lập Điểm Khớp Giao (Intersection Point Matching)

Sự so sánh hai đa hình học không gian bắt buộc phải được gắn kết trên một tập đối tượng con neo đậu duy nhất. Phép lọc được thiết lập thông qua phân tách danh sách từ khóa ở cả hai tệp từ điển (Vocab Arrays) của hai Tokenizer (Word2vec token list và GPT tokenizer vocab). Giả sử hệ thống thiết lập bộ đếm quét trích lọc (iteration filtering) 100 từ có số lượng ký tự chính xác bằng 6 ($length = 6$ letters).

Thuật toán dò ngược Try-Catch exception sẽ tạo được một tập ma trận trung gian gồm số lượng $N=100$ chữ khớp lệnh có mặt trong cả 2 từ điển bất chấp sự lệch pha của chỉ mục Index, triệt tiêu mọi biến dị các phần phụ kiện token lỗi hoặc khoảng trống ảo (Spaces issues).

---

## 2. Kiến Thiết Khối Phân Giải Cục Bộ

Bất lực hoàn toàn trước phép trừ hoặc cộng tuyến tính giữa một vector 300D và 768D, phương pháp lấy đạo chéo bắt đầu với tính độc lập từng phe không gian một.
Trích lấy cụm thông tin vector của $N=100$ token trong hai hộp không gian, áp dụng ma trận tích vô hướng khoảng cách chéo Cosine Similarity:
$$ 
S_{W2V} = \text{CosineSim}(E_{\text{w2v-100}}) \in \mathbb{R}^{100 \times 100}
$$
$$ 
S_{GPT2} = \text{CosineSim}(E_{\text{gpt2-100}}) \in \mathbb{R}^{100 \times 100}
$$

**Chắt Cất Đại Lượng (Upper Triangular Tiling):** 
Dọc theo chéo chính (Diagonal elements), tất cả các thông số đều vô nghĩa vì chúng luôn $\equiv 1.0$ (Tự soi chiếu gương). Tương tự, mặt đối xứng chéo dưới (Lower triangular) cũng là thông tin vi phạm lỗi dư thừa. Do đó, chỉ môt mảnh tam giác trên cùng (Upper components extract) có trị số vô hướng $\frac{100 \times 99}{2} = 4950$ điểm dữ liệu thô được dàn phẳng thành vector dây một chiều $v_{w2v}$ và $v_{gpt2}$.

---

## 3. Pearson Correlation Lên Ngôi Của Sự Phi Tuyến

Đây chính là điểm giao mùa của phân tích. Liệu chúng ta có nên làm một phép đo khoảng cách Cosine Similarity giữa $v_{w2v}$ và $v_{gpt2}$ để cho RSA Score không? Cấu trúc của mạng nơ-ron hồi đáp: **Không được sử dụng Cosine Similarity cho cấu hình RSA so sánh, điểm số này luôn luôn phải chạy bằng chuẩn Pearson Correlation.**

Điều này xảy ra do định dạng Không gian Dịch tâm Dị hướng (Distribution offsets deviation): 
Quang phổ Cosine của Word2Vec luôn được chuẩn hóa rộng rãi nằm giữa khu vực khoảng $[ -0.2 , 0.5 ]$. Trong khi đó, tính chất khối lượng đồ thị học mạng biến áp tự hồi quy (Autoregressive Transformers networks) như GPT mang đến hiệu ứng chùm điểm tụ lõi nón - tất cả mọi Cosine Similarities của GPT-2 lơ lửng ở đỉnh dư dương luôn lớn hơn $0$, loanh quanh khoảng $[ 0.3 , 0.8 ]$.

Nếu giả tưởng ta ép ma trận Word2vec tịnh tiến xuống trừ đi $-1$ trị số (Mean Offset subtract 1), chỉ số Cosine Similarity đột ngột nhảy vực thay đổi phương hướng đồ thị toàn tập. Nhưng tính chất **Hệ số Pearson ($\rho$) không bao giờ gãy đổ**:
$$
\rho = \frac{\text{Cov}(v_{w2v}, v_{gpt2})}{\sigma_{\text{w2v}} \sigma_{\text{gpt2}}}
$$
Luật tính hiệp phương sai chia chuẩn độ lệch $Cov(X,Y)$ tự động loại bỏ mọi độ lệch trung bình tâm (global mean offsets shift), khiến Pearson Correlation chỉ xét dựa trên tính chất "*Chúng nhảy nhót lên và xuống cùng một biên độ hay không*". 

Kết cục của điểm $\rho$ tính được RSA Score cung cấp một chỉ số cao ấn tượng, thừa nhận việc máy học dự đoán ngôn ngữ GPT-2 trên Transformer hay mô hình cửa sổ bối cảnh nhỏ Continuous Bag-of-Words như Word2vec, sự kiến thiết thông triệt của ngôn ngữ loài người ở mức sâu nhất trong AI là tương đồng đáng kinh ngạc.

---

## Tài liệu tham khảo

1. **Abnar, S., et al. (2019).** *Blackbox meets blackbox: Representational Similarity and Stability Analysis of Neural Language Models.* Proceedings of the 2019 ACL Workshop BlackboxNLP.
2. **Kriegeskorte, N., et al. (2008).** *Representational similarity analysis.* 
3. Tài liệu đào tạo nâng cao *Investigating embeddings - CodeChallenge Word2vec vs. GPT2*.
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
| 📌 **[So Sánh Không Gian Nhúng: Word2Vec Và GPT-2 Qua Phân Tích RSA](aero_LLM_13_CodeChallenge Word2vec vs. GPT2.md)** | [Xem bài viết →](aero_LLM_13_CodeChallenge Word2vec vs. GPT2.md) |
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
