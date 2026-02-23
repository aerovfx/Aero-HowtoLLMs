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
