Khảo sát và Phân tích Toán học Embedding Tiền huấn luyện GloVe

Từ Ma trận Đồng xuất hiện đến Cấu trúc Hình học Không gian Từ vựng

⸻

Tóm tắt

Dựa trên tài liệu đính kèm “Exploring GloVe Pretrained Embeddings”, bài viết này trình bày một phân tích khoa học về embedding tiền huấn luyện GloVe, bao gồm cơ sở lý thuyết, hàm mục tiêu tối ưu, cấu trúc hình học của không gian vector, và các đặc tính ngữ nghĩa học được học từ thống kê đồng xuất hiện toàn cục.

Bài viết đồng thời mở rộng bằng các nguồn học thuật nền tảng (Pennington et al., 2014; Mikolov et al., 2013; Levy & Goldberg, 2014) và cung cấp các công thức toán học minh hoạ chi tiết.

⸻

1. Giới thiệu

Biểu diễn từ (word representation) là nền tảng của nhiều hệ thống xử lý ngôn ngữ tự nhiên (NLP).

Mục tiêu là xây dựng ánh xạ:

E: V \rightarrow \mathbb{R}^d

Trong đó:
	•	V: tập từ vựng
	•	d: số chiều embedding

Khác với Word2Vec (dựa trên ngữ cảnh cục bộ), GloVe khai thác thống kê toàn cục của ma trận đồng xuất hiện.

⸻

2. Ma trận Đồng xuất hiện

Giả sử một corpus có tổng số từ T.

Định nghĩa:

X_{ij} = \text{số lần từ } w_j \text{ xuất hiện trong cửa sổ ngữ cảnh của } w_i

Tổng số lần xuất hiện của w_i:

X_i = \sum_j X_{ij}

Xác suất đồng xuất hiện:

P_{ij} = \frac{X_{ij}}{X_i}

⸻

3. Trực giác Tỷ lệ Xác suất

Pennington et al. (2014) lập luận rằng tỷ lệ xác suất đồng xuất hiện mang thông tin ngữ nghĩa:

\frac{P_{ik}}{P_{jk}}

Ví dụ:
	•	i = ice
	•	j = steam
	•	k = solid

Ta kỳ vọng:

\frac{P(\text{solid}|\text{ice})}{P(\text{solid}|\text{steam})} \gg 1

Do đó, embedding nên mã hóa các tỷ lệ này.

⸻

4. Hàm Mục tiêu của GloVe

GloVe tìm vector w_i và \tilde{w}_j sao cho:

w_i^\top \tilde{w}_j + b_i + b_j \approx \log X_{ij}

Hàm mất mát:

J = \sum_{i,j} f(X_{ij})
\left(
w_i^\top \tilde{w}_j + b_i + b_j - \log X_{ij}
\right)^2

Trong đó:

f(x) =
\begin{cases}
(x/x_{max})^\alpha & x < x_{max} \\
1 & \text{otherwise}
\end{cases}

Thường:

\alpha = 0.75

⸻

5. Liên hệ với PMI (Pointwise Mutual Information)

PMI được định nghĩa:

PMI(i,j) = \log \frac{P_{ij}}{P_i P_j}

Levy & Goldberg (2014) chỉ ra rằng Word2Vec với negative sampling xấp xỉ phân rã ma trận:

PMI(i,j) - \log k

GloVe gần tương đương với việc factorize ma trận log-count.

Do đó:

w_i^\top \tilde{w}_j \approx PMI(i,j)

⸻

6. Hình học của Không gian Embedding

Embedding sau huấn luyện nằm trong:

\mathbb{R}^d

Khoảng cách cosine:

\cos(\theta) =
\frac{w_i^\top w_j}
{\|w_i\| \|w_j\|}

Phản ánh độ tương đồng ngữ nghĩa.

⸻

6.1 Quan hệ Tuyến tính

Một tính chất nổi bật:

w_{king} - w_{man} + w_{woman} \approx w_{queen}

Điều này có thể diễn giải:

(w_{king} - w_{man}) \approx (w_{queen} - w_{woman})

Cho thấy tồn tại các hướng ngữ nghĩa trong không gian vector.

⸻

7. Phân tích Phổ Trị riêng (Eigenvalue Spectrum)

Ma trận đồng xuất hiện:

X \in \mathbb{R}^{|V| \times |V|}

Phân rã SVD:

X = U \Sigma V^\top

Embedding tương đương với chọn:

W = U_d \Sigma_d^{1/2}

Phổ trị riêng thường tuân theo luật Zipf:

\lambda_r \propto \frac{1}{r^\beta}

Theo George Kingsley Zipf.

⸻

8. Entropy và Thông tin

Entropy của phân bố từ:

H(W) = -\sum_i P(w_i)\log P(w_i)

Mutual information giữa hai từ:

I(i;j) = \sum_{i,j} P_{ij} \log \frac{P_{ij}}{P_i P_j}

GloVe học embedding sao cho:

w_i^\top w_j \approx I(i;j)

⸻

9. Độ phức tạp Tính toán

Giả sử số phần tử khác 0 của X là |X|.

Độ phức tạp:

O(|X|d)

So với Transformer như BERT:

O(n^2 d)

GloVe hiệu quả hơn cho embedding tĩnh.

⸻

10. Hạn chế của GloVe
	1.	Embedding tĩnh
	2.	Không phụ thuộc ngữ cảnh
	3.	Không mô hình hóa thứ tự từ

Biểu diễn cố định:

e(w) = \text{hằng số}

Trong khi mô hình ngữ cảnh:

e_t = f(w_1,\dots,w_T)

⸻

11. Thực nghiệm Khám phá Embedding

Các phép phân tích thường dùng:
	•	PCA:

Z = XW
	•	t-SNE:

P_{ij} \propto \exp(-\|x_i-x_j\|^2)

Cho thấy các cụm ngữ nghĩa rõ ràng:
	•	Quốc gia
	•	Giới tính
	•	Số nhiều

⸻

12. Kết luận

GloVe dựa trên nguyên lý:

w_i^\top w_j \approx \log X_{ij}

Embedding học được:
	•	Cấu trúc tuyến tính
	•	Quan hệ ngữ nghĩa
	•	Thông tin toàn cục

Mặc dù đã bị thay thế trong nhiều ứng dụng bởi mô hình Transformer, GloVe vẫn là nền tảng lý thuyết quan trọng trong biểu diễn từ phân bố.

⸻

Tài liệu tham khảo
	1.	Pennington, Socher & Manning (2014). GloVe: Global Vectors for Word Representation.
	2.	Mikolov et al. (2013). Efficient Estimation of Word Representations.
	3.	Levy & Goldberg (2014). Neural Word Embedding as Implicit Matrix Factorization.
	4.	Shannon (1948). A Mathematical Theory of Communication.
	5.	Zipf (1935). The Psycho-Biology of Language.

