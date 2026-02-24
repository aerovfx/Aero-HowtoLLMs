# Phân tích Chuyên Sâu Các Tầng Ẩn Trong Mô Hình Ngôn Ngữ Lớn (LLMs): Đo Lường, Biểu Diễn và Giải Mã Nội Tại

## Tóm tắt (Abstract)
Bài viết này trình bày một hệ thống các phương pháp luận tiên tiến để phân tích và diễn dịch các biểu diễn nội tại trong các mô hình ngôn ngữ lớn (Big Language Models - LLMs) ở cấp độ từng tầng (layer-level). Dựa vào các phân tích trực quan về cơ chế Attention, không gian biểu diễn đa chiều, và lý thuyết thông tin, chúng tôi tổng hợp lại 5 khía cạnh cốt lõi: (1) Đo lường độ tương đồng Token trong các ma trận Q, K, V; (2) Phương pháp Phân biệt Đặc trưng biểu diễn (RSA); (3) Phân tích số chiều hiệu quả với PCA; (4) Đối chiếu Thông tin Tương hỗ với Hiệp phương sai trong phân tích cụm; và (5) Sử dụng Logit Lens để đọc các dòng chảy suy diễn ngầm.

---

## 1. Mở đầu
Khả năng diễn dịch cơ học (Mechanistic Interpretability) cố gắng biến LLMs từ những "hộp đen" thành cấu trúc có thể kiểm chứng. Một bước quan trọng là chuyển góc nhìn từ các nơ-ron rời rạc lên một cấp độ vĩ mô hơn: cấp độ tầng mạng (layer). Tại đây, sự dịch chuyển về biểu diễn từ vựng, ngữ pháp và ngữ nghĩa qua các tầng ẩn có thể được phân tích bằng những công cụ toán học và thống kê bài bản.

---

## 2. Đo Lường Sự Tương Đồng Tokens và Phân Tích RSA

### 2.1. Độ tương đồng Cosine trong Ma Trận Attention (Q, K, V)
Trong mỗi tầng Transformer, các Token được chiếu vào không gian Truy vấn (Query - $Q$), Khóa (Key - $K$) và Giá trị (Value - $V$). Để định lượng mức độ giống nhau về mặt phân bổ kích hoạt (activation) giữa các token trong cùng một hoặc khác ngữ cảnh, ta dùng **Độ tương đồng Cosine**.
Giả sử có hai vector $\mathbf{u}$ và $\mathbf{v}$, độ tương đồng được tính theo:
$$ \text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} $$
Thực nghiệm (Ví dụ trên GPT-2) cho thấy các kích hoạt đối với cùng một token ở những ngữ cảnh khác nhau luôn duy trì mức độ tương đồng nhất định. Đáng chú ý, các liên kết ma trận $K$ thường có sự tương đồng nội bộ lớn hơn $Q$, bộc lộ tính chất hấp thụ ngữ cảnh của khóa $K$.

### 2.2. Phân Tích Tương Đồng Biểu Diễn (Representational Similarity Analysis - RSA)
RSA cho phép chúng ta trả lời câu hỏi: *Hình học không gian thông tin trong $Q$ có giống với $K$ hay không?* 
Bằng cách xây dựng các ma trận khoảng cách / tương đồng $R_Q$ và $R_K$ cho tập n tokens, sau đó lấy chuỗi nửa tam giác trên (upper triangle) của cả 2 ma trận để tính hệ số tương quan tuyến tính (Pearson):
$$ r = \frac{\sum (R_{Q,i} - \bar{R}_Q)(R_{K,i} - \bar{R}_K)}{\sqrt{\sum (R_{Q,i} - \bar{R}_Q)^2 \sum (R_{K,i} - \bar{R}_K)^2}} $$
Các hiện tượng tụ cụm (grouping clustering) thường xảy ra, chứng minh mô hình xử lý một tập các từ đồng nghĩa học hoặc chung một phân lớp ngữ pháp gần nhau trong không gian học.

---

## 3. Khảo Sát Chiều Không Gian Hiệu Quả (Effective Dimensionality) bằng PCA
Dù số chiều nhúng ($d_{model}$) có thể lên tới 768 (GPT-2) hoặc hàng ngàn (Pythia xB), nhưng thông tin ý nghĩa thực chất có thể chạy trên một đa tạp không gian ít chiều hơn (Effective Dimensionality).

Kỹ thuật này áp dụng Phân tích thành phần chính (PCA) thông qua Phân rã giá trị đặc dị (SVD) trên ma trận kích hoạt tầng $X$ đã chuẩn hoá trung bình tâm:
$$ X = U \Sigma V^T $$
Từ ma trận đường chéo $\Sigma$ chứa các giá trị đặc dị (Singular values) $\sigma_i$, phần trăm phương sai mà thành phần $i$ giải thích lập nên công thức:
$$ r^2_i = \frac{\sigma_i^2}{\sum_{j=1}^n \sigma_j^2} \times 100\% $$
Khai thác đồ thị biến bạo tích lũy (Cumulative Variance Explained), ta xác định được **Số chiều hiệu quả** là số đặc dị cực tiểu cần giữ lại để đạt một ngưỡng (ví dụ 90% hay 95% phương sai). Hiện tượng co giãn (Expansion and Contraction log) số lượng chiều qua từng lớp mạng đánh dấu những điểm thắt cổ chai tái tổ chức thông tin dữ liệu của mô hình.

---

## 4. Lý Thuyết Thông Tin: Mutual Information và Các Động Thái Phân Cụm Ngữ Pháp

### 4.1. Mutual Information vs. Covariance
Ở các tầng ẩn, ta thường so sánh mức độ chia sẻ thông tin giữa 2 không gian lưu trữ nơ-ron thay vì chỉ dựa vào phân tích tương quan cấu trúc tuyến tính:
- Tiêu chuẩn **hiệp phương sai (Covariance)**:
  $$ \text{Cov}(X,Y) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) $$
  *(Chỉ phát hiện mối tương tác tuyến tính).*
- Tiêu chuẩn **Mutual Information ($I$)** có khả năng chẩn đoán quy luật đa hệ quả, phi tuyến. Định luợng lượng chung đụng Entropy (độ bất định thông tin $H$):
  $$ I(X;Y) = \sum_{x,y} P(x,y) \log \left( \frac{P(x,y)}{P(x)P(y)} \right) = H(X) + H(Y) - H(X,Y) $$

### 4.2. Phân Cụm Dấu Câu (Internal vs. Terminal Punctuation)
Khi tính toán giá trị thông tin tương hỗ đôi (Pairwise Mutual Information), các tokens là dấu phẩy (internal) được mô hình ứng xử phi tuyến khác xa so với dấu chấm/chấm than (terminal). Cụm dữ liệu PCA và phân bố giá trị Covariance phân hóa thành từng block riêng rẽ, thể hiện sự am hiểu nội tại LLM về cấu trúc thời luồng đọc mà không cần khai báo nhãn trực tiếp.

---

## 5. Ống Kính Logit (Logit Lens): Giải Mã Sớm Suy Diễn Từ Cốt Lõi
Khái niệm **Logit Lens** hoạt động bằng phương thức "ép chín" đầu ra dự đoán. Thay vì chờ ma trận xuất ở lớp cuối $L$, ta lấy ngay trạng thái trung gian ẩn của token ở lớp $l$ (với $l \ll L$) và nhân với lớp truy hồi bộ từ vựng (Unembedding matrix $W_U$):
$$ Z_l = h_l \cdot W_U^T $$
$$ \text{Token Predicted}_l = \text{argmax}(\log (\text{Softmax}(Z_l))) $$
Trong các thử nghiệm trên GPT-2 hay BERT, ở các lớp ngoài $(l \in [1, 3])$, Logit Lens bộc lộ những dự đoán "ngây thơ" hoặc lập lại danh từ. Khi đi sâu $(l \in [6, 12])$, mô hình tinh chỉnh sự chọn lọc định hình nên một dự đoán bám sát dòng văn cảnh chính xác nhất. Ánh nhìn này giống như chụp ảnh X-quang, truy vết sự hình thành sự thông tuệ xuyên thấu qua mạng dọc.

---

## 6. Kết luận
Bộ khung phân tích đi từ Cosine Similarity, SVD (PCA), Entropy Information cho tới Logit Lens chính là những lớp áo bảo bộ thiết yếu nhằm hiểu cấu trúc vật lý mạng LLM. Chúng cung cấp lời giải khoa học cho việc tại sao, khi nào, và bằng cách nào - các Attention Layer tương tác nội dung, mã hóa thông điệp, tái phân bổ chi tiết phi tuyến, phục vụ cho quá trình tổng hợp kết quả cuối cùng hoàn mỹ.

---

## Tài Liệu Tham Khảo (Citations)
Dựa theo hệ thống mã nguồn và học liệu gốc định kèm:
1. **aero_LLM_01 - 03:** Token-related similarities within and across Q, K, V matrices.
2. **aero_LLM_04 - 05:** Grouping and RSA in Q and K matrices; Laminar profile. Tính toán đối xứng không gian nội.
3. **aero_LLM_06 - 07:** Effective dimensionality analysis with PCA; Dimensionalities in Pythia 2.3B.
4. **aero_LLM_08 - 12:** Khái niệm Mutual information theory & code, pairwise MI, vs covariance.
5. **aero_LLM_13 - 14:** Clusters in internal vs. terminal punctuation.
6. **aero_LLM_15 - 17:** Phương pháp The Logit Lens và sự thích ứng Logit Lens ở mô hình BERT.
