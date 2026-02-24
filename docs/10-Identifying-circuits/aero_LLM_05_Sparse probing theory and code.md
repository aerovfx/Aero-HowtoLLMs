# Lý Thuyết Và Ứng Dụng Của Kỹ Thuật Dò Thưa (Sparse Probing)

## Tóm tắt (Abstract)
Kỹ thuật "Dò thưa" (Sparse probing) là một thuật toán ứng dụng hồi quy logistic tích hợp cơ chế ép chuẩn L1 (L1 Regularization / Lasso regression). Mục đích của phương pháp này là ép phần lớn các hệ số hồi quy về 0 tuyệt đối, chỉ vinh danh một số lượng cực nhỏ các biến số (Tế bào nơ-ron). Báo cáo này trình bày nền tảng toán học của việc tối ưu hóa hàm mất mát bằng thuật toán Stochastic Ave18-RAGe Gradient Descent (SAGA), cùng thực nghiệm áp dụng trên 3072 nơ-ron MLP của mô hình GPT-2 Small. Mục tiếu trích xuất và cô lập một Tổ hợp vi não bộ (Ensemble circuit) siêu nhỏ có khả năng báo hiệu Mạo từ xác định ("The") và Mạo từ không xác định ("An").

---

## 1. Mở Đầu (Introduction)
Trong bối cảnh phân loại truyền thống, ta thường dùng Hồi quy Logistic để xem xét độc lập từng tính năng: liệu Nơ-ron X có biểu hiện mức độ phản ứng mạnh mẽ hơn cho thẻ loại A so với hạng mục B hay không. 
Nhưng với Sparse Probing, ta lật ngược lăng kính: Đầu vào là một ma trận khổng lồ lên tới hàng ngàn Nơ-ron ($K = 3000+$). Đầu ra là câu hỏi: Đâu là **Tổ hợp đa Tế bào** (Cluster/Circuit/Ensemble) tối giản nhất mà sự phối hợp toán học của chúng đủ để đưa ra dự báo chính xác tuyệt đối? 
Thủ thuật này cho phép đi sâu tìm kiếm những cấu trúc hàm số tinh giản giấu kín trong biển tham số khổng lồ, một khái niệm sống còn của Cơ học Giải diễn.

---

## 2. Tiết Thiết Lập Toán Học (Methodology)

### 2.1. Cấu Trúc Hàm Mất Mát Kéo Giảm Chiều L1
Giả định ta có tập hệ số hồi quy (Regression Coefficients) $B = \{\beta_1, \beta_2, ..., \beta_K\}$ ánh xạ với tập mức kích hoạt (Activations) $A$ của một tập mẫu.
Hàm mất mát gốc cho bài toán Logistic Regression là **Binary Cross-Entropy (BCE)**:
$$ Loss_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] $$
Sức mạnh của Dò Thưa nằm ở Hàm Phạt L1 (L1 Penalty) có nhiệm vụ trừng trị tính đa biến:
$$ Loss_{Reg} = \lambda \sum_{k=1}^{K} |\beta_k| $$
Tổng hợp quá trình tối ưu hàm mục tiêu: $\text{Minimize} \left( Loss_{BCE} + Loss_{Reg} \right)$. 
Tham số Siêu định hình (Hyperparameter) $\lambda$ quyết định cường độ của độ Thưa (Sparsity). $\lambda$ càng lớn, áp lực dập $\beta_k \to 0$ càng gắt, tỷ lệ Mật độ các nơ-ron còn sống sót (Density constraint) càng nhỏ. Thư viện `scikit-learn` sử dụng nghịch đảo cường độ cực biên $C = \frac{1}{\lambda}$. 

### 2.2. Huấn Luyện Cập Nhật Đạo Hàm SAGA
Thuật toán leo dốc gradient truyền thống gặp trở ngại với hàm L1 do đặc điểm đạo hàm không gián đoạn (Non-differentiable) tại tọa độ điểm 0. Do vậy, nền tảng tối ưu hóa Solver SAGA (Stochastic Ave18-RAGe Gradient) được lựa chọn. Nhằm đảm bảo quá trình dập đỉnh hội tụ hoàn toàn (Convergence), ngưỡng tối đa chu kỳ học (Max iterations/epochs) được đẩy lên con số hàng vạn. 

---

## 3. Khảo Sát & Giải Phẫu Mô Hình (Analysis)

### 3.1. Thiết Lập Tiền Xử Lý Dữ Liệu Ngôn Ngữ Học
Trên bộ sinh khối FineWeb, ta trích lục $100$ chuỗi văn bản xoay quanh mạo từ xác định "the" và $100$ nhãn cho "an". Yêu cầu kỹ năng xử lý BPE Tokenizer khắt khe: phải phân biệt rõ chữ "the" nằm độc lập có dấu cách đi kèm (Prefix spaces logic), tách biệt khỏi chùm phụ âm khởi đầu (Prefix substring) của những từ dài như "Theology".
Tiến hành chích xuất ma trận giá trị Activation tại đuôi Module $MLP$ hàm kích hoạt hàm `GELU`. 

### 3.2. Hiện Tượng Sập Mật Độ Nơ-ron (Extreme Sparsity Density)
Bộ Dataset 200 điểm mẫu được phân tách theo tỷ lệ Test/Train (140/60).
Sau khi huấn luyện mô hình Logistic kích hoạt mức phạt hằng số $C = 10$, mô hình sinh ra Dự báo (Accuracy / F1 Score) tuyệt đối $100\%$. 
Viễn cảnh siêu phân giải hiện ra từ hệ số $B$:
- Hệ số **Sparsity = 99.6%** (2987 trên 3000 Nơ-ron bị vô hiệu hóa triệt để có $\beta = 0$).
- Hệ số **Density = 0.4%** (Chỉ duy trì $13$ tế bào Nơ-ron sống sót tham chiến).
Biểu đồ phổ (Scatter scatter plot) chứng minh: Thay vì một Mạch dài vô tận, tập hợp chỉ vỏn vẹn $13$ chiếc công tắc Toán học siêu nhỏ này đã lĩnh xướng trọn vẹn toàn bộ gánh nặng Logic để bộ máy LLM phân định chính xác khái niệm "Mạo từ xác định" (Definite) và "Mạo từ phi xác định" (Indefinite).  

---

## 4. Kết Luận
Bằng việc sử dụng L1 Regularization, Logistic Regression đã được biến đổi hình thái từ một công cụ Phân loại đơn giản thành Tháp dò mìn (Sparse Probe). Biện pháp này mang đến góc nhìn định biên (Framing constraint): Bất chấp khối lượng tham số nở phình ở các Model quy mô lớn, luận lý bên trong các cấu trúc ẩn (Latent constructs) hoàn toàn có thể được cô đọng quy về một bó tia Mạch (Circuit Ensembles) hữu hạn, vô cùng tinh giản gọn nhẹ. Giai đoạn tiếp theo sẽ đòi hỏi giải quyết hiện tượng Thống kê đàn áp (Statistical suppression) nảy sinh từ các phép thử mẫu quy mô nhỏ của hàm phạt L1.

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm đo lường Hồi quy Logistic L1 và thủ thuật giải quyết tiền xử lý BPE Whitespace Tokenization từ `aero_LLM_05_Sparse probing theory and code.md`. Triển khai cụ thể qua thuật toán SAGA của `sklearn.linear_model.LogisticRegression(penalty='l1')`.
