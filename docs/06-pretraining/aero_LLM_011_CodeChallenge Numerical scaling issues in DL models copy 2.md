
# 📘 Các Vấn Đề Tỷ Lệ Số Học Trong Mô Hình Học Sâu: Phân Tích Vai Trò Của Scaling và Normalization Trong Cơ Chế Attention

---

## **Abstract**

Trong các mô hình học sâu hiện đại, đặc biệt là các mô hình dựa trên Transformer, việc kiểm soát độ lớn của giá trị số học đóng vai trò quan trọng trong đảm bảo tính ổn định và hiệu quả huấn luyện. Bài viết này phân tích các vấn đề liên quan đến việc nhân ma trận, sự khuếch đại phương sai, và ảnh hưởng của chúng đến hàm Softmax trong cơ chế attention. Dựa trên tài liệu *CodeChallenge: Numerical Scaling Issues in DL Models*, nghiên cứu làm rõ lý do cần chuẩn hóa tích QKᵀ bằng căn bậc hai của chiều không gian, đồng thời khảo sát phân phối tham số Layer Normalization trong GPT-2. Kết quả cho thấy scaling và normalization là các thành phần thiết yếu nhằm duy trì “vùng Goldilocks” cho logits trong quá trình học. 

---

## **1. Introduction**

Các mô hình ngôn ngữ lớn (Large Language Models – LLMs) dựa trên kiến trúc Transformer sử dụng hàng triệu phép nhân ma trận trong mỗi bước suy luận. Mặc dù các phép toán này giúp mô hình học được biểu diễn phức tạp, chúng cũng gây ra hiện tượng khuếch đại giá trị số học.

Theo tài liệu, Softmax là một phép biến đổi mạnh nhưng rất nhạy cảm với độ lớn của đầu vào. Khi logits có giá trị quá lớn, phân phối xác suất trở nên cực đoan, làm suy giảm khả năng học của mô hình. Do đó, việc nghiên cứu các vấn đề scaling là cần thiết để hiểu rõ cơ chế hoạt động của attention. 

Bài viết tập trung phân tích:

* Ảnh hưởng của nhân ma trận đến phương sai,
* Lý do cần scaling trong attention,
* Tác động đến Softmax,
* Vai trò của Layer Normalization.

---

## **2. Theoretical Background**

### **2.1. Dot Product trong Attention**

Trong cơ chế self-attention, điểm tương đồng giữa Query và Key được tính bằng:

[
A = QK^T
]

Mỗi phần tử của (A) là tích vô hướng của hai vector có chiều (d).

Nếu các phần tử của (Q) và (K) có phân phối chuẩn với phương sai bằng 1, thì phương sai của tích vô hướng xấp xỉ:

[
Var(QK^T) \approx d
]

Do đó, độ lệch chuẩn xấp xỉ:

[
\sigma \approx \sqrt{d}
]



---

### **2.2. Softmax và Độ Nhạy Số Học**

Hàm Softmax được định nghĩa:

[
Softmax(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
]

Khi (z_i) lớn, hàm mũ làm cho một số phần tử chiếm ưu thế tuyệt đối, dẫn đến:

* Hiện tượng bão hòa,
* Gradient gần bằng 0,
* Giảm khả năng học.

Theo tài liệu, đây là nguyên nhân chính khiến logits cần được kiểm soát về mặt số học. 

---

### **2.3. Scaling trong Attention**

Để giảm phương sai của (QK^T), Transformer áp dụng phép chia:

[
A_{scaled} = \frac{QK^T}{\sqrt{d}}
]

Phép scaling này đưa độ lệch chuẩn của ma trận attention về xấp xỉ 1, giúp Softmax hoạt động trong vùng ổn định. 

---

## **3. Methodology**

### **3.1. Thí Nghiệm 1: Ma Trận Ngẫu Nhiên**

Hai ma trận (Q, K \in \mathbb{R}^{50 \times 50}) được sinh từ phân phối Gaussian chuẩn.

Các đại lượng được tính:

* (\sigma(Q)),
* (\sigma(K)),
* (\sigma(QK^T)),
* (\sqrt{50}).

Kết quả cho thấy:

[
\sigma(QK^T) \approx \sqrt{50} \approx 7
]



---

### **3.2. Thí Nghiệm 2: Thay Đổi Chiều Không Gian**

Ma trận có kích thước (50 \times n), với (n) từ 2 đến 100.

Mỗi lần lặp, tính:

* Độ lệch chuẩn của (QK^T),
* Giá trị (\sqrt{n}).

Hai đại lượng này được so sánh bằng biểu đồ.

Kết quả cho thấy sự trùng khớp gần như hoàn hảo giữa lý thuyết và thực nghiệm. 

---

### **3.3. Thí Nghiệm 3: Softmax Trước và Sau Scaling**

Thí nghiệm này so sánh:

1. Softmax của (QK^T),
2. Softmax của (\frac{QK^T}{\sqrt{d}}),
3. Negative log-softmax tương ứng.

Các giá trị được trực quan hóa bằng scatter plot.

Mục tiêu là đánh giá ảnh hưởng của scaling đến phân phối xác suất. 

---

### **3.4. Thí Nghiệm 4: Phân Tích Layer Norm Trong GPT-2**

Tất cả tham số Layer Normalization của GPT-2 được trích xuất:

* Weight (γ – stretching),
* Bias (β – shifting).

Các giá trị này được biểu diễn bằng histogram với trục y ở dạng log-scale. 

---

## **4. Experimental Results**

### **4.1. Khuếch Đại Phương Sai Khi Nhân Ma Trận**

Kết quả cho thấy:

* (\sigma(Q) \approx 1),
* (\sigma(K) \approx 1),
* (\sigma(QK^T) \approx \sqrt{d}).

Điều này chứng minh rằng nhân ma trận làm tăng phương sai theo chiều không gian. 

---

### **4.2. Ảnh Hưởng Đến Softmax**

Trước scaling:

* Chỉ một vài token có xác suất lớn,
* Phần lớn xác suất ≈ 0.

Sau scaling:

* Phân phối trải đều hơn,
* Nhiều token có cơ hội được chọn.

Hiện tượng này giúp mô hình học đa dạng hơn ở giai đoạn đầu. 

---

### **4.3. Phân Phối Tham Số Layer Norm**

Phân tích GPT-2 cho thấy:

* Tham số γ chủ yếu nằm trong khoảng 0.2–0.4,
* Tham số β tập trung quanh 0.

Điều này cho thấy Layer Norm chủ yếu có tác dụng thu nhỏ (shrink) activation. 

---

## **5. Discussion**

### **5.1. Vùng “Goldilocks” Của Logits**

Theo tài liệu, logits cần nằm trong một vùng trung gian:

* Không quá lớn → tránh bão hòa,
* Không quá nhỏ → tránh mất phân biệt.

Scaling và normalization giúp duy trì vùng này. 

---

### **5.2. Vai Trò Của Normalization**

Layer Normalization giúp:

* Ổn định gradient,
* Giảm drift của activation,
* Cân bằng giữa các tầng.

Nó là thành phần không thể thiếu trong Transformer.

---

### **5.3. Liên Hệ Với Temperature Sampling**

Scaling trong attention có vai trò tương tự tham số temperature (T):

[
P_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
]

Cả hai đều điều chỉnh độ “sắc nét” của phân phối. 

---

## **6. Limitations**

Nghiên cứu còn tồn tại một số hạn chế:

* Chủ yếu dựa trên dữ liệu ngẫu nhiên,
* Chưa đánh giá ảnh hưởng đến downstream tasks,
* Chỉ khảo sát GPT-2,
* Không so sánh với các kiến trúc khác.

Do đó, kết quả mang tính minh họa nhiều hơn tổng quát.

---

## **7. Conclusion**

Bài viết đã phân tích các vấn đề scaling số học trong mô hình học sâu và cơ chế attention. Các kết luận chính gồm:

1. Nhân ma trận làm tăng phương sai theo (\sqrt{d}).
2. Scaling là cần thiết để ổn định Softmax.
3. Không scaling dẫn đến phân phối xác suất cực đoan.
4. Layer Norm giúp kiểm soát biên độ activation.
5. Các cơ chế này phối hợp để đảm bảo tính ổn định số học.

Nghiên cứu khẳng định rằng kiểm soát tỷ lệ số học là nền tảng cho việc huấn luyện thành công các mô hình Transformer quy mô lớn.

---

## **References**

1. CodeChallenge: Numerical Scaling Issues in DL Models. Lecture Transcript.

2. Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
3. Ba et al. (2016). Layer Normalization. *arXiv*.
4. Goodfellow et al. (2016). *Deep Learning*. MIT Press.

---