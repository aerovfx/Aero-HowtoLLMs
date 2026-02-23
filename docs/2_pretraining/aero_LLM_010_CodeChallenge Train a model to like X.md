Dưới đây là **bài viết khoa học** được xây dựng dựa trên tài liệu **“CodeChallenge: Train a Model to Like X”**, có bổ sung phân tích học thuật và trích dẫn nguồn, trình bày theo định dạng **Markdown**.

---

# 📘 Huấn Luyện Mô Hình Ngôn Ngữ Với Thiên Lệch Có Chủ Đích Bằng KL-Divergence: Một Nghiên Cứu Thực Nghiệm

---

## **Abstract**

Trong quá trình huấn luyện mô hình ngôn ngữ lớn (Large Language Models – LLMs), hàm mất mát đóng vai trò quyết định trong việc định hướng hành vi học của mô hình. Bài viết này phân tích phương pháp huấn luyện một biến thể của GPT-2 nhằm tạo ra thiên lệch có chủ đích: ưu tiên sinh ra các token chứa ký tự “X”. Nghiên cứu sử dụng hàm mất mát Kullback–Leibler Divergence (KL-divergence) để ép phân phối xác suất đầu ra của mô hình tiệm cận một phân phối mục tiêu được thiết kế trước. Kết quả cho thấy mô hình có thể bị “lập trình thiên lệch” với hiệu quả rất cao chỉ sau số lượng nhỏ epoch huấn luyện, qua đó đặt ra những vấn đề nghiêm trọng liên quan đến đạo đức và an toàn AI. 

---

## **1. Introduction**

Các mô hình ngôn ngữ hiện đại như GPT-2 và GPT-3 học cách sinh văn bản thông qua việc tối ưu hóa hàm mất mát dựa trên dữ liệu huấn luyện. Trong phần lớn nghiên cứu, mục tiêu là cải thiện độ chính xác và khả năng tổng quát hóa.

Tuy nhiên, tài liệu *Train a Model to Like X* đề xuất một thí nghiệm mang tính minh họa: huấn luyện mô hình để ưu tiên các token chứa chữ cái “X”, bất kể ngữ nghĩa. Thí nghiệm này vừa mang tính giáo dục, vừa cho thấy mức độ dễ dàng trong việc “bẻ lái” hành vi của mô hình ngôn ngữ. 

Mục tiêu của bài viết là:

* Phân tích phương pháp tạo thiên lệch bằng KL-divergence,
* Đánh giá kết quả thực nghiệm,
* Thảo luận các hệ quả về AI safety.

---

## **2. Theoretical Background**

### **2.1. Language Modeling và Phân Phối Xác Suất**

Mô hình GPT-2 học phân phối:

[
P(x_t \mid x_1, \dots, x_{t-1})
]

Trong đó, mỗi token được sinh dựa trên ngữ cảnh trước đó.

Đầu ra của mô hình là một vector xác suất trên toàn bộ từ vựng:

[
P = (p_1, p_2, \dots, p_V)
]

với (V) là kích thước vocab.

---

### **2.2. Kullback–Leibler Divergence**

KL-divergence đo độ khác biệt giữa hai phân phối xác suất (P) và (Q):

[
D_{KL}(Q||P) = \sum_i Q(i)\log\frac{Q(i)}{P(i)}
]

Trong đó:

* (Q): phân phối mục tiêu,
* (P): phân phối dự đoán của mô hình.

Tối thiểu hóa KL-divergence tương đương với việc ép (P) tiệm cận (Q). 

---

### **2.3. Bias Trong Mô Hình Ngôn Ngữ**

Thiên lệch (bias) trong LLMs có thể xuất hiện do:

* Dữ liệu huấn luyện,
* Hàm mất mát,
* Mục tiêu tối ưu hóa.

Trong nghiên cứu này, bias được tạo ra một cách có chủ đích thông qua phân phối mục tiêu nhân tạo.

---

## **3. Methodology**

### **3.1. Thiết Lập Mô Hình**

Mô hình sử dụng là Model 5, một biến thể rút gọn của GPT-2:

* Kiến trúc Transformer,
* Tokenizer GPT-2,
* Chạy trên GPU.

Việc huấn luyện trên CPU bị hạn chế nghiêm trọng về thời gian. 

---

### **3.2. Sinh Chuỗi Ban Đầu**

Dữ liệu đầu vào ban đầu gồm:

* Token ngẫu nhiên,
* Độ dài chuỗi: 256,
* Sinh thêm 200 token mới.

Trước huấn luyện, không có token nào chứa “X” trong 200 token sinh ra. 

---

### **3.3. Tạo Mask Cho Token Mục Tiêu**

Một vector mask được xây dựng:

[
M_i =
\begin{cases}
1, & \text{nếu token } i \text{ chứa X}\
0, & \text{ngược lại}
\end{cases}
]

Sau đó được chuẩn hóa thành phân phối xác suất:

[
Q_i = \frac{M_i}{\sum_j M_j}
]

Theo thống kê, chỉ khoảng 2% token chứa ký tự “X”. 

---

### **3.4. Xây Dựng Custom Loss Function**

Hàm loss được xây dựng bằng `torch.nn.Module` và sử dụng `F.kl_div`:

[
\mathcal{L} = D_{KL}(Q||P)
]

Lưu ý:

* Đầu vào thứ nhất: log-probability,
* Đầu vào thứ hai: probability.

Sai khác này là điểm kỹ thuật quan trọng trong triển khai. 

---

### **3.5. Quy Trình Huấn Luyện**

Quy trình huấn luyện gồm:

* 200 epoch,
* Dữ liệu đầu vào: token ngẫu nhiên,
* Chỉ dùng token cuối để tính loss,
* Optimizer: SGD/Adam.

Không sử dụng dữ liệu văn bản thực tế. 

---

## **4. Experimental Results**

### **4.1. Diễn Biến Loss**

Loss ban đầu xấp xỉ:

[
\log(V) \approx 11
]

Sau huấn luyện, loss giảm mạnh về gần 0, cho thấy mô hình đã học gần như hoàn hảo phân phối mục tiêu. 

---

### **4.2. Kết Quả Sinh Văn Bản**

Sau huấn luyện:

* 188/200 token chứa “X”,
* Tỷ lệ ≈ 95%.

So với 0% ban đầu, mức tăng này là rất đáng kể. 

---

### **4.3. Đặc Tính Văn Bản Sinh Ra**

Văn bản sau huấn luyện:

* Xuất hiện dày đặc ký tự “X”,
* Mất ngữ nghĩa tự nhiên,
* Bị chi phối mạnh bởi mục tiêu tối ưu.

Điều này cho thấy loss function có thể “bẻ cong” hành vi mô hình.

---

## **5. Discussion**

### **5.1. Hiệu Quả Của KL-Divergence**

KL-divergence cho phép:

* Điều khiển toàn bộ phân phối output,
* Không chỉ tác động lên một token đơn lẻ,
* Tạo bias mạnh và nhanh.

Đây là công cụ rất mạnh trong huấn luyện có điều kiện.

---

### **5.2. Khả Năng Thao Túng Mô Hình**

Thí nghiệm cho thấy:

* Việc tạo thiên lệch là rất dễ,
* Không cần dữ liệu thật,
* Chỉ cần thiết kế loss phù hợp.

Điều này đặt ra nguy cơ thao túng hành vi LLMs trong thực tế.

---

### **5.3. Liên Hệ Đến AI Safety**

Theo tài liệu, cùng một kỹ thuật có thể được dùng để:

* Thúc đẩy quan điểm chính trị,
* Gợi ý hành vi tiêu dùng,
* Tạo thiên lệch xã hội,
* Ảnh hưởng quyết định cá nhân.

Các bias tinh vi khó phát hiện hơn nhiều so với ví dụ “X”. 

---

## **6. Limitations**

Nghiên cứu còn một số hạn chế:

* Không dùng dữ liệu thực,
* Không đánh giá long-term generalization,
* Chỉ thử nghiệm một dạng bias,
* Không đo ảnh hưởng tới downstream tasks.

Do đó, kết quả mang tính minh họa nhiều hơn thực nghiệm ứng dụng.

---

## **7. Conclusion**

Bài viết đã phân tích phương pháp huấn luyện mô hình GPT-2 với thiên lệch có chủ đích thông qua KL-divergence. Các kết luận chính gồm:

1. KL-divergence cho phép điều khiển phân phối đầu ra hiệu quả.
2. Mô hình có thể học bias rất nhanh.
3. Custom loss có sức ảnh hưởng mạnh đến hành vi LLM.
4. Việc tạo thiên lệch kỹ thuật là tương đối dễ dàng.
5. Vấn đề AI safety cần được xem xét nghiêm túc.

Nghiên cứu nhấn mạnh rằng thiết kế hàm mất mát không chỉ là vấn đề kỹ thuật, mà còn là vấn đề đạo đức và xã hội.

---

## **References**

1. CodeChallenge: Train a Model to Like X. Lecture Transcript.

2. Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
3. Sutton, R., Barto, A. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
4. Amodei et al. (2016). Concrete Problems in AI Safety. *arXiv*.

---
