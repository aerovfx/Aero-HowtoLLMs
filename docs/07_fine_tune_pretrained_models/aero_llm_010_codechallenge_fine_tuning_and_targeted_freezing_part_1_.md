
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [07 fine tune pretrained models](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Fine-tuning Có Mục Tiêu và Đóng Băng Chính Xác Trọng Số Trong Mô Hình Ngôn Ngữ Lớn

## Tóm tắt

Bài viết này nghiên cứu phương pháp fine-tuning có mục tiêu (targeted fine-tuning) kết hợp với đóng băng chính xác (precision freezing) một phần trọng số trong mô hình ngôn ngữ lớn. Thực nghiệm được thực hiện trên tập dữ liệu từ tiểu thuyết *Moby-Dick* của *Herman Melville*. Hai mô hình giống hệt nhau được huấn luyện song song: một mô hình được đóng băng chọn lọc các lớp attention, và một mô hình được huấn luyện toàn phần. Kết quả cho thấy chiến lược đóng băng có mục tiêu giúp giảm chi phí tính toán, hạn chế overfitting và vẫn duy trì hiệu quả học tập.

---

## 1. Giới thiệu

Fine-tuning là kỹ thuật quan trọng giúp thích nghi mô hình tiền huấn luyện với dữ liệu chuyên biệt. Tuy nhiên, việc cập nhật toàn bộ tham số trong các mô hình lớn thường:

* Tốn nhiều tài nguyên,
* Dễ gây quá khớp,
* Khó kiểm soát quá trình học.

Tài liệu thực nghiệm  đề xuất phương pháp huấn luyện song song hai mô hình: một mô hình được đóng băng chọn lọc các lớp attention ở tầng thấp, và một mô hình huấn luyện đầy đủ, nhằm đánh giá tác động của precision freezing.

Mục tiêu nghiên cứu:

* Phân tích cơ chế fine-tuning có mục tiêu,
* Đánh giá ảnh hưởng của đóng băng attention,
* So sánh hiệu quả huấn luyện và chi phí tính toán.

---

## 2. Cơ sở lý thuyết

### 2.1. Mô hình ngôn ngữ tự hồi quy

Cho chuỗi token:

$$

$$

X = (x_1, x_2, \dots, x_n)

$$

$$

Xác suất sinh chuỗi:

$$

$$

P(X)=\prod_{i=1}^{n} P(x_i \mid x_1,\dots,x_{i-1})

$$

$$

Mô hình dự đoán token tiếp theo dựa trên toàn bộ ngữ cảnh trước đó.

---

### 2.2. Cơ chế Attention trong Transformer

Trong một lớp Transformer, self-attention được xác định bởi:

$$

$$

Q = XW_Q,\quad K = XW_K,\quad V = XW_V

$$

$$

$$

$$

\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

$$

$$

Trong đó:

* ($W_Q$, $W_K$, $W_V$): ma trận truy vấn, khóa và giá trị,
* $d_k$: số chiều của vector key.

Các ma trận này là trọng tâm của chiến lược fine-tuning có mục tiêu.

---

### 2.3. Hàm mất mát và cập nhật tham số

Hàm mất mát Cross-Entropy:

$$

$$

$\mathcal${L} = -\frac{1}{N}$\sum$_{i=1}^{N}$\log$ P($y_i$ \mid $x_i$)

$$

$$

Quy tắc cập nhật:

$$

$$

\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}

$$

$$

với $\eta$ là learning rate.

Nếu tham số bị đóng băng:

$$

$$

$\nabla$_\theta $\mathcal${L} = 0

$$

$$

⇒ không được cập nhật.

---

## 3. Phương pháp nghiên cứu

### 3.1. Dữ liệu

Nguồn dữ liệu là văn bản *Moby-Dick*, gồm:

$$

$$

N_{total} \approx 350,000

$$

$$

token, trong đó chỉ khoảng:

$$

$$

N_{unique} \approx 17,000

$$

$$

token là duy nhất .

---

### 3.2. Khởi tạo mô hình

Hai mô hình giống hệt nhau được tải:

* Mô hình Train: huấn luyện toàn bộ.
* Mô hình Freeze: đóng băng có mục tiêu.

Ban đầu:

$$

$$

\theta_{\text{train}}^{(0)} = \theta_{\text{freeze}}^{(0)}

$$

$$

---

### 3.3. Thống kê token phổ biến

Tần suất token:

$$

$$

f(w)=\sum_{i=1}^{N}\mathbf{1}(x_i=w)

$$

$$

Chọn tập 100 token phổ biến nhất:

$$

$$

S_{100}={w_1,\dots,w_{100}}

$$

$$

---

### 3.4. Đánh giá tỷ lệ token sinh

Cho chuỗi sinh:

$$

$$

G=(g_1,\dots,g_M)

$$

$$

Tỷ lệ token phổ biến:

$$

$$

p=\frac{1}{M}\sum_{i=1}^{M}\mathbf{1}(g_i\in S_{100})

$$

$$

Chỉ số này phản ánh mức độ mô hình học được phong cách văn bản.

---

### 3.5. Chiến lược đóng băng có mục tiêu

Theo tài liệu , chỉ huấn luyện:

* Trọng số ($W_Q$, $W_K$, $W_V$),
* Trong các block Transformer từ tầng 6 trở lên.

Mô tả toán học:

$$

$$

\theta_i = \begin{cases} \text{trainable}, & i \in \mathcal{A}_{6+} \\ \text{frozen}, & \text{ngược lại} \end{cases}

$$

$$

với \mathcal{A}_{6+} là tập attention layer từ block 6 trở lên.

$$
--- ## 4. Theo dõi quá trình huấn luyện ### 4.1. Đo thời gian huấn luyện Thời gian mỗi vòng lặp:
$$

$$
t_k = t_k^{end}-t_k^{start}
$$

$$
Tổng thời gian:
$$

$$
T=\sum_{k=1}^{K} t_k
$$

$$
So sánh T_{\text{freeze}} và T_{\text{train}}. --- ### 4.2. Theo dõi biến đổi trọng số Cho ma trận tại bước t:
$$

W_t

$$
Hiệu giữa hai bước:
$$

$$
\Delta W_t = W_t - W_{t-1}
$$

$$
Chuẩn Frobenius:
$$

|\Delta $W_t$|_F = \sqrt{$\sum$_{i,j}(\Delta W_{ij})^2}

$$
Chuẩn lớn ⇒ cập nhật mạnh. Chuẩn nhỏ ⇒ cập nhật yếu. --- ### 4.3. Theo dõi hàm mất mát Loss trung bình:
$$

$$
\bar{\mathcal{L}} = \frac{1}{K}\sum_{k=1}^{K}\mathcal{L}_k
$$

$$
Dùng để so sánh tốc độ hội tụ của hai mô hình. --- ## 5. Kết quả thực nghiệm ### 5.1. Trước fine-tuning Tỷ lệ token phổ biến:
$$

$$
p_{\text{train}}\approx 47%,\quad p_{\text{freeze}}\approx 44%
$$

$$
Hai mô hình gần như tương đương . --- ### 5.2. Sau fine-tuning Quan sát cho thấy: * Mô hình Train: học mạnh nhưng dễ overfit. * Mô hình Freeze: học ổn định hơn.
$$

p_{\text{freeze}}^{post} > p_{\text{freeze}}^{pre}

$$
và có độ biến động nhỏ hơn. --- ### 5.3. Chi phí tính toán Số tham số huấn luyện:
$$

P_{\text{freeze}} \ll P_{\text{train}}

$$
Do đó:
$$

T_{\text{freeze}} \lt  T_{\text{train}}

$$
--- ## 6. Thảo luận ### 6.1. Ưu điểm 1. Giảm thời gian huấn luyện. 2. Tiết kiệm bộ nhớ. 3. Hạn chế overfitting. 4. Bảo toàn tri thức nền. --- ### 6.2. Hạn chế * Khả năng thích nghi bị giới hạn. * Phụ thuộc cấu hình đóng băng. * Cần nhiều thử nghiệm để tối ưu. --- ### 6.3. Chiến lược mở rộng #### Đóng băng từng phần theo thời gian
$$

$$
\theta_i(t)= \begin{cases} \text{frozen}, & t\lt t_0\ \text{trainable}, & t\ge t_0 \end{cases}
$$

$$
#### Kết hợp LoRA/Adapter Giữ nguyên \theta, thêm tham số phụ \phi:
$$

$$
y = f(x;\theta)+g(x;\phi)
$$

