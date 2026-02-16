

# Mô Phỏng Hội Thoại Giữa Hai Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp *Alice* và *Edgar*

## Tóm tắt

Bài viết này trình bày phương pháp xây dựng và đánh giá hệ thống hội thoại nhân tạo giữa hai mô hình ngôn ngữ được fine-tuning theo phong cách văn học từ *Alice's Adventures in Wonderland* và các tác phẩm của *Edgar Allan Poe*. Thông qua cơ chế luân phiên sinh token, hai mô hình được cho “trò chuyện” với nhau trong cùng một ngữ cảnh. Nghiên cứu phân tích quy trình kỹ thuật, mô hình toán học nền tảng và đánh giá khả năng hình thành hội thoại của các mô hình sinh ngôn ngữ.

---

## 1. Giới thiệu

Trong những năm gần đây, các mô hình ngôn ngữ lớn (Large Language Models – LLMs) đã đạt được nhiều tiến bộ trong sinh văn bản và hội thoại. Tuy nhiên, phần lớn các chatbot hiện đại được huấn luyện đặc biệt cho nhiệm vụ đối thoại.

Tài liệu thực nghiệm  đề xuất một cách tiếp cận đơn giản: sử dụng hai mô hình đã fine-tuning theo hai phong cách khác nhau và cho chúng lần lượt sinh phản hồi cho nhau, từ đó tạo thành một chuỗi hội thoại tự động.

Mục tiêu của nghiên cứu gồm:

* Mô phỏng hội thoại giữa hai mô hình độc lập.
* Phân tích cơ chế sinh token tuần tự.
* Đánh giá khả năng duy trì ngữ cảnh và phong cách.

---

## 2. Cơ sở lý thuyết

### 2.1. Mô hình ngôn ngữ tự hồi quy

Cho chuỗi token:

[
X = (x_1, x_2, \dots, x_n)
]

Xác suất sinh chuỗi được mô hình hóa như sau:

[
P(X) = \prod_{i=1}^{n} P(x_i \mid x_1, x_2, \dots, x_{i-1})
]

Trong đó:

* (x_i) là token thứ (i),
* mỗi token phụ thuộc vào toàn bộ ngữ cảnh trước đó.

---

### 2.2. Biểu diễn ngữ cảnh (Context Window)

Cửa sổ ngữ cảnh tại bước (t):

[
C_t = (x_1, x_2, \dots, x_t)
]

Mô hình sinh token tiếp theo dựa trên:

[
x_{t+1} \sim P(x \mid C_t)
]

Khi hội thoại kéo dài, độ dài ngữ cảnh tăng dần:

[
|C_{t+1}| = |C_t| + 1
]

---

### 2.3. Fine-tuning mô hình

Quá trình fine-tuning cập nhật tham số (\theta) thông qua hàm mất mát Cross-Entropy:

[
\mathcal{L}(\theta)
===================

-\frac{1}{N}
\sum_{i=1}^{N}
\log P(y_i \mid x_i; \theta)
]

Mục tiêu:

[
\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)
]

---

## 3. Phương pháp nghiên cứu

### 3.1. Mô hình sử dụng

Hai mô hình được huấn luyện riêng biệt:

* Mô hình A: phong cách *Alice*.
* Mô hình E: phong cách *Edgar*.

Sau fine-tuning, mỗi mô hình có tập tham số:

[
\theta_A, \quad \theta_E
]

---

### 3.2. Khởi tạo hội thoại

Hội thoại bắt đầu bằng prompt ban đầu:

[
S_0 = \text{``Hello, my name is Alice.''}
]

Sau tokenization:

[
T_0 = (t_1, t_2, \dots, t_k)
]

Chuỗi này được đưa vào mô hình E.

---

### 3.3. Cơ chế luân phiên sinh phản hồi

Quy trình hội thoại gồm các bước:

#### Bước 1: Edgar sinh phản hồi

[
G_E^{(1)} \sim P(\cdot \mid T_0; \theta_E)
]

Sinh ra (m) token:

[
G_E^{(1)} = (g_1, \dots, g_m)
]

#### Bước 2: Cập nhật ngữ cảnh

[
C_1 = T_0 \oplus G_E^{(1)}
]

với (\oplus) là phép nối chuỗi.

#### Bước 3: Alice sinh phản hồi

[
G_A^{(1)} \sim P(\cdot \mid C_1; \theta_A)
]

#### Bước 4: Lặp

Quá trình được lặp lại (K) lần:

[
C_{k+1} = C_k \oplus G_{model}^{(k)}
]

Trong đó:

[
model =
\begin{cases}
E, & k \text{ lẻ} \
A, & k \text{ chẵn}
\end{cases}
]

---

### 3.4. Lấy mẫu ngẫu nhiên (Sampling)

Token được sinh bằng phương pháp sampling:

[
x_{t+1} \sim \text{Categorical}(p_1, \dots, p_V)
]

với:

[
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
]

Trong đó:

* (z_i): logit,
* (T): temperature.

---

## 4. Thực nghiệm

### 4.1. Cấu hình

Theo tài liệu gốc :

* Mỗi lượt sinh: 50 token.
* Số vòng lặp: 5.
* Tổng số lượt sinh: 10.

Tổng số token sinh:

[
M \approx 500
]

---

### 4.2. Quản lý ngữ cảnh

Chỉ in ra token mới sinh:

[
G^{(k)} = C_k[|C_{k-1}|+1 : |C_k|]
]

Điều này giúp tránh in lại toàn bộ lịch sử.

---

### 4.3. Ví dụ hội thoại

Một số đặc điểm được quan sát:

* Edgar đặt câu hỏi mang tính triết lý.
* Alice phản hồi theo hướng tưởng tượng.
* Xuất hiện hiện tượng “hoàn thiện từ” giữa hai mô hình.

Ví dụ: Edgar sinh “astan-”, Alice hoàn thiện thành “astonishment”.

---

## 5. Phân tích và thảo luận

### 5.1. Tính chất của hội thoại

Khác với chatbot chuyên dụng, hai mô hình trong nghiên cứu:

* Không có token đặc biệt cho vai trò (user/assistant),
* Không được huấn luyện hội thoại,
* Chỉ thực hiện “hoàn thành chuỗi” (sequence completion).

Do đó, hội thoại thực chất là:

[
\hat{X} = \arg\max_X P(X \mid C_0)
]

chứ không phải đối thoại có mục đích.

---

### 5.2. Ưu điểm

* Dễ triển khai.
* Không cần dữ liệu hội thoại.
* Minh họa rõ cơ chế sinh tự hồi quy.
* Tạo ra kết quả sáng tạo.

---

### 5.3. Hạn chế

1. Thiếu cấu trúc vai trò.
2. Không kiểm soát chủ đề.
3. Dễ lan man.
4. Phụ thuộc mạnh vào prompt ban đầu.

---

### 5.4. Hướng cải tiến

Có thể mở rộng bằng:

* Instruction tuning.
* RLHF.
* Special tokens cho hội thoại.
* Memory compression.

Ví dụ, bổ sung token vai trò:

[ <USER>, <ASSISTANT>
]

giúp mô hình học cấu trúc đối thoại.

---

## 6. Đánh giá định lượng bổ trợ

Có thể đo mức ổn định hội thoại bằng entropy:

[
H = -\sum_{i=1}^{V} p_i \log p_i
]

Entropy cao → phản hồi đa dạng.
Entropy thấp → phản hồi lặp.

Hoặc độ dài phụ thuộc ngữ cảnh:

[
D = \frac{1}{K}\sum_{k=1}^{K} |C_k|
]

---

## 7. Kết luận

Nghiên cứu cho thấy việc cho hai mô hình ngôn ngữ fine-tuning “trò chuyện” với nhau là một phương pháp trực quan để khảo sát khả năng duy trì ngữ cảnh và phong cách.

Mặc dù chưa đạt đến mức hội thoại thực sự, phương pháp này:

* Giúp hiểu rõ cơ chế sinh token,
* Minh họa vai trò của context window,
* Là nền tảng cho nghiên cứu chatbot chuyên sâu.

Trong tương lai, việc kết hợp instruction tuning và đánh giá đa chiều sẽ giúp cải thiện chất lượng đối thoại.

---

## Tài liệu tham khảo

1. Tài liệu hướng dẫn mô phỏng hội thoại giữa Alice và Edgar 
2. Vaswani et al. (2017). *Attention Is All You Need*.
3. Jurafsky, D., & Martin, J. (2023). *Speech and Language Processing*.
4. OpenAI (2024). *Large Language Model Evaluation Guide*.

---
