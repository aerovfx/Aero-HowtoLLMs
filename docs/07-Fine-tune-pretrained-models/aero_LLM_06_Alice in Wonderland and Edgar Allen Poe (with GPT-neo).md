
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [07 Fine tune pretrained models](../index.md)

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
Dưới đây là **bài viết khoa học bằng tiếng Việt**, được xây dựng dựa trên tài liệu bạn cung cấp , có bổ sung **nguồn tham khảo**, **công thức toán học minh họa**, và trình bày theo định dạng **Markdown**.

---

# Tinh Chỉnh Mô Hình GPT-Neo Để Mô Phỏng Phong Cách Văn Học Alice in Wonderland và Edgar Allan Poe

---

## Tóm tắt

Tinh chỉnh mô hình ngôn ngữ lớn (Large Language Models – LLMs) theo phong cách văn học cụ thể là một hướng nghiên cứu quan trọng trong lĩnh vực trí tuệ nhân tạo sáng tạo. Bài báo này trình bày phương pháp fine-tuning mô hình GPT-Neo nhằm mô phỏng phong cách viết của *Alice in Wonderland* và *Edgar Allan Poe*. Nghiên cứu tập trung vào phân tích kiến trúc mô hình, quá trình tiền xử lý dữ liệu, phương pháp huấn luyện, đánh giá định lượng và định tính. Kết quả cho thấy rằng mô hình sau tinh chỉnh có khả năng tái hiện rõ nét phong cách văn học đặc trưng của từng tác giả. 

---

## Từ khóa

GPT-Neo, Fine-tuning, Language Modeling, Style Transfer, Transformer, Sinh văn bản

---

## 1. Giới thiệu

Trong những năm gần đây, các mô hình ngôn ngữ dựa trên Transformer đã đạt được nhiều thành tựu nổi bật trong lĩnh vực xử lý ngôn ngữ tự nhiên. Một trong những ứng dụng quan trọng là sinh văn bản theo phong cách cụ thể.

Mục tiêu của nghiên cứu này là huấn luyện hai mô hình GPT-Neo giống nhau về kiến trúc nhưng được tinh chỉnh trên hai tập dữ liệu khác nhau:

* Văn bản *Alice in Wonderland*
* Tuyển tập tác phẩm của Edgar Allan Poe

Qua đó, đánh giá khả năng học phong cách văn học của mô hình. 

---

## 2. Cơ sở lý thuyết

### 2.1 Mô hình ngôn ngữ tự hồi quy

GPT-Neo thuộc nhóm mô hình ngôn ngữ tự hồi quy (Autoregressive Language Model), với xác suất sinh chuỗi:

[
P(x_1,x_2,...,x_T)=\prod_{t=1}^{T}P(x_t|x_1,...,x_{t-1})
]

Trong đó:

* (x_t): token tại thời điểm (t)
* (T): độ dài chuỗi

Mô hình dự đoán token tiếp theo dựa trên toàn bộ ngữ cảnh trước đó.

---

### 2.2 Kiến trúc Transformer

Mỗi block Transformer gồm:

* Multi-head Self-Attention
* Feed-forward Network (MLP)
* Layer Normalization

Công thức Attention:

[
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

Trong đó:

* (Q,K,V): ma trận truy vấn, khóa, giá trị
* (d_k): chiều vector khóa

---

### 2.3 Hàm Softmax và Log-likelihood

Đầu ra của mô hình là vector logit (\mathbf{z}):

[
\mathbf{z}=(z_1,z_2,...,z_V)
]

Xác suất token thứ (i):

[
P(i)=\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
]

Log-likelihood:

[
\log P(i)=z_i-\log\sum_{j}e^{z_j}
]

---

## 3. Phương pháp nghiên cứu

### 3.1 Mô hình GPT-Neo

Mô hình sử dụng trong nghiên cứu là GPT-Neo 125M với:

* Số tham số: ~125 triệu
* Embedding dimension: 768
* Vocabulary size: 50,257
* Số block Transformer: 12

Mô hình có kích thước tương đương GPT-2 Small. 

---

### 3.2 Tập dữ liệu

Hai tập dữ liệu chính:

| Tập dữ liệu         | Số token |
| ------------------- | -------- |
| Alice in Wonderland | ~50,000  |
| Edgar Allan Poe     | ~200,000 |

Tập Poe có độ đa dạng cao hơn do gồm nhiều truyện và thơ khác nhau. 

---

### 3.3 Tokenization

Dữ liệu được mã hóa bằng tokenizer GPT-2:

[
x = (x_1,x_2,...,x_T), \quad x_i \in {1,...,V}
]

Trong đó (V = 50257) là kích thước từ vựng.

Tokenizer của GPT-Neo trùng với GPT-2 tokenizer. 

---

### 3.4 Hàm mất mát

Mô hình sử dụng Negative Log-Likelihood Loss:

[
\mathcal{L}=-\frac{1}{T}\sum_{t=1}^{T}\log P(x_t|x_{<t})
]

Hàm này đo độ phù hợp giữa phân phối dự đoán và dữ liệu thực tế.

---

### 3.5 Quy trình huấn luyện

Mỗi vòng huấn luyện gồm:

1. Lấy batch token ngẫu nhiên
2. Forward pass
3. Tính loss
4. Backpropagation
5. Cập nhật trọng số

Cập nhật tham số:

[
\theta_{k+1}=\theta_k-\eta\nabla_\theta\mathcal{L}
]

Trong đó:

* (\eta): learning rate
* (\theta): tham số mô hình



---

### 3.6 Tối ưu hóa

Sử dụng Adam Optimizer:

[
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t
]

[
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
]

[
\theta_t=\theta_{t-1}-\eta\frac{m_t}{\sqrt{v_t}+\epsilon}
]

Trong đó (g_t) là gradient tại bước (t).

---

## 4. Thực nghiệm

### 4.1 Thiết lập

| Tham số         | Giá trị |
| --------------- | ------- |
| Batch size      | 16      |
| Sequence length | 256     |
| Số vòng lặp     | 500     |
| Optimizer       | Adam    |
| GPU             | Có      |



---

### 4.2 Phân tích hàm mất mát

Kết quả:

* Alice: Loss → 0.19
* Poe: Loss → 1.46

Biểu đồ loss cho thấy tốc độ hội tụ của Alice nhanh hơn.

Nguyên nhân:

* Dữ liệu Alice đồng nhất hơn
* Văn phong gần tiếng Anh hiện đại hơn

---

### 4.3 Đánh giá định lượng

Perplexity được sử dụng để đánh giá:

[
PPL = e^{\mathcal{L}}
]

Perplexity thấp cho thấy mô hình dự đoán tốt hơn.

Mô hình Alice có perplexity thấp hơn mô hình Poe.

---

### 4.4 Đánh giá định tính

Với cùng prompt:

> “What did the Red Queen say to Alice?”

* Mô hình Alice sinh hội thoại, đối thoại
* Mô hình Poe sinh văn bản u ám, siêu thực

Điều này cho thấy mô hình học được phong cách riêng biệt. 

---

## 5. Thảo luận

### 5.1 Ưu điểm

* Học được phong cách tác giả
* Dễ triển khai
* Không cần huấn luyện từ đầu
* Linh hoạt với nhiều tập dữ liệu

---

### 5.2 Hạn chế

* Dễ overfitting
* Phụ thuộc chất lượng dữ liệu
* Khó đánh giá tự động
* Tốn tài nguyên tính toán

Loss thấp không đồng nghĩa với chất lượng sinh văn bản tốt.

---

## 6. Kết luận

Nghiên cứu đã chứng minh rằng mô hình GPT-Neo có thể được tinh chỉnh thành công để mô phỏng phong cách văn học khác nhau. Việc sử dụng cùng kiến trúc nhưng huấn luyện trên dữ liệu khác nhau dẫn đến sự khác biệt rõ rệt trong đầu ra.

Hướng phát triển tiếp theo:

* Kết hợp nhiều phong cách
* Prompt tuning
* RLHF
* Style regularization
* Đánh giá tự động nâng cao

---

## Tài liệu tham khảo

1. *Alice in Wonderland and Edgar Allen Poe (with GPT-Neo)*. “6 - Alice in Wonderland and Edgar Allen Poe (with GPT-neo).txt”. 
2. Vaswani et al. (2017). *Attention Is All You Need*.
3. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners*.
4. Goodfellow et al. (2016). *Deep Learning*.

---
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
