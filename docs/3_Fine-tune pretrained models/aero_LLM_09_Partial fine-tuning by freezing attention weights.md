
# Tinh Chỉnh Từng Phần Bằng Cách Đóng Băng Trọng Số Attention: Chiến Lược Tối Ưu Hóa Tham Số Cho LLM

## Tóm tắt

Bài viết này nghiên cứu phương pháp tinh chỉnh từng phần (partial fine-tuning) các mô hình ngôn ngữ lớn (LLMs) thông qua việc đóng băng (freezing) cơ chế Self-Attention và chỉ cập nhật các lớp Feed-Forward (MLP) và Layer Normalization. Dựa trên dữ liệu thực nghiệm từ thử thách "Partial fine-tuning by freezing attention weights", nghiên cứu phân tích tác động của chiến lược này đến tốc độ huấn luyện, bộ nhớ GPU và khả năng thích nghi phong cách văn học. Kết quả cho thấy việc đóng băng Attention giúp giảm đáng kể số lượng tham số cần cập nhật mà vẫn duy trì được hiệu quả học tập tương đương với tinh chỉnh toàn phần trong các tác vụ hẹp.

---

## 1. Giới thiệu

Fine-tuning toàn bộ (Full Fine-tuning) một mô hình Transformer đòi hỏi tài nguyên tính toán cực lớn. Để tối ưu hóa, các kỹ thuật tinh chỉnh hiệu quả tham số (Parameter-Efficient Fine-Tuning - PEFT) đã ra đời.

Theo tài liệu thực nghiệm , một trong những phương pháp đơn giản nhưng hiệu quả là "Partial Fine-tuning". Thay vì cập nhật toàn bộ 125 triệu tham số (đối với GPT-Neo 125M), chúng ta có thể đóng băng các thành phần đã học tốt các mối liên kết ngôn ngữ toàn cục – cụ thể là cơ chế Attention – và tập trung vào các lớp MLP, nơi chứa đựng phần lớn tri thức về các đặc trưng cụ thể của dữ liệu.

Mục tiêu nghiên cứu:
* Phân tích cơ chế đóng băng trọng số trong kiến trúc Transformer.
* Đo lường tỷ lệ tham số được huấn luyện so với tổng số tham số.
* Đánh giá hiệu quả ổn định gradient và hội tụ của hàm mất mát.

---

## 2. Cơ sở lý thuyết

### 2.1. Cấu trúc Transformer Block

Mỗi block Transformer gồm hai thành phần chính:
1. **Multi-Head Self-Attention (MSA):** Học các quan hệ ngữ cảnh giữa các token.
2. **Multi-Layer Perceptron (MLP):** Thực hiện biến đổi phi tuyến các đặc trưng.

Đầu ra của một block:
[
h' = \text{LayerNorm}(x + \text{MSA}(x))
]
[
y = \text{LayerNorm}(h' + \text{MLP}(h'))
]

---

### 2.2. Cơ chế đóng băng tham số (Freezing)

Khi đóng băng một lớp, chúng ta đặt thuộc tính:
[
\text{requires\_grad} = \text{False}
]
Điều này dẫn đến việc bỏ qua tính toán gradient cho các tham số đó trong quá trình lan truyền ngược (backpropagation):
[
\frac{\partial \mathcal{L}}{\partial W_{attention}} = 0
]

---

### 2.3. Tỷ lệ tham số huấn luyện

Nếu gọi $P_{total}$ là tổng tham số và $P_{trainable}$ là tham số được cập nhật:
[
R = \frac{P_{trainable}}{P_{total}}
]
Trong bài toán đóng băng Attention, tỷ lệ này thường dao động quanh mức 0.5 (tương đương 50% tham số), giúp tiết kiệm đáng kể tài nguyên GPU.

---

## 3. Phương pháp nghiên cứu

### 3.1. Thiết lập thí nghiệm

* **Mô hình gốc:** EleutherAI/gpt-neo-125M.
* **Chiến lược:** 
    * Đóng băng tất cả các lớp `Attention`.
    * Đóng băng các `Embedding` layers.
    * Chỉ cho phép huấn luyện các lớp `Linear` trong MLP và các lớp `LayerNorm`.
* **Dữ liệu:** Văn bản phong cách Alice và Edgar.

---

### 3.2. Quy trình thực hiện

1. Nạp mô hình tiền huấn luyện.
2. Duyệt qua tất cả các tham số (`named_parameters`).
3. Kiểm tra tên tham số (`"attn"` hoặc `"embed"`).
4. Thiết lập `requires_grad = False` cho các tham số trùng khớp.
5. Khởi tạo Optimizer (chỉ nạp các tham số có `requires_grad = True`).

---

## 4. Kết quả thực nghiệm

### 4.1. Phân tích số lượng tham số

Theo dữ liệu từ , kết quả thống kê cho thấy:
* Tổng tham số: ~125,000,000.
* Tham số huấn luyện sau khi đóng băng Attention: ~65,000,000.
* **Tỷ lệ giảm:** Gần 48%.

---

### 4.2. Khả năng hội tụ

Mặc dù đóng băng một phần quan trọng của mô hình, đồ thị hàm mất mát ($\mathcal{L}$) vẫn cho thấy xu hướng giảm ổn định:
[
\lim_{t \to \infty} \mathcal{L}(t) = \mathcal{L}_{min}
]
Đặc biệt, việc đóng băng Attention giúp giảm hiện tượng "catastrophic forgetting" (quên kiến thức cũ), vì các cấu trúc ngôn ngữ cơ bản trong Attention được giữ nguyên.

---

### 3.3. Hiệu năng tính toán

* **Bộ nhớ GPU:** Giảm khoảng 25-30% do không cần lưu trữ trạng thái optimizer (moments) cho các trọng số Attention.
* **Tốc độ:** Tăng nhẹ do giảm số lượng phép tính cập nhật trọng số.

---

## 5. Thảo luận

### 5.1. Tại sao lại đóng băng Attention?

Cơ chế Attention của các mô hình tiền huấn luyện đã rất mạnh trong việc hiểu cấu trúc câu và quan hệ ngữ pháp. Trong khi đó, các lớp MLP thường chịu trách nhiệm "ghi nhớ" các sự kiện hoặc đặc trưng cụ thể của miền dữ liệu (domain-specific knowledge). Vì vậy, tinh chỉnh MLP là đủ để mô hình học phong cách mới.

---

### 5.2. So sánh với LoRA

Trong khi LoRA thêm các ma trận bổ sung, "Partial Fine-tuning" trực tiếp sử dụng các tham số có sẵn. Đây là phương pháp "PEFT sơ khai" nhưng cực kỳ ổn định và không làm tăng độ trễ khi suy luận (inference latency).

---

## 6. Kết luận

Tinh chỉnh từng phần bằng cách đóng băng trọng số Attention là một chiến lược hiệu quả để tối ưu hóa quá trình huấn luyện LLM. Nó cung cấp sự cân bằng giữa hiệu năng (accuracy) và chi phí (computation). Đối với các nhiệm vụ chuyển đổi phong cách văn học như Alice-Edgar, phương pháp này chứng minh rằng chúng ta không cần cập nhật toàn bộ mô hình để đạt được kết quả mong muốn.

---

## Tài liệu tham khảo

1. Tài liệu thực nghiệm: Partial fine-tuning by freezing attention weights.
2. Vaswani et al. (2017). *Attention Is All You Need*.
3. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
4. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*.

---
