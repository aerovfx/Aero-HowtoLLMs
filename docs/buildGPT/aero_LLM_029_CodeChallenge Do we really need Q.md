Dưới đây là **bài viết khoa học trình bày theo định dạng Markdown**, được xây dựng dựa trên tài liệu **“CodeChallenge: Do We Really Need Q?”**, có bổ sung phân tích học thuật và trích dẫn nguồn.

---

# 🧠 Phân Tích Nhân Quả Trong GPT-2: Vai Trò Của Ma Trận Query Thông Qua Can Thiệp Tham Số

## Tóm tắt (Abstract)

Nghiên cứu này phân tích vai trò của ma trận Query (WQ) trong cơ chế self-attention của GPT-2 thông qua phương pháp can thiệp nhân quả (causal mechanistic interpretability). Bằng cách thay thế có kiểm soát các trọng số WQ bằng nhiễu ngẫu nhiên có cùng đặc tính thống kê, nghiên cứu đánh giá ảnh hưởng của thành phần này lên chất lượng sinh văn bản. Kết quả cho thấy GPT-2 vẫn duy trì được khả năng sinh câu hợp cú pháp trong giai đoạn đầu, ngay cả khi một phần Query bị phá vỡ, phản ánh tính dư thừa và khả năng phân tán thông tin của kiến trúc Transformer.

---

## 1. Giới thiệu (Introduction)

Cơ chế self-attention là nền tảng của các mô hình Transformer, trong đó ba thành phần chính là Query (Q), Key (K) và Value (V). Trong các nghiên cứu truyền thống, ba thành phần này thường được xem là không thể tách rời.

Tuy nhiên, tài liệu *CodeChallenge: Do We Really Need Q?* đề xuất một hướng tiếp cận mới: can thiệp trực tiếp vào trọng số Q để đánh giá vai trò nhân quả của nó trong quá trình suy luận của mô hình. Phương pháp này thuộc lĩnh vực *causal mechanistic interpretability* 

---

## 2. Cơ sở lý thuyết (Theoretical Background)

### 2.1. Self-Attention trong Transformer

Cơ chế attention được mô tả bằng công thức:

[
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

Trong đó:

* (Q): Query matrix
* (K): Key matrix
* (V): Value matrix
* (d_k): số chiều vector khóa

Q đóng vai trò xác định vị trí cần tập trung thông tin từ K và V.

---

### 2.2. Interpretability Nhân Quả

Khác với interpretability quan sát (observational), interpretability nhân quả tập trung vào việc:

* Can thiệp tham số,
* Đánh giá tác động trực tiếp,
* Xác định vai trò chức năng.

Phương pháp này tương tự như thí nghiệm trong khoa học tự nhiên, nơi một biến được thay đổi có kiểm soát 

---

## 3. Phương pháp nghiên cứu (Methodology)

### 3.1. Thiết lập mô hình

Nghiên cứu sử dụng hai phiên bản GPT-2:

* Mô hình gốc (CPU) làm bản sao lưu,
* Mô hình can thiệp (GPU) để chỉnh sửa tham số.

Việc tách hai phiên bản cho phép khôi phục nhanh tham số gốc thông qua `state_dict` 

---

### 3.2. Kiểm soát ngẫu nhiên (Random Seed Control)

Cùng một seed ngẫu nhiên được thiết lập cho CPU và GPU. Tuy nhiên, kết quả sinh văn bản vẫn khác nhau do:

* Sai khác làm tròn số,
* Cách xử lý số thực khác nhau,
* Trình sinh số ngẫu nhiên phụ thuộc phần cứng.

Điều này ảnh hưởng đến khả năng tái lập thí nghiệm 

---

### 3.3. Thay thế ma trận Query

Quy trình can thiệp gồm:

1. Trích xuất ma trận WQ của block đầu tiên,
2. Tính mean và standard deviation,
3. Sinh nhiễu Gaussian tương ứng,
4. Ghi đè lên WQ gốc.

Mục tiêu là giữ nguyên phân bố thống kê để tránh làm lệch thí nghiệm 

---

### 3.4. Can thiệp tuần tự theo layer

Trong giai đoạn mở rộng, nghiên cứu:

* Thay thế WQ theo từng block,
* Sinh văn bản sau mỗi bước,
* Quan sát sự suy giảm chất lượng.

Cách tiếp cận này cho phép đánh giá mức độ nhạy cảm theo chiều sâu mô hình.

---

## 4. Kết quả thực nghiệm (Experimental Results)

### 4.1. Thay thế WQ ở một block

Sau khi thay thế WQ của block đầu tiên:

* Văn bản vẫn mạch lạc,
* Ngữ pháp vẫn chính xác,
* Nội dung hơi suy giảm logic.

Ví dụ:

> “I'm in the process of making a new movie...”

Cho thấy mô hình vẫn hoạt động hiệu quả dù một thành phần bị phá vỡ 

---

### 4.2. Thay thế nhiều block liên tiếp

Khi mở rộng can thiệp:

| Số Block Bị Thay | Chất Lượng Văn Bản  |
| ---------------- | ------------------- |
| 1–3              | Gần như bình thường |
| 4–6              | Mất ngữ nghĩa       |
| 7–9              | Lặp, rối            |
| >9               | Nhiễu hoàn toàn     |

Kết quả cho thấy sự suy giảm có tính tích lũy 

---

### 4.3. Hiện tượng chuyển pha (Phase Transition)

Một đặc điểm nổi bật là sự chuyển pha:

1. Giai đoạn hợp cú pháp nhưng vô nghĩa,
2. Giai đoạn mất cấu trúc ngôn ngữ.

Điều này phản ánh quá trình suy sụp dần của biểu diễn nội tại.

---

## 5. Phân tích và Thảo luận (Discussion)

### 5.1. Tính dư thừa kiến trúc

Kết quả cho thấy:

* Thông tin không chỉ nằm trong WQ,
* K và V có thể bù trừ,
* Residual connection giúp ổn định.

Kiến trúc GPT-2 mang tính dư thừa cao.

---

### 5.2. Phân tán thông tin (Distributed Representation)

Tri thức không nằm ở một vị trí cụ thể mà:

* Phân bố trên nhiều layer,
* Chia sẻ qua nhiều head,
* Tái biểu diễn qua MLP.

Điều này làm tăng độ bền của mô hình trước nhiễu.

---

### 5.3. Ý nghĩa với interpretability

Nghiên cứu cho thấy:

* Quan sát trọng số là chưa đủ,
* Cần thí nghiệm can thiệp,
* Interpretability cần gắn với thực nghiệm.

Cách tiếp cận này mở đường cho phân tích nhân quả trong LLM.

---

### 5.4. Hạn chế

Một số hạn chế chính:

* Chỉ can thiệp WQ,
* Chưa phân tích từng head riêng lẻ,
* Đánh giá chủ yếu định tính.

Do đó, cần các thí nghiệm chi tiết hơn trong tương lai.

---

## 6. Ứng dụng và Hướng phát triển (Applications and Future Work)

### 6.1. Kiểm định độ bền mô hình

Phương pháp này có thể dùng để:

* Đánh giá robustness,
* Phát hiện điểm yếu,
* Thiết kế mô hình chịu lỗi.

---

### 6.2. An toàn AI (AI Safety)

Can thiệp tham số có thể giúp:

* Xác định neuron nguy hiểm,
* Loại bỏ hành vi lệch chuẩn,
* Thiết kế cơ chế kiểm soát.

---

### 6.3. Nghiên cứu tương lai

Các hướng mở rộng:

* Thay thế từng head,
* Can thiệp từng chiều embedding,
* Kết hợp probing tasks,
* Áp dụng cho GPT-3/4.

---

## 7. Kết luận (Conclusion)

Bài viết đã phân tích vai trò của ma trận Query trong GPT-2 thông qua phương pháp can thiệp nhân quả. Các kết quả chính bao gồm:

1. GPT-2 vẫn hoạt động khi WQ bị nhiễu cục bộ.
2. Chất lượng suy giảm dần theo số layer bị phá.
3. Kiến trúc có tính dư thừa cao.
4. Tri thức được phân bố phi tập trung.

Nghiên cứu cho thấy self-attention không phụ thuộc tuyệt đối vào Q, mà hoạt động dựa trên sự phối hợp toàn cục giữa nhiều thành phần.

---

## Tài liệu tham khảo (References)

[1] CodeChallenge: Do We Really Need Q?, Lecture Transcript.


---
