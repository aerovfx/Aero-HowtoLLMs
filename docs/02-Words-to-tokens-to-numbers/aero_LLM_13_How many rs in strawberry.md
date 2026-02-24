Phân tích Lỗi Đếm Ký Tự trong Mô Hình Ngôn Ngữ Lớn: Trường hợp “How many r’s in strawberry?”

⸻

Tóm tắt

Câu hỏi “How many r’s are in strawberry?” đã trở thành một ví dụ điển hình cho việc mô hình ngôn ngữ lớn (LLMs) đôi khi trả lời sai các nhiệm vụ đếm ký tự đơn giản. Dựa trên tài liệu đính kèm và mở rộng học thuật, bài viết này phân tích nguyên nhân từ góc độ tokenization, biểu diễn xác suất, và kiến trúc Transformer của các mô hình do OpenAI phát triển. Chúng tôi xây dựng mô hình toán học để giải thích vì sao nhiệm vụ đếm ký tự không tương thích tự nhiên với cơ chế dự đoán xác suất theo token.

⸻

1. Giới thiệu

Câu hỏi:

How many r’s are in “strawberry”?

Đáp án đúng:

\text{count}("r", "strawberry") = 3

Tuy nhiên, nhiều LLM từng trả lời sai (ví dụ: 2).

Vấn đề không nằm ở “kiến thức” mà ở cách mô hình xử lý chuỗi ký tự.

⸻

2. Phân tích Dưới Góc độ Tokenization

2.1 Biểu diễn Chuỗi

Chuỗi ký tự:

S = (s_1, s_2, ..., s_n)

Với:

S = \text{"strawberry"}

Nếu xử lý ở mức ký tự:

n = 10

Và:

\sum_{i=1}^{10} \mathbf{1}(s_i = r) = 3

Trong đó:

\mathbf{1}(\cdot)

là hàm chỉ thị.

⸻

2.2 Tokenization Thực tế

LLMs không xử lý ở mức ký tự mà theo token:

T = (t_1, t_2, ..., t_m)

Ví dụ (minh họa):

straw + berry

Hoặc:

st + raw + berry

Số token m < n.

Do đó, thông tin ký tự r không được biểu diễn trực tiếp mà nằm bên trong embedding vector của token.

⸻

3. Mô hình Xác suất của LLM

LLM học phân phối:

P(t_i | t_{<i})

Toàn chuỗi:

P(T) = \prod_{i=1}^{m} P(t_i | t_{<i})

Mô hình không tối ưu cho phép toán đếm ký tự, mà tối ưu cho:

\mathcal{L} = - \sum_{i=1}^{m} \log P(t_i | t_{<i})

Tức là tối thiểu hóa cross-entropy giữa token dự đoán và token thật.

⸻

4. Nguyên nhân Sai Số

4.1 Không có Cơ chế Đếm Tường minh

Bài toán đếm yêu cầu:

f(S) = \sum_{i=1}^{n} \mathbf{1}(s_i = r)

Nhưng mô hình chỉ có:

g(T) = \text{argmax}_{y} P(y | T)

Không có bước lặp tuần tự ở mức ký tự.

⸻

4.2 Biểu diễn Vector Phân tán

Embedding:

E(t) \in \mathbb{R}^d

Thông tin về ký tự r nằm phân tán trong không gian:

E(\text{"strawberry"}) = f(E(\text{"straw"}), E(\text{"berry"}))

Không tồn tại biến riêng biệt đếm số lần xuất hiện của r.

⸻

4.3 Attention Không Tương đương Đếm

Self-attention:

\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

Attention học mối quan hệ ngữ nghĩa, không học phép toán cộng số học chính xác trên ký tự.

⸻

5. Phân tích Toán học Sai số Xác suất

Giả sử mô hình ước lượng xác suất:

P(Y = k | S)

Trong đó:
	•	Y: số lượng r
	•	k \in \{0,1,2,3,...\}

Do không huấn luyện trực tiếp cho nhiệm vụ đếm:

P(Y=2) \approx P(Y=3)

Nếu trong dữ liệu huấn luyện, mẫu “2” phổ biến hơn, mô hình có thể thiên lệch.

⸻

6. So sánh với Máy Tính Thuật toán

Thuật toán truyền thống:

O(n)

Pseudo-code:

count = 0
for char in string:
    if char == 'r':
        count += 1

LLM không thực thi thuật toán tuần tự như vậy.

⸻

7. Phân tích Dưới Góc độ Thông tin

Entropy của chuỗi ký tự:

H(S) = - \sum_{c \in \Sigma} P(c)\log P(c)

LLM tối ưu hóa dự đoán token, không tối ưu hóa:

I(Y; S)

(tương hỗ thông tin giữa số lượng r và chuỗi ký tự)

⸻

8. Tại sao Mô hình Mới Ít Sai Hơn?

Các mô hình mới có thể:
	•	Sử dụng chain-of-thought
	•	Mô phỏng đếm nội bộ
	•	Tăng kích thước context

Nhưng vẫn không đảm bảo 100% chính xác vì không phải mô hình symbolic.

⸻

9. Hàm Đếm như một Bài toán Học Máy

Ta có thể định nghĩa:

h_\theta(S) \approx \sum_{i=1}^{n} \mathbf{1}(s_i = r)

Với:

\theta = \text{tham số mô hình}

Sai số kỳ vọng:

\mathbb{E}[(h_\theta(S) - f(S))^2]

Không được tối ưu trực tiếp trong huấn luyện LLM.

⸻

10. Thảo luận

Hiện tượng “How many r’s in strawberry?” minh họa:
	•	Tokenization làm mất granularity ký tự
	•	LLM là mô hình xác suất, không phải bộ xử lý ký tự chính xác
	•	Attention ≠ thuật toán đếm

Đây là khác biệt giữa:
	•	Hệ thống symbolic computation
	•	Hệ thống neural probabilistic modeling

⸻

11. Kết luận

Sai số đếm ký tự có thể giải thích bởi:

\text{Token-level modeling} \neq \text{Character-level counting}

\min \mathcal{L}_{\text{cross-entropy}} \not\Rightarrow \min \mathcal{L}_{\text{counting}}

Do đó, nhiệm vụ tưởng chừng đơn giản lại không phù hợp tự nhiên với mục tiêu tối ưu của LLM.

⸻

Tài liệu tham khảo
	1.	Vaswani et al. (2017). Attention Is All You Need.
	2.	Shannon (1948). A Mathematical Theory of Communication.
	3.	Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
	4.	Brown et al. (2020). Language Models are Few-Shot Learners.
	5.	Merrill et al. (2022). On the Ability of Transformers to Perform Counting.

