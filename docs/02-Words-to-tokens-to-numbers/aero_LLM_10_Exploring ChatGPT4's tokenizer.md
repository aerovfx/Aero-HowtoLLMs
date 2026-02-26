
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [02 Words to tokens to numbers](../index.md)

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
Khám phá Cơ chế Tokenizer của ChatGPT-4: Phân tích Kỹ thuật và Mô hình Toán học

Tóm tắt

Tokenizer đóng vai trò nền tảng trong các mô hình ngôn ngữ lớn (Large Language Models – LLMs), đặc biệt là các hệ thống do OpenAI phát triển như GPT-4. Bài viết này phân tích cơ chế hoạt động của tokenizer trong GPT-4, tập trung vào thuật toán Byte Pair Encoding (BPE), biểu diễn xác suất, cấu trúc từ vựng, cũng như các mô hình toán học minh hoạ. Ngoài ra, bài viết mở rộng so sánh với các phương pháp token hóa hiện đại và thảo luận về ảnh hưởng của tokenizer đến hiệu năng mô hình.

⸻

1. Giới thiệu

Trong các mô hình Transformer, văn bản không được xử lý trực tiếp ở mức ký tự hoặc từ hoàn chỉnh, mà được chuyển đổi thành token — các đơn vị rời rạc đại diện cho chuỗi ký tự. Quá trình này gọi là tokenization.

Cho chuỗi đầu vào:

X = (x_1, x_2, ..., x_n)

Tokenizer thực hiện ánh xạ:

f: X \rightarrow T = (t_1, t_2, ..., t_m)

Trong đó:
	•	x_i: ký tự hoặc byte
	•	t_j: token trong từ vựng V
	•	m \leq n

⸻

2. Byte Pair Encoding (BPE)

2.1 Nguyên lý cơ bản

BPE là thuật toán nén dữ liệu được điều chỉnh để xây dựng từ vựng token. Ý tưởng chính:
	1.	Bắt đầu với tập ký tự cơ sở (byte-level).
	2.	Tìm cặp ký tự xuất hiện nhiều nhất.
	3.	Gộp cặp đó thành một token mới.
	4.	Lặp lại cho đến khi đạt kích thước từ vựng mong muốn.

⸻

2.2 Mô hình toán học của BPE

Giả sử ta có tập dữ liệu huấn luyện D gồm các chuỗi ký tự.

Tần suất xuất hiện của cặp ký tự (a,b):

\text{freq}(a,b) = \sum_{w \in D} \text{count}_{w}(a,b)

Cặp được chọn để gộp:

(a^*, b^*) = \arg\max_{(a,b)} \text{freq}(a,b)

Sau mỗi bước gộp, từ vựng được cập nhật:

V_{k+1} = V_k \cup \{ a^*b^* \}

⸻

2.3 Ví dụ minh họa

Chuỗi:

low
lower
lowest

Ban đầu token theo ký tự:

l o w
l o w e r
l o w e s t

Nếu cặp lo xuất hiện nhiều nhất → tạo token mới:

lo w
lo w e r
lo w e s t

Tiếp tục quá trình đến khi đạt kích thước từ vựng yêu cầu.

⸻

3. Biểu diễn Vector của Token

Sau khi token hóa, mỗi token t_i \in V được ánh xạ sang embedding vector:

E: V \rightarrow \mathbb{R}^d

Với:
	•	d: chiều không gian embedding (ví dụ 768, 1024, 4096…)

Chuỗi token:

T = (t_1, t_2, ..., t_m)

được chuyển thành ma trận embedding:

\mathbf{X} =
\begin{bmatrix}
E(t_1) \\
E(t_2) \\
\vdots \\
E(t_m)
\end{bmatrix}
\in \mathbb{R}^{m \times d}

⸻

4. Tokenization ở mức Byte

GPT-4 sử dụng byte-level BPE, nghĩa là mọi chuỗi Unicode đều được biểu diễn qua:

\text{Unicode} \rightarrow \text{UTF-8 bytes}

Điều này đảm bảo:

\forall s \in \text{Unicode}, \exists \text{token sequence}

Không xảy ra trường hợp “out-of-vocabulary”.

⸻

5. Xác suất và Ngôn ngữ học Thống kê

Sau tokenization, mô hình học phân phối xác suất:

P(t_i | t_1, ..., t_{i-1})

Toàn bộ xác suất chuỗi:

P(T) = \prod_{i=1}^{m} P(t_i | t_{<i})

Loss function huấn luyện:

\mathcal{L} = - \sum_{i=1}^{m} \log P(t_i | t_{<i})

Tokenizer ảnh hưởng trực tiếp đến:
	•	Độ dài chuỗi m
	•	Phân phối xác suất
	•	Độ phức tạp tính toán O(m^2) trong self-attention

⸻

6. Ảnh hưởng của Tokenizer đến Hiệu Năng

6.1 Độ dài chuỗi

Nếu tokenizer tạo quá nhiều token cho một từ hiếm:

\text{computational cost} \propto m^2

Chi phí attention tăng nhanh khi m lớn.

⸻

6.2 Độ nén ngôn ngữ

Entropy của hệ token:

H(T) = - \sum_{t \in V} P(t)\log P(t)

Tokenizer tốt sẽ:
	•	Giảm entropy
	•	Tăng tính nén
	•	Giữ cấu trúc ngữ nghĩa

⸻

7. So sánh với các phương pháp khác

Phương pháp	Nguyên lý	Ưu điểm	Nhược điểm
Word-level	Theo từ hoàn chỉnh	Dễ hiểu	OOV cao
Character-level	Theo ký tự	Không OOV	Chuỗi dài
BPE	Gộp cặp phổ biến	Cân bằng	Phụ thuộc corpus
Unigram LM	Mô hình xác suất	Linh hoạt	Tính toán phức tạp


⸻

8. Hạn chế và Thách thức
	1.	Phụ thuộc ngôn ngữ
Ngôn ngữ không dấu và có dấu (ví dụ tiếng Việt) có thể bị phân mảnh token.
	2.	Bias thống kê
Token phổ biến chiếm ưu thế trong huấn luyện.
	3.	Không phản ánh cấu trúc ngữ pháp thực sự

⸻

9. Hướng phát triển tương lai
	•	Adaptive tokenization
	•	Dynamic vocabulary
	•	Morphology-aware tokenization
	•	Neural tokenizers học trực tiếp từ dữ liệu

⸻

10. Kết luận

Tokenizer không chỉ là bước tiền xử lý, mà là thành phần quyết định cấu trúc xác suất và hiệu năng của mô hình ngôn ngữ lớn. BPE cung cấp sự cân bằng giữa tính nén và khả năng biểu diễn, trong khi byte-level encoding đảm bảo tính toàn diện với Unicode.

Về mặt toán học, tokenizer ảnh hưởng đến:

m, \quad H(T), \quad \mathcal{L}, \quad O(m^2)

Do đó, việc tối ưu tokenizer có thể cải thiện cả hiệu suất lẫn chất lượng sinh ngôn ngữ của mô hình.

⸻

Tài liệu tham khảo
	1.	Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units.
	2.	Vaswani, A. et al. (2017). Attention Is All You Need.
	3.	Kudo, T. (2018). Subword Regularization.
	4.	Brown, T. et al. (2020). Language Models are Few-Shot Learners.
	5.	Jurafsky, D. & Martin, J. (Speech and Language Processing).
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
