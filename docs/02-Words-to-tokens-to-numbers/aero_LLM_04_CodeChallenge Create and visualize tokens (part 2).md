
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
# Tạo và trực quan hóa Token (Phần 2): Phân tích hình học không gian embedding và Attention Map trong mô hình Transformer

---

## Tóm tắt

Bài viết này mở rộng phân tích quá trình tạo và trực quan hóa token trong mô hình ngôn ngữ lớn, tập trung vào hình học của không gian embedding, cấu trúc attention map và các phương pháp giảm chiều để quan sát đặc trưng học được. Nghiên cứu dựa trên kiến trúc Transformer được đề xuất bởi Vaswani et al. (2017) và các mô hình GPT do OpenAI phát triển. Các công thức toán học được sử dụng để mô tả cấu trúc đại số tuyến tính của embedding, self-attention và phép chiếu không gian.

---

# 1. Giới thiệu

Trong mô hình Transformer, token không chỉ là ID số nguyên mà là:

[
t_i \rightarrow e_i \in \mathbb{R}^d
]

Không gian embedding có thể xem như một đa tạp (manifold) cao chiều, trong đó:

* Khoảng cách phản ánh quan hệ ngữ nghĩa
* Hướng vector phản ánh quan hệ ngữ pháp

Việc trực quan hóa giúp ta hiểu:

* Cụm ngữ nghĩa
* Sự phân tách lớp từ loại
* Ảnh hưởng của attention

---

# 2. Không gian embedding: Góc nhìn hình học

Giả sử từ vựng có kích thước (N), embedding dimension (d):

[
E \in \mathbb{R}^{N \times d}
]

Mỗi token là một điểm:

[
e_i \in \mathbb{R}^d
]

Khoảng cách cosine giữa hai token:

[
\text{cosine}(e_i, e_j)
=======================

\frac{e_i \cdot e_j}
{|e_i||e_j|}
]

Nếu:

[
\text{cosine}(e_i, e_j) \approx 1
]

→ Hai token gần nhau về ngữ nghĩa.

---

# 3. Biến đổi qua Transformer Layer

Một layer Transformer gồm:

1. Multi-head attention
2. Feed-forward network

Biểu diễn đầu ra:

[
Z' = \text{LayerNorm}(Z + \text{Attention}(Z))
]

[
Z'' = \text{LayerNorm}(Z' + \text{MLP}(Z'))
]

Qua nhiều layer:

[
Z^{(L)} = f^{(L)}(Z^{(0)})
]

Không gian embedding ban đầu bị biến đổi phi tuyến.

---

# 4. Trực quan hóa Attention Map

Attention matrix:

[
A = \text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
]

Phần tử:

[
A_{ij}
======

P(\text{token } j \mid \text{token } i)
]

Tính chất:

[
\sum_j A_{ij} = 1
]

Ma trận (A) có thể trực quan hóa dưới dạng heatmap:

* Vùng sáng → tương tác mạnh
* Vùng tối → ít tương tác

---

# 5. Phân tích Eigenstructure của Embedding

Ma trận hiệp phương sai:

[
\Sigma = \frac{1}{N} E^T E
]

Giải bài toán:

[
\Sigma v = \lambda v
]

Trị riêng lớn phản ánh:

* Hướng phương sai lớn nhất
* Cấu trúc ngữ nghĩa chính

Chiếu embedding:

[
E_{proj} = E W_k
]

với (W_k) chứa (k) vector riêng lớn nhất.

---

# 6. t-SNE và cấu trúc cụm

t-SNE tối ưu:

[
\min_Y D_{KL}(P | Q)
]

Trong đó:

[
P_{ij} =
\frac
{\exp(-|x_i - x_j|^2 / 2\sigma^2)}
{\sum_{k,l} \exp(-|x_k - x_l|^2 / 2\sigma^2)}
]

[
Q_{ij} =
\frac
{(1 + |y_i - y_j|^2)^{-1}}
{\sum_{k,l}(1 + |y_k - y_l|^2)^{-1}}
]

Mục tiêu:

[
D_{KL}(P|Q)
===========

\sum_{i,j} P_{ij}
\log
\frac{P_{ij}}{Q_{ij}}
]

Kết quả:

* Token cùng chủ đề → cụm gần nhau
* Token trái nghĩa → phân tách

---

# 7. Biến đổi qua nhiều tầng (Representation Drift)

Giả sử embedding tại layer (l):

[
Z^{(l)}
]

Khoảng cách giữa hai layer:

[
\Delta^{(l)} =
| Z^{(l)} - Z^{(l-1)} |
]

Quan sát thực nghiệm:

* Layer đầu → cú pháp
* Layer giữa → ngữ nghĩa
* Layer cuối → dự đoán xác suất

---

# 8. Liên hệ với mô hình tự hồi quy

GPT tối ưu:

[
P(x) = \prod_{t=1}^{T} P(x_t | x_{<t})
]

Logits:

[
\text{logits} = Z^{(L)} W_{out}
]

Softmax:

[
P(x_t | x_{<t})
===============

\frac
{\exp(z_t W_{out})}
{\sum_j \exp(z_j W_{out})}
]

Việc trực quan hóa logits cho thấy:

* Phân phối xác suất
* Độ chắc chắn của mô hình

---

# 9. Phân tích độ phức tạp

Self-attention:

[
\mathcal{O}(L T^2 d)
]

Visualization chi phí:

* PCA: (\mathcal{O}(Nd^2))
* t-SNE: (\mathcal{O}(N^2))

---

# 10. Thảo luận

Từ góc nhìn đại số tuyến tính:

* Embedding là ánh xạ tuyến tính
* Attention là phép chiếu có trọng số
* MLP là biến đổi phi tuyến

Toàn bộ Transformer có thể xem như:

[
f: \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}
]

Việc trực quan hóa giúp:

1. Phát hiện bias
2. Phân tích cấu trúc
3. Giải thích hành vi mô hình

---

# 11. Kết luận

Tạo và trực quan hóa token (phần 2) cho thấy:

* Không gian embedding có cấu trúc hình học rõ ràng
* Attention phản ánh tương tác ngữ cảnh
* Biến đổi qua layer mang tính phi tuyến mạnh

Toán học giúp ta hiểu rằng token là điểm trong không gian vector cao chiều, và Transformer là chuỗi phép biến đổi hình học phức tạp.

---

# Tài liệu tham khảo

1. Vaswani, A. et al. (2017). *Attention Is All You Need*.
2. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*.
3. van der Maaten, L., Hinton, G. (2008). *Visualizing Data using t-SNE*.
4. Jolliffe, I. (2002). *Principal Component Analysis*.
5. Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Tại sao văn bản cần được đánh số?](aero_LLM_01_Why text needs to be numbered.md) | [Xem bài viết →](aero_LLM_01_Why text needs to be numbered.md) |
| [Phân tích và chuyển đổi văn bản thành chuỗi token được đánh số: Cơ sở toán học và ứng dụng trong mô hình ngôn ngữ lớn](aero_LLM_02_Parsing text to numbered tokens.md) | [Xem bài viết →](aero_LLM_02_Parsing text to numbered tokens.md) |
| [Tạo và trực quan hóa Token trong mô hình ngôn ngữ lớn: Cơ sở toán học và phân tích biểu diễn](aero_LLM_03_CodeChallenge Create and visualize tokens (part 1).md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Create and visualize tokens (part 1).md) |
| 📌 **[Tạo và trực quan hóa Token (Phần 2): Phân tích hình học không gian embedding và Attention Map trong mô hình Transformer](aero_LLM_04_CodeChallenge Create and visualize tokens (part 2).md)** | [Xem bài viết →](aero_LLM_04_CodeChallenge Create and visualize tokens (part 2).md) |
| [Chuẩn bị văn bản cho Tokenization trong mô hình Transformer: Cơ sở lý thuyết và phân tích toán học](aero_LLM_05_Preparing text for tokenization.md) | [Xem bài viết →](aero_LLM_05_Preparing text for tokenization.md) |
| [Phân tích quy trình Tokenization qua ví dụ *The Time Machine*: Cơ sở thuật toán và mô hình hóa toán học](aero_LLM_06_CodeChallenge Tokenizing The Time Machine.md) | [Xem bài viết →](aero_LLM_06_CodeChallenge Tokenizing The Time Machine.md) |
| [So sánh Tokenization mức ký tự, từ và subword: Phân tích lý thuyết và mô hình toán học](aero_LLM_07_Tokenizing characters vs. subwords vs. words.md) | [Xem bài viết →](aero_LLM_07_Tokenizing characters vs. subwords vs. words.md) |
| [aero_LLM_08_Byte-pair encoding algorithm.md](aero_LLM_08_Byte-pair encoding algorithm.md) | [Xem bài viết →](aero_LLM_08_Byte-pair encoding algorithm.md) |
| [Thuật toán Byte Pair Encoding (BPE) và Bài toán Tối ưu Kích thước Từ vựng trong Mô hình Ngôn ngữ](aero_LLM_09_CodeChallenge Byte-pair encoding to a desired vocab size.md) | [Xem bài viết →](aero_LLM_09_CodeChallenge Byte-pair encoding to a desired vocab size.md) |
| [aero_LLM_10_Exploring ChatGPT4's tokenizer.md](aero_LLM_10_Exploring ChatGPT4's tokenizer.md) | [Xem bài viết →](aero_LLM_10_Exploring ChatGPT4's tokenizer.md) |
| [aero_LLM_11_CodeChallenge Token count by subword length (part 1).md](aero_LLM_11_CodeChallenge Token count by subword length (part 1).md) | [Xem bài viết →](aero_LLM_11_CodeChallenge Token count by subword length (part 1).md) |
| [aero_LLM_12_CodeChallenge Token count by subword length (part 2).md](aero_LLM_12_CodeChallenge Token count by subword length (part 2).md) | [Xem bài viết →](aero_LLM_12_CodeChallenge Token count by subword length (part 2).md) |
| [aero_LLM_13_How many rs in strawberry.md](aero_LLM_13_How many rs in strawberry.md) | [Xem bài viết →](aero_LLM_13_How many rs in strawberry.md) |
| [aero_LLM_14_CodeChallenge Create your algorithmic rapper name ).md](aero_LLM_14_CodeChallenge Create your algorithmic rapper name ).md) | [Xem bài viết →](aero_LLM_14_CodeChallenge Create your algorithmic rapper name ).md) |
| [aero_LLM_15_Tokenization in BERT.md](aero_LLM_15_Tokenization in BERT.md) | [Xem bài viết →](aero_LLM_15_Tokenization in BERT.md) |
| [aero_LLM_16_CodeChallenge Character counts in BERT tokens.md](aero_LLM_16_CodeChallenge Character counts in BERT tokens.md) | [Xem bài viết →](aero_LLM_16_CodeChallenge Character counts in BERT tokens.md) |
| [aero_LLM_17_Translating between tokenizers.md](aero_LLM_17_Translating between tokenizers.md) | [Xem bài viết →](aero_LLM_17_Translating between tokenizers.md) |
| [aero_LLM_18_CodeChallenge More on token translation.md](aero_LLM_18_CodeChallenge More on token translation.md) | [Xem bài viết →](aero_LLM_18_CodeChallenge More on token translation.md) |
| [aero_LLM_19_CodeChallenge Tokenization compression ratios.md](aero_LLM_19_CodeChallenge Tokenization compression ratios.md) | [Xem bài viết →](aero_LLM_19_CodeChallenge Tokenization compression ratios.md) |
| [aero_LLM_20_Tokenization in different languages.md](aero_LLM_20_Tokenization in different languages.md) | [Xem bài viết →](aero_LLM_20_Tokenization in different languages.md) |
| [aero_LLM_21_CodeChallenge Zipf's law in characters and tokens.md](aero_LLM_21_CodeChallenge Zipf's law in characters and tokens.md) | [Xem bài viết →](aero_LLM_21_CodeChallenge Zipf's law in characters and tokens.md) |
| [aero_LLM_22_Word variations in Claude tokenizer.md](aero_LLM_22_Word variations in Claude tokenizer.md) | [Xem bài viết →](aero_LLM_22_Word variations in Claude tokenizer.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
