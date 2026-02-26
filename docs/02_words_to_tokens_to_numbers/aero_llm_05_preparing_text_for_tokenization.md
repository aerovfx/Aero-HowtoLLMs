
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [02 words to tokens to numbers](index.md)

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
# Chuẩn bị văn bản cho Tokenization trong mô hình Transformer: Cơ sở lý thuyết và phân tích toán học

---

## Tóm tắt

Bài báo này trình bày một cách hệ thống quy trình **chuẩn bị văn bản trước khi tokenization** trong các mô hình ngôn ngữ lớn dựa trên kiến trúc Attention Is All You Need. Nội dung phân tích các bước tiền xử lý (text normalization, cleaning, encoding), cơ chế mã hóa Byte Pair Encoding (BPE), và vai trò của tokenizer trong các mô hình GPT do OpenAI phát triển. Bài viết bổ sung các công thức toán học mô tả xác suất chuỗi, ánh xạ rời rạc–liên tục và cấu trúc đại số của quá trình mã hóa.

---

# 1. Giới thiệu

Trong các mô hình Transformer hiện đại như GPT-4, văn bản đầu vào không được xử lý trực tiếp dưới dạng ký tự mà phải trải qua quá trình:

$$
\text{Raw Text} \rightarrow \text{Normalization} \rightarrow \text{Tokenization} \rightarrow \text{Embedding}
$$

Tokenization đóng vai trò là cầu nối giữa:

* Không gian rời rạc của ký tự
* Không gian vector liên tục của embedding

Nếu gọi chuỗi văn bản ban đầu là:

X = (c_1, c_2, \dots, c_n)

thì tokenizer ánh xạ:

\tau: \Sigma^{\ast} \rightarrow \mathbb{Z}^m

với \Sigma là bảng chữ cái và \mathbb{Z}^m là chuỗi ID token.

$$
--- # 2. Chuẩn hóa văn bản (Text Normalization) Chuẩn hóa giúp đảm bảo tính nhất quán của dữ liệu huấn luyện. ## 2.1 Lowercasing Ánh xạ: f_{lower}(c) = \text{lowercase}(c) Ví dụ:
$$

\text{"ChatGPT"} \rightarrow \text{"chatgpt"}

$$
## 2.2 Unicode Normalization Văn bản Unicode có thể biểu diễn cùng một ký tự theo nhiều cách. Chuẩn NFC: \text{é} = e + \acute{} Chuẩn hóa đảm bảo: NFC(x_1) = NFC(x_2) nếu hai chuỗi tương đương về mặt ngữ nghĩa. --- # 3. Tokenization: Cơ sở xác suất Mô hình ngôn ngữ tối ưu xác suất: P(X) = \prod_{t=1}^{T} P(x_t \mid x_{\lt t}) Tuy nhiên, nếu làm việc ở mức ký tự: T = n Số bước dự đoán lớn → chi phí cao. Giải pháp: Chia thành token: X = (w_1, w_2, \dots, w_m), \quad m \lt  n Giảm độ dài chuỗi và tăng tính biểu diễn. --- # 4. Byte Pair Encoding (BPE) BPE được giới thiệu cho NLP bởi Sennrich et al. (2016). ## 4.1 Thuật toán Ban đầu: V_0 = { \text{tập ký tự đơn} } Lặp: 1. Tìm cặp ký tự xuất hiện nhiều nhất 2. Gộp thành token mới 3. Cập nhật từ vựng Giả sử tần suất cặp ((a,b)): f(a,b) = \sum_{i} \mathbb{I}[(a,b) \in X_i] Chọn: (a^{\ast}, b^{\ast}) = \arg\max_{a,b} f(a,b) Cập nhật: V_{k+1} = V_k \cup {ab} --- # 5. Không gian rời rạc và ánh xạ embedding Sau tokenization:
$$

$w_i$ \rightarrow id_i \in {1, \dots, |V|}

$$
Embedding matrix: E \in \mathbb{R}^{|V| \times d} Ánh xạ: e_i = E[id_i] Toàn bộ chuỗi:
$$

X \rightarrow ($e_1$, $e_2$, \dots, $e_m$)

$$
--- # 6. Phân tích độ phức tạp Nếu: * N là số ký tự * V là kích thước từ vựng Chi phí xây dựng BPE:
$$

$\mathcal${O}(N $\log$ V)

$$
Chi phí suy luận tokenization:
$$

$\mathcal${O}(m)

$$
--- # 7. Vấn đề Out-of-Vocabulary (OOV) Không như Word2Vec truyền thống, BPE đảm bảo:
$$

\forall x \in \Sigma^{\ast}, \exists \text{ decomposition into subwords}

$$
Ví dụ: tokenization → token + ization Điều này đảm bảo:
$$

P(x) > 0

$$
cho mọi chuỗi hợp lệ. --- # 8. So sánh với các phương pháp khác \mid Phương pháp     \mid Đặc điểm        \mid Hạn chế           \mid \mid --------------- \mid --------------- \mid ----------------- \mid \mid Word-level      \mid Ngắn, dễ hiểu   \mid OOV cao           \mid \mid Character-level \mid Không OOV       \mid Chuỗi dài         \mid \mid BPE             \mid Cân bằng        \mid Phụ thuộc dữ liệu \mid \mid Unigram LM      \mid Xác suất tối ưu \mid Tính toán cao     \mid Unigram Language Model tối ưu: \max_{\theta} \prod_i \sum_{z \in \mathcal{Z}(x_i)} P(z \mid \theta) --- # 9. Tác động đến Attention Độ dài chuỗi ảnh hưởng trực tiếp đến chi phí self-attention: \text{Complexity} = \mathcal{O}(T^2 d) Nếu tokenization kém → T lớn → chi phí tăng. Do đó, tokenizer tối ưu giúp: * Giảm memory footprint * Tăng tốc inference * Cải thiện chất lượng ngữ nghĩa --- # 10. Liên hệ thực tế trong GPT Các mô hình GPT sử dụng biến thể của BPE hoặc byte-level BPE. Xác suất sinh token: P(w_t  \mid  w_{\lt t}) = \frac{\exp(z_t W_{out})} {\sum_j \exp(z_j W_{out})} Chất lượng tokenization ảnh hưởng trực tiếp đến phân phối logits. --- # 11. Thảo luận Chuẩn bị văn bản không chỉ là bước tiền xử lý kỹ thuật mà còn là: * Bài toán tối ưu thông tin * Bài toán mã hóa nguồn (source coding) * Bài toán nén dữ liệu Theo định lý Shannon: H(X) = - \sum_x P(x) \log P(x) Tokenizer tốt giúp: \text{Length}(X_{tokens}) \approx \frac{H(X)}{\log |V|} --- # 12. Kết luận Quy trình chuẩn bị văn bản cho tokenization bao gồm: 1. Chuẩn hóa Unicode 2. Làm sạch dữ liệu 3. Áp dụng BPE hoặc Unigram LM 4. Ánh xạ sang embedding Toán học cho thấy tokenization là quá trình:
$$

\Sigma^{\ast} \rightarrow V^{\ast}