
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [04 buildgpt](index.md)

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
# Phân Tích Kiến Trúc và Cơ Chế Hoạt Động của Mô Hình Ngôn Ngữ Transformer Cơ Bản

## Tóm tắt

Bài viết này trình bày phân tích chi tiết về kiến trúc và cơ chế hoạt động của mô hình ngôn ngữ transformer cơ bản (zero-layer transformer), tập trung vào các khía cạnh kỹ thuật của quá trình embedding, sinh token, và luồng xử lý thông tin ngữ cảnh. Nghiên cứu làm rõ các nguyên lý nền tảng trong việc xây dựng mô hình GPT-2 từ đầu, với mục tiêu sư phạm là tạo nền tảng hiểu biết sâu sắc về kiến trúc transformer.

---

## 1. Giới Thiệu

### 1.1 Bối Cảnh Nghiên Cứu

Mô hình ngôn ngữ dựa trên kiến trúc transformer đã cách mạng hóa lĩnh vực xử lý ngôn ngữ tự nhiên. Tuy nhiên, việc hiểu sâu sắc về cơ chế hoạt động bên trong của các mô hình này đòi hỏi phương pháp tiếp cận từ cơ bản đến phức tạp.

### 1.2 Mục Tiêu Nghiên Cứu

Nghiên cứu này nhằm:
- Phân tích kiến trúc mô hình transformer đơn giản nhất (Model 1)
- Làm rõ cơ chế xử lý ngữ cảnh và sinh token
- Khám phá vai trò của các thành phần kỹ thuật như embedding, softmax, và multinomial sampling

---

## 2. Kiến Trúc Mô Hình

### 2.1 Cấu Trúc Tổng Thể

Mô hình cơ bản được nghiên cứu bao gồm các thành phần chính:

**Luồng xử lý dữ liệu:**
Text → Tokens → Embeddings → Non-linearity → Unembeddings → Tokens → Text

**Đặc điểm kỹ thuật:**
- **Embedding dimension**: 64 (kích thước nhỏ gọn cho mục đích giáo dục)
- **Sequence length**: 8 tokens (cho phép quan sát chi tiết)
- **Vocabulary size**: 100,000 tokens (GPT-4 tokenizer)
- **Batch size**: 5

### 2.2 Mô Hình "Zero-Layer Transformer"

Mô hình này được gọi là "zero-layer transformer" vì thiếu các khối transformer trung gian. Kiến trúc bao gồm:

**Thành phần chính:**
1. **Lớp Embedding** (`nn.Embedding`): Chuyển đổi token indices thành vector số thực
2. **Hàm kích hoạt phi tuyến** (GELU): Áp dụng biến đổi phi tuyến
3. **Lớp Unembedding** (`nn.Linear`): Ánh xạ ngược về không gian vocabulary

**Cài đặt PyTorch:**
```python
class Model1(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

$$

$$

self.embeddings = nn.Embedding(vocab_size, embed_dim)

$$

$$

$$
self.gelu = nn.GELU()
$$

$$

$$

self.final_layer = nn.Linear(embed_dim, vocab_size)

$$

$$

    
    def forward(self, tokens):

$$

$$

x = self.embeddings(tokens)

$$

$$

$$
x = self.gelu(x)
$$

$$

$$

logits = self.final_layer(x)

$$

$$

        return logits

---

## 3. Cơ Chế Xử Lý Ngữ Cảnh

### 3.1 Hiểu Lầm Phổ Biến về Token Cuối Cùng

**Phát biểu thường gặp:** "Mô hình chỉ sử dụng token cuối cùng để dự đoán token tiếp theo."

**Thực tế:** Đây là một hiểu lầm nghiêm trọng. Mặc dù mô hình trích xuất token cuối cùng để sinh token mới, nhưng thông tin trong token cuối cùng đã được tích hợp từ tất cả các token trước đó.

### 3.2 Phân Tích Tích Lũy Ngữ Cảnh

**Ví dụ minh họa:** "I prefer oat milk in ___"

| Giai đoạn | Tokens đã xử lý | Không gian tìm kiếm | Cơ chế |
|-----------|----------------|---------------------|---------|
| 1 | "I" | ~10,000+ khả năng | Chỉ dựa vào embedding của "I" |
| 2 | "I prefer" | ~3,000 khả năng | Kết hợp "I" + "prefer" |
| 3 | "I prefer oat" | ~500 khả năng | Tích hợp ngữ cảnh từ 3 tokens |
| 4 | "I prefer oat milk" | ~100 khả năng | Ngữ cảnh đầy đủ hơn |
| 5 | "I prefer oat milk in" | ~50 khả năng | Tối đa hóa thông tin ngữ cảnh |

**Nguyên lý cốt lõi:**
> Mỗi vector embedding không phải là vector ban đầu, mà đã được biến đổi để chứa thông tin về các token trước đó. Việc sử dụng chỉ token cuối cùng để dự đoán token mới thực chất có nghĩa là sử dụng tất cả các tokens—chúng ta chỉ tập trung vào token có nhiều thông tin ngữ cảnh nhất.

### 3.3 Cơ Chế Biến Đổi Vector

Trong quá trình feedforward, các vector embedding trải qua:
- **Scaling** (co giãn tỉ lệ)
- **Addition/Subtraction** (cộng/trừ)
- **Non-linear transformations** (biến đổi phi tuyến)

Kết quả: Token cuối cùng mang thông tin tổng hợp từ toàn bộ chuỗi.

---

## 4. Quy Trình Sinh Token

### 4.1 Luồng Xử Lý Feedforward

**Input:** Vector của N tokens (ví dụ: 10 tokens)
**Output:** Tensor kích thước [N × V] (N tokens × V vocab size)

**Các bước:**
1. Tokenization: Text → Token indices
2. Embedding: Indices → Dense vectors
3. Processing: Phi tuyến hóa
4. Unembedding: Vectors → Logits space
5. Output: Logits matrix [tokens × vocab_size]

### 4.2 Chuyển Đổi Logits sang Xác Suất

**Hàm Softmax:**

$$

$$

\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{V} e^{x_j}}

$$

$$

**Đặc điểm:**
- Chuyển đổi logits (có thể âm, >1) thành xác suất (0 ≤ p ≤ 1)
- Tổng tất cả xác suất = 1
- Phi tuyến: Tăng cường logits cao, giảm logits thấp

**Quan sát thực nghiệm:**
```python
# Logits: [-2.3, 0.5, 3.1, -1.2, ...]
# Sau softmax: [0.001, 0.02, 0.85, 0.003, ...]
# Sum = 1.0

### 4.3 Lựa Chọn Token: Multinomial Sampling

**Phương pháp `torch.multinomial`:**

Không phải lựa chọn ngẫu nhiên đều, mà là **lấy mẫu xác suất** dựa trên phân phối:

- Token có xác suất cao → Khả năng được chọn cao hơn
- Token có xác suất thấp → Vẫn có cơ hội được chọn (tạo đa dạng)

**Ý nghĩa:**
- Giải thích tại sao ChatGPT cho câu trả lời khác nhau với cùng câu hỏi
- Cân bằng giữa chất lượng và sáng tạo
- Tránh tính quyết định cứng nhắc

### 4.4 Thuật Toán Sinh Token Tự Hồi Quy

```python

$$

$$

def generate(self, tokens, n_new_tokens=30):

$$

$$

    for _ in range(n_new_tokens):
        # Bước 1: Feedforward

$$

$$

x = self(tokens)  # [batch, seq_len, vocab_size]

$$

$$

        
        # Bước 2: Trích xuất token cuối cùng

$$

$$

final_logits = x[:, -1, :]  # [batch, vocab_size]

$$

$$

        
        # Bước 3: Softmax

$$

$$

probs = torch.softmax(final_logits, dim=-1)

$$

$$

        
        # Bước 4: Lấy mẫu

$$

$$

next_token = torch.multinomial(probs, num_samples=1)

$$

$$

        
        # Bước 5: Nối token mới

$$

$$

tokens = torch.cat([tokens, next_token], dim=1)

$$

$$

    
    return tokens

**Đặc điểm auto-regressive:**
- Mỗi token mới phụ thuộc vào tất cả tokens trước đó
- Chuỗi tokens tăng dần: N → N+1 → N+2 → ... → N+M

---

## 5. Các Khía Cạnh Kỹ Thuật

### 5.1 Batch Processing

**Tại sao sử dụng batches?**

1. **Hiệu quả tính toán:**
   - Xử lý song song nhiều sequences
   - Tận dụng tối ưu GPU/TPU
   - Giảm thời gian huấn luyện

2. **Regularization:**
   - Loss được tính trung bình trên batch
   - Giảm overfitting
   - Làm mượt quá trình học

**Cấu trúc tensor:**

$$

$$

Input:  [batch_size, seq_len]           = [5, 8]

$$

$$

$$
Output: [batch_size, seq_len, vocab]    = [5, 8, 100000]
$$

$$
### 5.2 So Sánh `nn.Embedding` vs `nn.Linear` | Khía cạnh | nn.Embedding | nn.Linear | |-----------|--------------|-----------| | Mục đích | Tra cứu vector từ bảng | Phép biến đổi tuyến tính | | Input | Token indices (integers) | Dense vectors (floats) | | Cơ chế | Lookup operation | Matrix multiplication | | Gradient | Chỉ update vectors được dùng | Update toàn bộ ma trận | | Hiệu quả | Tối ưu cho sparse inputs | Tối ưu cho dense inputs | **Điểm chung:** Về bản chất, cả hai đều thực hiện phép nhân ma trận, nhưng với interface và tối ưu hóa khác nhau. ### 5.3 Hàm Kích Hoạt GELU vs ReLU **GELU (Gaussian Error Linear Unit):**
$$

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

$$
Trong đó Φx là hàm phân phối chuẩn tích lũy. **Ưu điểm của GELU trong LLMs:** - Mượt hơn ReLU (khả vi tại mọi điểm) - Cho phép giá trị âm có trọng số - Hiệu suất tốt hơn trong các mô hình lớn - Được sử dụng trong GPT-2, BERT, và hầu hết LLMs hiện đại **So sánh:**
$$

ReLU(x) = max(0, x)           # Cứng, không mượt

$$

$$

GELU(x) = x * Φ(x)            # Mượt, xác suất

$$
--- ## 6. Tổ Chức Dữ Liệu và Tiền Xử Lý ### 6.1 Tokenization và Tensor Conversion **Quy trình:** ```python # Bước 1: Text → Token list
$$

$$
tokens_list = tokenizer.encode(text)  # List[int]
$$

$$
# Bước 2: List → PyTorch Tensor
$$

$$
tokens_tensor = torch.tensor(tokens_list)  # Tensor
$$

$$
# Lý do: PyTorch functions yêu cầu tensor inputs ### 6.2 Sequence Dataset Organization **Cấu trúc dữ liệu:** - **Inputs:** Chuỗi N tokens - **Targets:** Cùng chuỗi đó, shift 1 vị trí **Ví dụ:** Inputs:  [47, 45, 38, 439, 12, 89, 234, 56] Targets: [45, 38, 439, 12, 89, 234, 56, 77] **Mục đích:** Mỗi token input học dự đoán token tiếp theo (next-token prediction). ### 6.3 Hyperparameters ```python
$$

HYPERPARAMETERS = {

$$
'vocab_size': 100000,      # GPT-4 tokenizer 'embed_dim': 64,           # Nhỏ cho mục đích giáo dục 'seq_length': 8,           # Cho phép in toàn bộ sequence 'batch_size': 5,           # Xử lý song song 'stride': 1,               # Dữ liệu overlap } **Lưu ý:** Các giá trị này được chọn để tối ưu hóa việc học và hiểu, không phải cho hiệu suất thực tế. --- ## 7. Phân Tích Kết Quả Thực Nghiệm ### 7.1 Quan Sát Logits và Probabilities **Đặc điểm Logits:** - Phạm vi: (-∞, +∞) - Có thể âm, dương, >1 - Không chuẩn hóa - Phản ánh "raw preferences" của model **Sau Softmax:** - Phạm vi: (0, 1) - Tổng = 1 - Phân phối xác suất hợp lệ - Phi tuyến: Tăng khoảng cách giữa high/low logits ### 7.2 Tính Ngẫu Nhiên và Đa Dạng **Thực nghiệm:** Sinh 5 sequences từ cùng input **Kết quả:** Run 1: "The one I had seen above ground in [random tokens...]" Run 2: "The one I had seen above ground in [different random tokens...]" Run 3: "The one I had seen above ground in [yet different tokens...]" ... **Phân tích:** - Cùng prefix → Khác suffix - Multinomial sampling → Stochastic generation - Giải thích behavior của production LLMs (ChatGPT, Claude, etc.) ### 7.3 Weights Chưa Huấn Luyện **Trạng thái hiện tại:** - Weights ngẫu nhiên (random initialization) - Không có pre-training - Output là "gibberish" (vô nghĩa) **Mục đích:** - Tập trung vào **kiến trúc** và **cơ chế** - Hiểu **data flow** và **token generation** - Nền tảng cho training trong sections tiếp theo --- ## 8. Ý Nghĩa Sư Phạm và Phương Pháp Luận ### 8.1 Chiến Lược Học Tập Tăng Dần Cách tiếp cận "5 models" trong khóa học: 1. **Model 1** (hiện tại): Zero-layer transformer 2. **Model 2-5** (sắp tới): Thêm dần components - Attention mechanisms - Multi-head attention - Feed-forward networks - Layer normalization - Residual connections
$$

$$
**Nguyên tắc:** Mỗi model = Previous model + New components
$$

$$
### 8.2 Phương Pháp Giảng Dạy **Không chỉ là code:** - **Experimentation:** Thay đổi hyperparameters - **Problem-solving:** Debug và fix errors - **Code exploration:** Hiểu mọi dòng code - **Exercise completion:** Thực hành chủ động **Sắp tới:** - Các videos chuyên sâu về từng component - Unpacking technical details - Comparative analysis (GELU vs ReLU, Embedding vs Linear, etc.) --- ## 9. Kết Luận ### 9.1 Tóm Tắt Các Phát Hiện Chính 1. **Xử lý ngữ cảnh:** - Token cuối cùng chứa thông tin tổng hợp từ toàn bộ sequence - Không phải chỉ sử dụng một token mà là tất cả tokens 2. **Sinh token:** - Quy trình auto-regressive với multinomial sampling - Cân bằng giữa quality và diversity 3. **Kiến trúc:** - Zero-layer transformer là nền tảng đơn giản nhất - Chuẩn bị cho việc xây dựng full GPT-2 ### 9.2 Hướng Phát Triển **Các bước tiếp theo:** - Thêm transformer blocks - Implement attention mechanisms - Incorporate positional encodings - Add normalization layers - Implement training procedures ### 9.3 Đóng Góp Học Thuật Nghiên cứu này cung cấp: - Framework sư phạm cho việc giảng dạy LLM architecture - Phân tích chi tiết về information flow trong transformers - Làm rõ các hiểu lầm phổ biến về context processing - Methodology cho việc xây dựng mô hình từ cơ bản đến phức tạp --- ## Tài Liệu Tham Khảo 1. Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS. 2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." OpenAI. 3. Hendrycks, D., & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)." arXiv. --- ## Phụ Lục: Code Reference ### A.1 Model Definition ```python class Model1(nn.Module):
$$

$$
def __init__(self, vocab_size=100000, embed_dim=64):
$$

$$
super().__init__()
$$

$$
self.embeddings = nn.Embedding(vocab_size, embed_dim)
$$

$$

$$

self.gelu = nn.GELU()

$$

$$

$$
self.final_layer = nn.Linear(embed_dim, vocab_size)
$$

$$
### A.2 Generation Method ```python
$$

$$
def generate(self, tokens, n_new_tokens=30):
$$

$$
for _ in range(n_new_tokens):
$$

$$
x = self(tokens)
$$

$$

$$

final_logits = x[:, -1, :]

$$

$$

$$
probs = torch.softmax(final_logits, dim=-1)
$$

$$

$$

next_token = torch.multinomial(probs, num_samples=1)

$$

$$

$$
tokens = torch.cat([tokens, next_token], dim=1)
$$
