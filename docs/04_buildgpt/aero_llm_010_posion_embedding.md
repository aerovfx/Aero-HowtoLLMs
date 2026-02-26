
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
````md
# Mở rộng Kiến trúc GPT: Position Embedding, Layer Normalization, Weight Tying và Temperature Scaling

## Tóm tắt (Abstract)

Bài báo này phân tích các thành phần quan trọng trong việc mở rộng mô hình GPT cơ bản, bao gồm position embedding, layer normalization, weight tying và temperature scaling. Dựa trên tài liệu giảng dạy về xây dựng mô hình GPT-2 đơn giản , chúng tôi trình bày cơ sở lý thuyết, cơ chế triển khai và tác động thực nghiệm của từng thành phần. Kết quả cho thấy các kỹ thuật này đóng vai trò thiết yếu trong việc ổn định huấn luyện, giảm số tham số và cải thiện chất lượng sinh văn bản.

---

## 1. Giới thiệu (Introduction)

Mô hình Transformer và các biến thể GPT đã trở thành nền tảng cho nhiều hệ thống xử lý ngôn ngữ tự nhiên hiện đại. Một GPT tối thiểu chỉ gồm embedding, MLP và linear output thường không đủ ổn định để huấn luyện và suy luận hiệu quả.

Theo tài liệu xây dựng mô hình GPT-2 dạng học thuật , việc bổ sung position embedding, layer normalization, weight tying và temperature scaling giúp mô hình:

- Nhận biết vị trí từ trong chuỗi,
- Ổn định phân phối kích hoạt,
- Giảm số lượng tham số,
- Kiểm soát tính ngẫu nhiên khi sinh văn bản.

Mục tiêu của bài báo là phân tích có hệ thống các kỹ thuật này trong bối cảnh mô hình ngôn ngữ quy mô nhỏ đến trung bình.

---

## 2. Cơ sở lý thuyết (Theoretical Background)

### 2.1. Token Embedding và Position Embedding

Trong GPT, mỗi token được ánh xạ thành vector thông qua embedding:

$$

$$

E_{tok} \in \mathbb{R}^{V \times d}

$$

$$

với $V$ là kích thước từ vựng, $d$ là chiều embedding.

Position embedding được định nghĩa:

$$

$$

E_{pos} \in \mathbb{R}^{L \times d}

$$

$$

với $L$ là độ dài chuỗi tối đa.

Biểu diễn đầu vào:

$$

$$

X = E_{tok}(w_i) + E_{pos}(i)

$$

$$

Cách cộng trực tiếp này cho phép mô hình học thông tin thứ tự mà không cần kiến trúc hồi quy.

---

### 2.2. Layer Normalization

Layer normalization chuẩn hóa theo chiều embedding:

$$

$$

\hat{x} = \frac{x - \mu}{\sigma + \epsilon}

$$

$$

$$
y = \gamma \hat{x} + \beta
$$

$$
Trong đó \mu, \sigma được tính theo từng token. Tác dụng chính: - Giảm hiện tượng exploding/vanishing gradients, - Ổn định phân phối kích hoạt, - Tăng tốc hội tụ. --- ### 2.3. Weight Tying (Tied Embeddings) Weight tying ràng buộc:
$$

$$
W_{out} = E_{tok}^T
$$

$$
Trong đó W_{out} là ma trận unembedding. Ưu điểm: - Giảm ~30–40% số tham số, - Tăng tính nhất quán giữa biểu diễn và dự đoán, - Giảm overfitting. --- ### 2.4. Logit Scaling và Temperature #### Logit Scaling Logits cuối cùng được chuẩn hóa:
$$

$$
z' = \frac{z}{\sqrt{d}}
$$

$$
Mục đích: giữ phương sai logits ở mức ổn định, phù hợp với giả thuyết lý thuyết. #### Temperature Scaling Trong suy luận:
$$

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

$$
- T < 1: sinh văn bản quyết định hơn, - T > 1: sinh văn bản đa dạng hơn. --- ## 3. Phương pháp nghiên cứu (Methodology) ### 3.1. Kiến trúc mô hình Mô hình được xây dựng từ GPT-1 cơ bản và mở rộng theo tài liệu tham khảo : 1. Token Embedding 2. Position Embedding 3. LayerNorm 4. MLP + Activation (GELU) 5. Linear Output (Weight Tying) 6. Logit Scaling Sơ đồ tổng quát: ```text Input Tokens ↓ Token Embedding + Position Embedding ↓ LayerNorm ↓ MLP (GELU) ↓ Tied Linear Output ↓ Scaled Logits ```` --- ### 3.2. Quy trình huấn luyện và đánh giá * Khởi tạo tham số ngẫu nhiên (Gaussian/Xavier), * Không huấn luyện đầy đủ, tập trung vào phân tích thống kê, * So sánh loss thực nghiệm và loss lý thuyết. Loss lý thuyết của mô hình ngẫu nhiên:
$$

$$
\mathcal{L}_{theory} = \log(V)
$$

$$
với (V) là vocab size. --- ### 3.3. Sinh văn bản Thuật toán sinh: 1. Lấy logits cuối, 2. Chia cho temperature, 3. Softmax, 4. Multinomial sampling, 5. Lặp autoregressive. Chỉ sử dụng cửa sổ ngữ cảnh gần nhất (sliding window) để giới hạn bộ nhớ. --- ## 4. Kết quả (Results) ### 4.1. Phân phối Loss Khi khởi tạo ngẫu nhiên, loss thực nghiệm:
$$

* Gần bằng (\log(V)),

$$
* Phù hợp với dự đoán lý thuyết. Điều này xác nhận mô hình được cài đặt đúng. --- ### 4.2. Phân phối Xác suất Softmax đầu ra thể hiện: * Phân phối thưa (sparse), * Một số token có xác suất nổi trội, * Phần lớn token có xác suất rất nhỏ (~1/V). Khi thay đổi temperature: | Temperature | Độ đa dạng | Độ mạch lạc | | ----------- | ---------- | ----------- | | 0.5         | Thấp       | Cao         | | 1.0         | Trung bình | Trung bình  | | 1.3         | Cao        | Thấp hơn    | --- ### 4.3. Chất lượng Sinh văn bản Mô hình chưa huấn luyện: * Tạo chuỗi có cấu trúc ngắn hạn, * Nhanh chóng suy biến sang nhiễu. Điều này phản ánh vai trò cốt lõi của dữ liệu huấn luyện. --- ## 5. Thảo luận (Discussion) ### 5.1. Vai trò của Position Embedding Việc cộng trực tiếp embedding vị trí: * Đơn giản, * Hiệu quả, * Không làm tăng độ phức tạp tính toán. Tuy nhiên, hạn chế là không ngoại suy tốt cho chuỗi dài hơn L. --- ### 5.2. Tác động của LayerNorm LayerNorm đóng vai trò then chốt trong: * Ổn định forward pass, * Cho phép huấn luyện sâu, * Giảm phụ thuộc vào learning rate. Thiếu LayerNorm → huấn luyện không ổn định. --- ### 5.3. Lợi ích của Weight Tying Weight tying: * Giảm chi phí huấn luyện, * Phù hợp với mô hình nhỏ/trung bình. Tuy nhiên, với mô hình cực lớn, untying có thể tăng tính biểu diễn. --- ### 5.4. Temperature và Điều khiển Hành vi Sinh Temperature cho phép: * Điều chỉnh phong cách sinh, * Cân bằng sáng tạo và chính xác. Trong hệ thống chatbot thực tế, temperature thường nằm trong [0.7, 1.0]. --- ## 6. Hạn chế (Limitations) Nghiên cứu hiện tại có một số hạn chế: * Không huấn luyện trên tập dữ liệu lớn, * Chưa tích hợp attention nhiều tầng, * Chưa đánh giá trên benchmark chuẩn. Do đó, kết quả chủ yếu mang tính minh họa kiến trúc. --- ## 7. Kết luận (Conclusion) Bài báo đã phân tích chi tiết các thành phần mở rộng của GPT gồm: * Position embedding, * Layer normalization, * Weight tying, * Logit và temperature scaling. Các kỹ thuật này: * Ổn định huấn luyện, * Giảm tham số, * Cải thiện kiểm soát suy luận. Chúng là nền tảng cho việc xây dựng LLM hiện đại. --- ## 8. Hướng nghiên cứu tiếp theo (Future Work) Các hướng mở rộng: * Rotary/ALiBi position embedding, * Pre-LN Transformer, * RMSNorm, * Adaptive temperature, * Scaling lên hàng tỷ tham số. --- ## Tài liệu tham khảo (References) 1. Tài liệu xây dựng GPT-2 mở rộng, Position Embedding, LayerNorm, Weight Tying và Temperature Scaling. Dưới đây là phần **Pseudocode + PyTorch Implementation** được viết theo **chuẩn bài báo khoa học**, phù hợp để đưa vào: ✅ Phần Method / Appendix ✅ Supplementary Material ✅ Thesis / Journal Paper Trình bày theo format học thuật. --- ````md # Phụ lục A: Thuật toán và Triển khai (Pseudocode and Implementation) ## A.1. Tổng quan Phần này trình bày thuật toán huấn luyện và suy luận cho mô hình GPT mở rộng với: - Position Embedding - Layer Normalization - Weight Tying - Temperature Scaling Mục tiêu là mô tả rõ ràng quy trình thực thi và khả năng tái lập. --- ## A.2. Pseudocode của Mô hình GPT Mở rộng ### Thuật toán 1: Forward Pass của GPT với Position Embedding và Weight Tying **Input:** - Chuỗi token: x = (x_1, x_2, ..., x_T) - Token embedding: E_{tok} - Position embedding: E_{pos} - MLP parameters: \Theta - Output projection: W_{out} = E_{tok}^T **Output:** - Logits: z --- ```text Algorithm 1: GPT-Forwardx
$$

1:  for i = 1 → T do

$$
2:      e_tok ← E_tok[x_i] 3:      e_pos ← E_pos[i] 4:      h_i ← e_tok + e_pos 5:  end for 6:  H ← LayerNormh 7:  for each layer l do 8:      H ← MLP_lH 9:      H ← LayerNormH 10: end for 11: Z ← H · W_out 12: return Z ```` --- ### Thuật toán 2: Huấn luyện Mô hình **Input:** * Dataset ( D ) * Learning rate ( \eta ) * Batch size ( B ) * Epochs ( E ) --- ```text Algorithm 2: Training(D, η, B, E) 1:  Initialize θ randomly
$$

2:  for epoch = 1 → E do

$$
3:      for batch (x, y) ∈ D do 4:          Z ← GPT-Forwardx 5:          L ← CrossEntropy(Z, y) 6:          Compute ∇θL 7:          θ ← θ − η∇θL 8:      end for 9:  end for --- ### Thuật toán 3: Sinh Văn bản với Temperature **Input:** * Prompt P * Temperature T * Max tokens N --- ```text Algorithm 3: Generate(P, T, N) 1:  x ← TokenizeP(
$$

2:  for t = 1 → N do

$$
3:      Z ← GPT-Forward)x 4:      z_t ← Z_last / T 5:      p ← Softmaxz_t 6:      s ← Samplep 7:      x ← Append(x, s) 8:  end for 9:  return x --- ## A.3. Triển khai PyTorch (PyTorch Implementation) ### A.3.1. Mô hình GPT Mở rộng ```python import torch import torch.nn as nn import math --- ### Token + Position Embedding ```python class GPTEmbedding(nn.Module): def __init__(self, vocab_size, max_len, d_model): super().__init__()
$$

$$
self.token_emb = nn.Embedding(
$$

$$
vocab_size, d_model )
$$

$$
self.pos_emb = nn.Embedding(
$$

$$
max_len, d_model ) def forward(self, x):
$$

$$
B, T = x.shape
$$

$$

$$

pos = torch.arange(

$$

$$

$$
T, device=x.device
$$

$$
)
$$

$$
tok = self.token_embx
$$

$$

$$

pos = self.pos_emb(pos)

$$

$$

        return tok + pos

---

### Feedforward Block

```python
class FeedForward(nn.Module):

    def __init__(self, d_model):
        super().__init__()

$$

$$

self.net = nn.Sequential(

$$

$$

            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net$x$

---

### Transformer Block (Pre-LN)

```python
class GPTBlock(nn.Module):

$$

$$

def __init__(self, d_model, heads, dropout=0.1):

$$

$$

        super().__init__()

$$

$$

self.ln1 = nn.LayerNorm(d_model)

$$

$$

$$
self.ln2 = nn.LayerNorm(d_model)
$$

$$

$$

self.attn = nn.MultiheadAttention(

$$

$$

            d_model,
            heads,

$$

$$

dropout=dropout,

$$

$$

$$
batch_first=True
$$

$$
)
$$

$$
self.ffn = FeedForward(d_model)
$$

$$
def forward(self, x, mask):
$$

$$
h = self.ln1x
$$

$$

$$

attn_out, _ = self.attn(

$$

$$

$$
h, h, h, attn_mask=mask
$$

$$
)
$$

$$
x = x + attn_out
$$

$$

$$

h = self.ln2x

$$

$$

$$
x = x + self.ffnh
$$

$$
return x --- ### GPT Model với Weight Tying ```python class MiniGPT(nn.Module): def __init__( self, vocab_size, max_len, d_model, heads, layers ): super().__init__()
$$

$$
self.embed = GPTEmbedding(
$$

$$
vocab_size, max_len, d_model )
$$

$$
self.blocks = nn.ModuleList([
$$

$$
GPTBlock(d_model, heads) for _ in range(layers) ])
$$

$$
self.ln_f = nn.LayerNorm(d_model)
$$

$$

$$

self.lm_head = nn.Linear(

$$

$$

            d_model,
            vocab_size,

$$

$$

bias=False

$$

$$

        )

        # Weight tying

$$

$$

self.lm_head.weight = \

$$

$$

            self.embed.token_emb.weight

$$

$$

self.max_len = max_len

$$

$$

    def causal_mask(self, T, device):

        return torch.triu(

$$

$$

torch.ones(T, T, device=device),

$$

$$

diagonal=1

$$
).bool() def forward(self, x):
$$

$$
B, T = x.shape
$$

$$

$$

h = self.embedx

$$

$$

$$
mask = self.causal_mask(
$$

$$
T, x.device ) for block in self.blocks:
$$

$$
h = block(h, mask)
$$

$$

$$

h = self.ln_fh

$$

$$

$$
logits = self.lm_headh
$$

$$
return logits --- ## A.3.2. Huấn luyện (Training) ```python def train_step( model, optimizer, loss_fn, x, y ):
$$

$$
logits = modelx
$$

$$

$$

loss = loss_fn(

$$

$$

        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

---

### Training Loop

```python
def train(
    model,
    dataloader,
    epochs,

$$
lr=3e-4,
$$

$$
device="cuda"
$$

):

    model.to(device)

$$

$$

optimizer = torch.optim.AdamW(

$$

$$

        model.parameters(),

$$

$$

lr=lr

$$

$$

    )

$$

$$

loss_fn = nn.CrossEntropyLoss()

$$

$$

    for epoch in range(epochs):

$$
total = 0
$$

        for x, y in dataloader:

$$

$$

x = x.to(device)

$$

$$

$$
y = y.to(device)
$$

$$

$$

loss = train_step(

$$

$$

                model,
                optimizer,
                loss_fn,
                x,
                y
            )

$$
total += loss
$$

        print(
            f"Epoch {epoch}: "

$$
f"Loss = {total/len(dataloader):.4f}"
$$

        )

---

## A.3.3. Sinh Văn bản (Inference + Temperature)

```python
@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,

$$

$$

max_new=200,

$$

$$

temperature=1.0

$$
): model.eval()
$$

$$
device = next(model.parameters()).device
$$

$$

$$

ids = torch.tensor(

$$

$$

        tokenizer.encode(prompt),

$$

$$

device=device

$$

$$

    ).unsqueeze(0)

    for _ in range$max_new$:

$$

$$

logits = model(ids)

$$

$$

$$
next_logits = logits[:, -1]
$$

$$

$$

next_logits /= temperature

$$

$$

$$
probs = torch.softmax(
$$

$$

$$

next_logits, dim=-1

$$

$$

        )

$$

$$

next_id = torch.multinomial(

$$

$$

            probs, 1
        )

$$

$$

ids = torch.cat(

$$

$$

$$
[ids, next_id], dim=1
$$

$$
) return tokenizer.decode( ids[0].tolist() ) --- ## A.4. Độ phức tạp tính toán (Computational Complexity) Với: * Sequence length: T * Hidden size: d * Layers: L Chi phí forward:
$$

O(L \cdot T^2 \cdot d)

$$
Bộ nhớ:
$$

O(L \cdot T \cdot d)

$$
Khi dùng KV-cache:
$$

O(L \cdot T \cdot d)