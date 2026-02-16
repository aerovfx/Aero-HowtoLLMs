# Biểu diễn Tính Nhân Quả Thời Gian trong Cơ Chế Attention bằng Đại Số Tuyến Tính

## Tóm tắt (Abstract)

Bài báo này trình bày phân tích lý thuyết về cơ chế nhân quả thời gian (temporal causality) trong mô hình Transformer, đặc biệt trong kiến trúc GPT, thông qua góc nhìn đại số tuyến tính. Dựa trên tài liệu giảng dạy về causal attention mask :contentReference[oaicite:0]{index=0}, nghiên cứu làm rõ vai trò của ma trận mặt nạ (mask matrix), hàm softmax, và cách chúng đảm bảo mô hình chỉ khai thác thông tin từ quá khứ khi dự đoán tương lai. Kết quả cho thấy causal masking là yếu tố cốt lõi giúp mô hình ngôn ngữ sinh văn bản một cách hợp lệ và ổn định về mặt số học.

---

## 1. Giới thiệu (Introduction)

Các mô hình ngôn ngữ hiện đại như GPT và BERT đều dựa trên kiến trúc Transformer với cơ chế self-attention. Tuy nhiên, sự khác biệt cốt lõi giữa hai dòng mô hình này nằm ở việc có hay không áp dụng ràng buộc nhân quả thời gian.

Theo tài liệu lý thuyết về causal attention :contentReference[oaicite:1]{index=1}, GPT sử dụng mặt nạ nhân quả để ngăn mô hình truy cập thông tin trong tương lai, trong khi BERT cho phép truy cập toàn bộ ngữ cảnh.

Mục tiêu của bài báo này là:

- Phân tích cơ sở toán học của causal masking,
- Làm rõ vai trò của softmax trong việc đảm bảo tính ổn định,
- So sánh cơ chế nhân quả trong GPT và BERT,
- Đánh giá tác động đến khả năng sinh văn bản.

---

## 2. Cơ sở lý thuyết (Theoretical Background)

### 2.1. Attention trong Transformer

Cơ chế attention tiêu chuẩn được định nghĩa:

\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

trong đó:

- \(Q\): Query matrix  
- \(K\): Key matrix  
- \(V\): Value matrix  
- \(d\): số chiều ẩn  

Kết quả attention là tổ hợp tuyến tính của các vector giá trị dựa trên mức độ liên quan.

---

### 2.2. Biểu diễn Nhân quả Thời gian

Trong dự đoán chuỗi, tại thời điểm \(t\), mô hình chỉ được phép sử dụng thông tin từ:

\[
\{1,2,...,t\}
\]

và không được truy cập:

\[
\{t+1, t+2, ...\}
\]

Nguyên tắc này phản ánh thực tế rằng tương lai chưa xảy ra và không thể được biết trước.

---

### 2.3. Vector Trọng số Thời gian

Một cách trực quan, sự tích hợp thông tin quá khứ có thể biểu diễn bằng vector:

\[
a = (a_1, a_2, ..., a_T)
\]

với:

- \(a_i > 0\) nếu \(i \leq t\),
- \(a_i = 0\) nếu \(i > t\).

Tuy nhiên, vector này chưa được chuẩn hóa và không phù hợp cho tính toán số học ổn định.

---

## 3. Softmax và Vấn đề Truy cập Tương lai

### 3.1. Hiệu ứng của Softmax

Softmax được định nghĩa:

\[
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
\]

Nếu một phần tử có giá trị bằng 0:

\[
e^0 = 1 \neq 0
\]

Do đó, việc gán giá trị 0 cho tương lai không đảm bảo xác suất bằng 0 sau softmax.

---

### 3.2. Giải pháp: Giá trị Âm Vô Cùng

Theo tài liệu tham khảo :contentReference[oaicite:2]{index=2}, để đảm bảo xác suất bằng 0, ta đặt:

\[
x_i = -\infty \quad \text{với } i > t
\]

vì:

\[
\lim_{x \to -\infty} e^x = 0
\]

Do đó:

\[
\text{softmax}(-\infty) = 0
\]

Giải pháp này đảm bảo tương lai hoàn toàn bị loại bỏ.

---

### 3.3. Lợi ích Số học

Cách tiếp cận này mang lại:

- Tính ổn định số,
- Tránh tràn số,
- Tạo phân phối xác suất hợp lệ,
- Tăng tính thưa (sparsity).

---

## 4. Ma trận Nhân quả (Causal Mask Matrix)

### 4.1. Cấu trúc Ma trận

Thay vì vector riêng lẻ, causal attention được biểu diễn bằng ma trận:

\[
M \in \mathbb{R}^{T \times T}
\]

với:

\[
M_{ij} =
\begin{cases}
0 & \text{nếu } j \le i \\
-\infty & \text{nếu } j > i
\end{cases}
\]

Ma trận này có dạng tam giác dưới.

---

### 4.2. Tích hợp vào Attention

Công thức attention mở rộng:

\[
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d}} + M
\right)V
\]

Trong đó \(M\) đóng vai trò loại bỏ tương tác với tương lai.

---

### 4.3. Softmax Theo Hàng

Việc softmax được áp dụng theo từng hàng:

\[
\text{softmax}(M_i)
\]

đảm bảo mỗi token chỉ quan tâm đến quá khứ của chính nó.

---

## 5. Vai trò của Softmax trong Causal Attention

Theo phân tích từ :contentReference[oaicite:3]{index=3}, softmax mang lại hai lợi ích chính:

### 5.1. Xử lý Giá trị Âm

Các giá trị attention có thể âm do là tích vô hướng. Softmax:

- Biến đổi thành xác suất không âm,
- Chuẩn hóa về tổng bằng 1.

---

### 5.2. Tăng Tính Chọn Lọc

Softmax khuếch đại giá trị lớn và làm suy giảm giá trị nhỏ, dẫn đến:

- Tập trung vào token quan trọng,
- Giảm nhiễu,
- Cải thiện khả năng suy luận.

Điều này giúp mô hình dần thu hẹp không gian tìm kiếm khi sinh văn bản.

---

## 6. GPT và BERT: So sánh Cơ chế Nhân quả

### 6.1. Mô hình GPT (Decoder-based)

Đặc trưng:

- Có causal mask,
- Huấn luyện autoregressive,
- Phù hợp cho sinh văn bản.

GPT tuân thủ chặt chẽ nguyên lý nhân quả.

---

### 6.2. Mô hình BERT (Encoder-based)

Đặc trưng:

- Không dùng causal mask,
- Attention hai chiều,
- Dùng cho phân loại, tóm tắt, phân tích.

BERT khai thác toàn bộ ngữ cảnh để tối ưu biểu diễn.

---

### 6.3. So sánh

| Tiêu chí | GPT | BERT |
|----------|-----|------|
| Causal Mask | Có | Không |
| Sinh văn bản | Tốt | Hạn chế |
| Phân tích văn bản | Trung bình | Tốt |
| Truy cập tương lai | Không | Có |

---

## 7. Kết quả và Ứng dụng (Results and Applications)

### 7.1. Hiệu quả Huấn luyện

Causal masking giúp:

- Đồng bộ quá trình train và inference,
- Tránh leakage thông tin,
- Ổn định gradient.

---

### 7.2. Ứng dụng Thực tế

Cơ chế này được áp dụng trong:

- Chatbot,
- Trình sinh văn bản,
- Hệ thống dịch máy,
- Hệ thống viết tự động.

---

### 7.3. Khả năng Mở rộng

Causal masking cho phép:

- Huấn luyện song song,
- Duy trì tính tuần tự khi suy luận,
- Kết hợp với KV-cache.

---

## 8. Thảo luận (Discussion)

### 8.1. Góc nhìn Đại số Tuyến tính

Causal attention có thể xem là:

- Phép nhân ma trận có ràng buộc tam giác,
- Phép chiếu không gian thông tin vào miền quá khứ.

Điều này cho phép phân tích bằng lý thuyết phổ và chuẩn ma trận.

---

### 8.2. Hạn chế

Một số hạn chế:

- Không tận dụng được thông tin tương lai khi train,
- Giảm hiệu quả cho tác vụ hiểu văn bản,
- Phụ thuộc độ dài ngữ cảnh.

---

### 8.3. Mở rộng

Các hướng phát triển:

- Mask mềm (soft mask),
- ALiBi,
- Rotary Embedding,
- Hybrid encoder-decoder.

---

## 9. Hạn chế nghiên cứu (Limitations)

Nghiên cứu này:

- Chỉ tập trung vào lý thuyết,
- Không có thực nghiệm quy mô lớn,
- Chưa đánh giá trên benchmark chuẩn.

Do đó, kết luận mang tính phân tích nền tảng.

---

## 10. Kết luận (Conclusion)

Bài báo đã phân tích cơ chế causal attention thông qua:

- Softmax,
- Ma trận mask,
- Giới hạn âm vô cùng,
- Đại số tuyến tính.

Kết quả cho thấy causal masking là nền tảng toán học cốt lõi giúp GPT duy trì tính nhân quả và khả năng sinh văn bản đáng tin cậy.

Cơ chế này phân biệt rõ ràng giữa mô hình sinh (GPT) và mô hình hiểu (BERT), đồng thời là trụ cột của các LLM hiện đại.

---

## 11. Hướng nghiên cứu tiếp theo (Future Work)

Các hướng phát triển gồm:

- Causal attention thích nghi,
- Mask học được,
- Attention thưa,
- Causal attention đa chiều,
- Kết hợp RAG và causal modeling.

---

## Tài liệu tham khảo (References)

1. Bài giảng về Temporal Causality và Causal Attention trong Transformer. :contentReference[oaicite:4]{index=4}

Dưới đây là phần **Pseudocode + PyTorch Implementation cho Causal Mask** được viết theo **chuẩn bài báo khoa học**, phù hợp để đưa vào:

✅ Appendix / Supplementary Material
✅ Phần Methodology
✅ Luận văn / Journal Paper

Trình bày theo phong cách học thuật và dễ tái lập.

---

````md
# Phụ lục B: Causal Mask – Thuật toán và Triển khai

## B.1. Tổng quan

Trong mô hình Transformer dạng autoregressive, causal mask được sử dụng để đảm bảo rằng tại thời điểm \( t \), mô hình chỉ truy cập được các token trong quá khứ và hiện tại, không truy cập được token trong tương lai.

Phần này trình bày:

- Mã giả (pseudocode) cho causal mask,
- Cách tích hợp vào attention,
- Triển khai bằng PyTorch.

---

## B.2. Pseudocode: Tạo Ma trận Causal Mask

### Thuật toán 4: Sinh Ma trận Nhân quả

**Input:**

- Độ dài chuỗi: \( T \)

**Output:**

- Ma trận mask: \( M \in \mathbb{R}^{T \times T} \)

---

```text
Algorithm 4: Generate-Causal-Mask(T)

1:  Initialize M as matrix of size T × T

2:  for i = 1 → T do
3:      for j = 1 → T do
4:          if j ≤ i then
5:              M[i, j] ← 0
6:          else
7:              M[i, j] ← -∞
8:          end if
9:      end for
10: end for

11: return M
````

---

### Giải thích

* Phần tử ( M_{ij} = 0 ): cho phép attention,
* Phần tử ( M_{ij} = -\infty ): chặn attention,
* Dạng tam giác dưới đảm bảo tính nhân quả.

---

## B.3. Pseudocode: Attention với Causal Mask

### Thuật toán 5: Causal Self-Attention

**Input:**

* Query: ( Q \in \mathbb{R}^{T \times d} )
* Key: ( K \in \mathbb{R}^{T \times d} )
* Value: ( V \in \mathbb{R}^{T \times d} )
* Mask: ( M \in \mathbb{R}^{T \times T} )

**Output:**

* Output: ( O \in \mathbb{R}^{T \times d} )

---

```text
Algorithm 5: Causal-Attention(Q, K, V, M)

1:  S ← Q × Kᵀ
2:  S ← S / sqrt(d)

3:  S ← S + M

4:  A ← softmax(S)

5:  O ← A × V

6:  return O
```

---

### Giải thích

* Bước (3) đảm bảo tương lai bị loại bỏ,
* Softmax biến mask thành xác suất bằng 0,
* Attention chỉ tập trung vào quá khứ.

---

## B.4. Triển khai PyTorch: Causal Mask Cơ bản

### B.4.1. Tạo Mask Tam giác

```python
import torch
```

---

```python
def generate_causal_mask(T, device=None):
    """
    Generate causal attention mask.

    Args:
        T (int): Sequence length
        device (torch.device): Target device

    Returns:
        mask (Tensor): (T, T) boolean mask
    """

    mask = torch.triu(
        torch.ones(T, T),
        diagonal=1
    )

    if device is not None:
        mask = mask.to(device)

    return mask.bool()
```

---

### Dạng Kết quả

Ví dụ với `T = 4`:

```text
0 1 1 1
0 0 1 1
0 0 0 1
0 0 0 0
```

Trong đó:

* `1` = bị chặn,
* `0` = cho phép.

---

## B.5. Causal Mask với Giá trị -∞ (Logit Mask)

Trong thực tế, mask thường được biểu diễn bằng giá trị âm lớn.

---

### B.5.1. Mask dạng Float

```python
def generate_causal_logit_mask(T, device=None):
    """
    Generate causal mask with -inf values.
    """

    mask = torch.triu(
        torch.ones(T, T),
        diagonal=1
    )

    mask = mask.masked_fill(
        mask == 1,
        float("-inf")
    )

    if device is not None:
        mask = mask.to(device)

    return mask
```

---

### Công dụng

Dùng trực tiếp cho:

```python
scores = scores + mask
```

---

## B.6. Tích hợp vào Multi-Head Attention

---

### B.6.1. Attention Layer với Mask

```python
class CausalAttention(torch.nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(
            d_model,
            num_heads,
            batch_first=True
        )

    def forward(self, x):

        B, T, _ = x.shape

        mask = generate_causal_mask(
            T, x.device
        )

        out, weights = self.attn(
            x, x, x,
            attn_mask=mask
        )

        return out, weights
```

---

### Lưu ý

* `attn_mask=True` → bị chặn,
* `attn_mask=False` → cho phép.

---

## B.7. Causal Mask cho Batch và KV Cache

Trong inference với cache, chỉ cần mask cho token mới.

---

### B.7.1. Mask cho Incremental Decoding

```python
def generate_incremental_mask(
    past_len,
    current_len,
    device
):
    """
    Mask for KV-cache decoding.
    """

    total = past_len + current_len

    mask = torch.triu(
        torch.ones(current_len, total),
        diagonal=1 + past_len
    )

    return mask.bool().to(device)
```

---

### Công dụng

Dùng cho sinh từng token:

```text
Past tokens | New token
```

Chỉ cho phép new token nhìn về quá khứ.

---

## B.8. Ví dụ Hoàn chỉnh

---

### B.8.1. Demo Attention với Mask

```python
def demo():

    B = 2
    T = 5
    D = 64
    H = 4

    x = torch.randn(B, T, D)

    attn = CausalAttention(D, H)

    out, w = attn(x)

    print("Output:", out.shape)
    print("Weights:", w.shape)


if __name__ == "__main__":
    demo()
```

---

### Output

```text
Output:  (2, 5, 64)
Weights: (2, 5, 5)
```

---

## B.9. Độ phức tạp (Complexity Analysis)

Với chuỗi độ dài ( T ):

### Thời gian

[
O(T^2)
]

### Bộ nhớ

[
O(T^2)
]

Khi dùng KV Cache:

[
O(T)
]

---

## B.10. Thảo luận Kỹ thuật (Technical Discussion)

### Ưu điểm

* Đảm bảo tính nhân quả,
* Tránh leakage thông tin,
* Đồng bộ train–inference,
* Dễ cài đặt.

### Hạn chế

* Chi phí bậc hai theo T,
* Không phù hợp chuỗi rất dài,
* Phụ thuộc vào precision.

---

## B.11. Tái lập Thực nghiệm (Reproducibility)

Khuyến nghị cố định:

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Lưu trữ:

* Mask implementation,
* Version PyTorch,
* GPU driver,
* Config.

---

## B.12. Tóm tắt

Phụ lục này đã trình bày:

* Pseudocode tạo causal mask,
* Cách tích hợp vào attention,
* Triển khai PyTorch,
* Phân tích độ phức tạp.

Causal mask là nền tảng toán học và kỹ thuật đảm bảo tính hợp lệ của mô hình ngôn ngữ tự hồi quy.

```

---
