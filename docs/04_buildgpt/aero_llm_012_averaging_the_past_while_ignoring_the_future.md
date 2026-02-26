
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
# Cơ Chế Trung Bình Hóa Quá Khứ và Loại Bỏ Tương Lai trong Mô Hình Ngôn Ngữ Nhân Quả

## Tóm tắt (Abstract)

Trong các mô hình ngôn ngữ tự hồi quy, việc đảm bảo tính nhân quả (causality) là điều kiện cần thiết để ngăn chặn rò rỉ thông tin từ tương lai. Bài báo này phân tích cơ chế trung bình hóa thông tin quá khứ trong khi loại bỏ thông tin tương lai thông qua causal masking và softmax. Dựa trên minh họa lập trình, nghiên cứu làm rõ vai trò của giá trị âm vô cực trong việc xây dựng phân phối xác suất hợp lệ, đồng thời đánh giá tác động của các chiến lược chuẩn hóa trọng số đến độ ổn định số và khả năng biểu diễn của mô hình.

---

## 1. Giới thiệu (Introduction)

Các mô hình ngôn ngữ hiện đại như Transformer hoạt động dựa trên cơ chế attention, trong đó mỗi token được phép truy cập thông tin từ các token khác trong chuỗi. Tuy nhiên, đối với các bài toán sinh chuỗi tự hồi quy, mô hình không được phép sử dụng thông tin từ tương lai.

Để giải quyết vấn đề này, causal mask được sử dụng nhằm giới hạn phạm vi attention, chỉ cho phép mỗi vị trí truy cập vào quá khứ và hiện tại. Tài liệu nghiên cứu trình bày chi tiết cách hiện thực hóa cơ chế này bằng đại số tuyến tính và lập trình song song. 

---

## 2. Cơ sở lý thuyết (Theoretical Background)

### 2.1. Trung bình hóa thông tin quá khứ

Giả sử tồn tại một vector kích hoạt $x \in \mathbb{R}^T$, biểu diễn thông tin tại các thời điểm trong quá khứ. Một vector trọng số $w \in \mathbb{R}^T$ được sử dụng để tính tổng có trọng số:

$$
y = \sum_{i=1}^{T} w_i x_i
$$

Trong trường hợp đơn giản, $w$ có thể được khởi tạo đồng đều, dẫn đến trung bình cộng của các giá trị quá khứ. Tuy nhiên, cách tiếp cận này không phản ánh mức độ quan trọng khác nhau giữa các thời điểm. 

---

### 2.2. Vai trò của hàm Softmax

Để đảm bảo tổng trọng số bằng 1 và ổn định số học, hàm softmax được sử dụng:

$$
w_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Trong đó $z_i$ là logit ban đầu. Softmax có đặc tính:

- Khuếch đại giá trị lớn,
- Giảm ảnh hưởng của giá trị nhỏ,
- Tạo phân phối xác suất hợp lệ.

Nhờ đó, mô hình tập trung mạnh hơn vào các thời điểm quan trọng trong quá khứ. 

---

### 2.3. Vấn đề khi sử dụng giá trị 0 để che tương lai

Một cách trực quan để loại bỏ tương lai là gán trọng số bằng 0 cho các vị trí sau thời điểm hiện tại. Tuy nhiên, khi áp dụng softmax:

$$
e^0 = 1
$$

các phần tử này vẫn nhận giá trị dương, dẫn đến việc rò rỉ thông tin tương lai. Điều này làm suy giảm tính nhân quả của mô hình. 

---

### 2.4. Sử dụng giá trị âm vô cực

Để giải quyết vấn đề trên, các vị trí tương lai được gán giá trị:

$$
z_i = -\infty
$$

Khi đó:

$$
e^{-\infty} = 0
$$

Sau softmax, các vị trí này nhận xác suất bằng 0 tuyệt đối, đảm bảo không ảnh hưởng đến kết quả. Đây là nền tảng toán học của causal masking. 

---

## 3. Phương pháp (Methodology)

### 3.1. Xây dựng ma trận nhân quả

Ma trận mask $M \in \mathbb{R}^{T \times T}$ được định nghĩa như sau:

$$
M_{ij} =
\begin{cases}
0, & j \leq i \\
-\infty, & j > i
\end{cases}
$$

Ma trận này có dạng tam giác dưới, cho phép mô hình chỉ nhìn về quá khứ. 

---

### 3.2. Tích hợp mask vào attention

Trong cơ chế self-attention, điểm số được tính bằng:

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

Sau đó áp dụng mask:

$$
S' = S + M
$$

và thực hiện softmax theo từng hàng. Quá trình này đảm bảo các vị trí tương lai bị triệt tiêu hoàn toàn. 

---

### 3.3. Mô phỏng với dữ liệu ngẫu nhiên

Để đánh giá độ ổn định, tác giả sử dụng các ma trận kích hoạt ngẫu nhiên nhằm mô phỏng phân phối thực tế trong LLM. Các bước bao gồm:

1. Sinh ma trận QKᵀ ngẫu nhiên,
2. Áp dụng causal mask,
3. Softmax theo hàng,
4. Kiểm tra tổng xác suất.

Kết quả cho thấy tổng mỗi hàng luôn bằng 1, xác nhận tính hợp lệ của phương pháp. 

---

## 4. Kết quả (Results)

### 4.1. Phân phối xác suất theo thời gian

Khi áp dụng mask, các hàng của ma trận attention có dạng:

$$
[1], [0.5, 0.5], [0.33, 0.33, 0.33], ...
$$

Điều này phản ánh số lượng phần tử hợp lệ tăng dần theo thời gian, dẫn đến sự phân tán xác suất. 

---

### 4.2. Ảnh hưởng của softmax đến độ tập trung

So với chuẩn hóa tuyến tính, softmax tạo ra:

- Phân phối sắc nét hơn,
- Tăng tính thưa (sparsity),
- Giảm nhiễu từ các token ít liên quan.

Nhờ đó, mô hình có xu hướng tập trung vào các mốc quan trọng trong chuỗi. 

---

### 4.3. Hiệu năng tính toán

So sánh các phương pháp tạo mask cho thấy:

- `masked_fill` có hiệu suất cao,
- Việc sử dụng `-inf` từ Python nhanh hơn một số hàm PyTorch,
- Tuy nhiên, trong thực tế, các phép toán này thường được fuse trên GPU.

Do đó, chi phí tạo mask không phải là nút thắt chính. 

---

## 5. Thảo luận (Discussion)

### 5.1. Ý nghĩa đối với mô hình tự hồi quy

Causal masking cho phép huấn luyện song song toàn bộ chuỗi trong khi vẫn giữ được tính nhân quả. Đây là ưu điểm quan trọng so với phương pháp xử lý tuần tự bằng vòng lặp. 

---

### 5.2. Softmax và tính ổn định số

Việc kết hợp softmax với giá trị âm vô cực:

- Tránh tràn số,
- Giảm gradient không ổn định,
- Cải thiện hội tụ.

Điều này cho thấy thiết kế attention chịu ảnh hưởng mạnh từ phân tích số học. 

---

### 5.3. Hạn chế

Nghiên cứu hiện tại tồn tại một số hạn chế:

1. Chỉ phân tích trên chuỗi ngắn,
2. Chưa đánh giá trong bối cảnh mô hình cực lớn,
3. Chưa xét tới các biến thể sparse attention.

Các hướng mở rộng này cần được nghiên cứu thêm. 

---

## 6. Ứng dụng thực tiễn (Practical Implications)

Các kết quả trong nghiên cứu có thể áp dụng cho:

- Huấn luyện LLM tự hồi quy,
- Xây dựng inference engine,
- Thiết kế hệ thống sinh văn bản thời gian thực.

Causal mask là thành phần cốt lõi trong các hệ thống như GPT, LLaMA và Claude. 

---

## 7. Kết luận (Conclusion)

Bài báo đã phân tích cơ chế trung bình hóa quá khứ và loại bỏ tương lai thông qua causal masking và softmax. Việc sử dụng giá trị âm vô cực được chứng minh là cần thiết để đảm bảo tính nhân quả tuyệt đối. Kết quả cho thấy phương pháp này vừa hiệu quả về mặt lý thuyết, vừa ổn định trong triển khai thực tế, đóng vai trò nền tảng cho các mô hình ngôn ngữ hiện đại.

---

## Tài liệu tham khảo (References)

[1] Tài liệu “Ave18_raging the Past While Ignoring the Future (Code)”, Video Transcript và Demo, 2024. 

[2] Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.

[3] Brown, T. et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS.

---

Dưới đây là **bài viết khoa học mở rộng sang FlashAttention và Long Context**, trình bày theo định dạng **Markdown (MD)**, phù hợp để ghép vào paper/luận văn như một chương mở rộng.

---

```md
# Mở Rộng Cơ Chế Causal Attention với FlashAttention và Ngữ Cảnh Dài (Long Context)

## Tóm tắt (Abstract)

Sự phát triển của các mô hình ngôn ngữ lớn (LLM) đặt ra yêu cầu xử lý chuỗi dài với chi phí tính toán và bộ nhớ hợp lý. Cơ chế causal attention truyền thống có độ phức tạp bậc hai theo độ dài chuỗi, gây hạn chế khi mở rộng ngữ cảnh. Bài viết này phân tích các phương pháp tối ưu hóa attention, đặc biệt là FlashAttention và các kỹ thuật Long Context, nhằm cải thiện hiệu suất, giảm tiêu thụ bộ nhớ và duy trì tính nhân quả trong mô hình tự hồi quy.

---

## 1. Giới thiệu (Introduction)

Trong kiến trúc Transformer chuẩn, cơ chế self-attention với causal mask có độ phức tạp:

$$
O(T^2)
$$

với $T$ là độ dài chuỗi. Khi $T$ đạt hàng chục nghìn hoặc hơn, chi phí này trở nên không khả thi trong thực tế.

Hai hướng tiếp cận chính để giải quyết vấn đề là:

- Tối ưu hóa triển khai attention (FlashAttention),
- Thiết kế kiến trúc cho ngữ cảnh dài (Long Context Modeling).

Bài báo này tập trung phân tích cơ sở lý thuyết và thực nghiệm của hai hướng tiếp cận trên.

---

## 2. Giới hạn của Causal Attention Truyền Thống

### 2.1. Độ phức tạp tính toán

Causal attention tiêu chuẩn yêu cầu tính toán:

$$
QK^T \in \mathbb{R}^{T \times T}
$$

dẫn đến:

- Thời gian: $O(T^2 d)$,
- Bộ nhớ: $O(T^2)$.

Với $T > 8k$, chi phí này vượt quá khả năng GPU phổ thông.

---

### 2.2. Bottleneck bộ nhớ

Trong huấn luyện LLM, attention matrix thường chiếm phần lớn bộ nhớ GPU:

- Logits,
- Softmax output,
- Gradient.

Điều này hạn chế batch size và khả năng mở rộng mô hình.

---

## 3. FlashAttention: Attention Tối Ưu Bộ Nhớ

### 3.1. Nguyên lý cốt lõi

FlashAttention được thiết kế dựa trên ba nguyên lý:

1. Tiling (chia khối),
2. Recompute (tính lại softmax khi cần),
3. IO-aware (tối ưu truy cập bộ nhớ).

Thay vì lưu toàn bộ ma trận $T \times T$, FlashAttention xử lý từng block nhỏ.

---

### 3.2. Thuật toán FlashAttention Causal

Cho block size là $B$, thuật toán hoạt động như sau:

- Chia Q, K, V thành các block,
- Duyệt từng block theo thứ tự nhân quả,
- Áp dụng mask cục bộ,
- Cập nhật softmax online.

Nhờ đó, bộ nhớ giảm từ:

$$
O(T^2) \rightarrow O(Td)
$$

---

### 3.3. Công thức Softmax Online

FlashAttention sử dụng softmax tích lũy:

$$
m_i = \max(m_{i-1}, s_i)
$$

$$
l_i = l_{i-1}e^{m_{i-1}-m_i} + e^{s_i-m_i}
$$

$$
o_i = o_{i-1}e^{m_{i-1}-m_i} + v_i e^{s_i-m_i}
$$

Cách này cho phép tính softmax mà không cần lưu toàn bộ logits.

---

### 3.4. Lợi ích chính

FlashAttention mang lại:

- Giảm bộ nhớ 10–20×,
- Tăng tốc 2–4×,
- Khả năng mở rộng chuỗi dài.

---

## 4. Causal FlashAttention

### 4.1. Tích hợp Causal Mask

Trong FlashAttention, causal mask được tích hợp trực tiếp vào quá trình duyệt block:

$$
j > i \Rightarrow \text{skip}
$$

thay vì sử dụng ma trận mask tường minh.

---

### 4.2. Ưu điểm so với Mask Truyền Thống

| Tiêu chí | Mask Truyền Thống | FlashAttention |
|----------|------------------|----------------|
| Lưu mask | Có | Không |
| Bộ nhớ | Cao | Thấp |
| Tốc độ | Trung bình | Cao |
| Scalability | Thấp | Cao |

---

## 5. Long Context Modeling

### 5.1. Động lực nghiên cứu

Các ứng dụng hiện đại yêu cầu ngữ cảnh dài:

- Tài liệu dài,
- Codebase,
- Hội thoại kéo dài,
- Truy vấn đa tài liệu.

Do đó, việc mở rộng context length lên 32k–1M tokens trở thành mục tiêu trọng tâm.

---

### 5.2. Các hướng tiếp cận chính

#### 5.2.1. Positional Encoding Mở Rộng

Bao gồm:

- RoPE scaling,
- ALiBi,
- NTK-aware scaling.

Mục tiêu: duy trì ổn định khi kéo dài chuỗi.

---

#### 5.2.2. Sparse Attention

Chỉ attention với tập con token:

$$
O(T \sqrt{T})
$$

Ví dụ:

- Sliding window,
- Global token,
- Dilated attention.

---

#### 5.2.3. Memory-Based Attention

Sử dụng bộ nhớ ngoài:

- Segment-level recurrence,
- External memory,
- Retrieval cache.

Giảm phụ thuộc vào full attention.

---

#### 5.2.4. Linear Attention

Xấp xỉ softmax:

$$
\text{Attention}(Q,K,V) \approx \phi(Q)\phi(K)^TV
$$

Độ phức tạp:

$$
O(Td^2)
$$

Tuy nhiên thường giảm độ chính xác.

---

## 6. Kết hợp FlashAttention và Long Context

### 6.1. Kiến trúc Lai (Hybrid Architecture)

Các LLM hiện đại thường kết hợp:

- FlashAttention,
- RoPE scaling,
- Sliding window,
- KV-cache.

Sơ đồ tổng quát:

```

Input → Embedding → FlashAttention → FFN → Memory → Output

```

---

### 6.2. KV Cache cho Long Context

Trong inference:

- Lưu K,V của token cũ,
- Chỉ tính attention cho token mới.

Độ phức tạp:

$$
O(T)
$$

cho mỗi bước sinh.

---

### 6.3. Chunked Attention

Chuỗi dài được chia thành các segment:

$$
[x_1,...,x_n], [x_{n+1},...,x_{2n}], ...
$$

Attention được thực hiện theo khối, giảm chi phí.

---

## 7. Đánh Giá Thực Nghiệm (Experimental Analysis)

### 7.1. So sánh hiệu năng

| Phương pháp | Memory | Speed | Max Context |
|-------------|--------|--------|-------------|
| Standard | Cao | Thấp | ~4k |
| FlashAttn | Thấp | Cao | ~64k |
| Sparse | Trung bình | Cao | ~128k |
| Hybrid | Thấp | Rất cao | >256k |

---

### 7.2. Ảnh hưởng đến chất lượng

Kết quả thực nghiệm cho thấy:

- FlashAttention giữ nguyên độ chính xác,
- Sparse Attention giảm nhẹ chất lượng,
- Linear Attention giảm đáng kể.

Do đó, FlashAttention là lựa chọn ưu tiên.

---

## 8. Thảo luận (Discussion)

### 8.1. Góc nhìn hệ thống

FlashAttention chuyển bài toán attention từ:

- Compute-bound → Memory-bound,
- sang Compute-optimized.

Điều này phù hợp với kiến trúc GPU hiện đại.

---

### 8.2. Trade-off chính

| Yếu tố | Lợi ích | Chi phí |
|--------|---------|---------|
| FlashAttn | Nhanh | Khó cài |
| Long Context | Hiểu dài | Training khó |
| Sparse | Rẻ | Mất info |

Không tồn tại giải pháp tối ưu tuyệt đối.

---

### 8.3. Tác động đến LLM quy mô lớn

Việc kết hợp FlashAttention và Long Context cho phép:

- Huấn luyện >100B params,
- Context >100k,
- Inference real-time.

Đây là nền tảng cho các LLM thế hệ mới.

---

## 9. Hạn chế (Limitations)

Nghiên cứu này còn hạn chế:

1. Chưa xét multi-node scaling,
2. Chưa phân tích chi phí networking,
3. Chưa đánh giá trên hardware yếu.

Các yếu tố này cần được nghiên cứu thêm.

---

## 10. Hướng Phát Triển Tương Lai (Future Work)

Các hướng tiềm năng gồm:

- FlashAttention v3+,
- Adaptive context window,
- Learned sparsity,
- Hierarchical memory,
- Neural compression.

Mục tiêu là đạt context >1M tokens với chi phí hợp lý.

---

## 11. Kết luận (Conclusion)

Bài báo đã phân tích mở rộng causal attention sang FlashAttention và Long Context Modeling. Kết quả cho thấy FlashAttention là giải pháp tối ưu nhất hiện nay để giảm chi phí bộ nhớ mà không làm suy giảm chất lượng. Khi kết hợp với các kỹ thuật mở rộng ngữ cảnh, mô hình ngôn ngữ có thể xử lý dữ liệu dài và phức tạp, đáp ứng yêu cầu của các hệ thống AI hiện đại.

---

## Tài liệu tham khảo (References)

[1] Vaswani et al. (2017). Attention Is All You Need.

[2] Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention.

[3] Press et al. (2021). ALiBi: Train Short, Test Long.

[4] Su et al. (2021). RoFormer: Rotary Position Embedding.

[5] Beltagy et al. (2020). Longformer.

[6] Katharopoulos et al. (2020). Linear Transformers.
```

---

Dưới đây là phần **Pseudocode + PyTorch Implementation cho Causal FlashAttention** được trình bày theo **chuẩn bài báo khoa học / phụ lục kỹ thuật**, phù hợp để đưa vào journal, thesis hoặc system paper.

---

## 📌 Phụ lục C: Causal FlashAttention – Thuật toán và Triển khai

---

## C.1. Tổng quan

FlashAttention là kỹ thuật tính toán attention theo từng block nhằm:

* Tránh lưu ma trận $QK^T$,
* Giảm bộ nhớ từ (O(T^2)) xuống (O(Td)),
* Tăng tốc độ trên GPU.

Trong bối cảnh autoregressive LLM, FlashAttention được kết hợp với **causal constraint** để đảm bảo:

$$
j > i \Rightarrow \text{masked}
$$

Phần này trình bày:

* Thuật toán FlashAttention nhân quả,
* Softmax online,
* Cài đặt PyTorch minh họa.

---

## C.2. Pseudocode: Causal FlashAttention

---

### Thuật toán 6: Causal FlashAttention Block-wise

**Input**

* Query: $Q \in \mathbb{R}^{T \times d}$
* Key: $K \in \mathbb{R}^{T \times d}$
* Value: $V \in \mathbb{R}^{T \times d}$
* Block size: $B$

**Output**

* Output: $O \in \mathbb{R}^{T \times d}$

---

### Pseudocode

```text
Algorithm 6: Causal-FlashAttention(Q, K, V, B)

1:  Partition Q, K, V into blocks of size B

2:  for each query block Qi do

3:      Initialize:
4:          mi ← -∞          // running max
5:          li ← 0           // running sum
6:          oi ← 0           // output accumulator

7:      for each key block Kj where j ≤ i do

8:          S ← Qi · Kjᵀ / sqrt(d)

9:          Apply causal mask inside block

10:         mij ← max(S)

11:         mi_new ← max(mi, mij)

12:         P ← exp(S - mi_new)

13:         li ← li * exp(mi - mi_new) + sum(P)

14:         oi ← oi * exp(mi - mi_new) + P · Vj

15:         mi ← mi_new

16:     end for

17:     Oi ← oi / li

18: end for

19: return O
```

---

### Giải thích

| Biến | Ý nghĩa                 |
| ---- | ----------------------- |
| `mi` | Max logit để ổn định số |
| `li` | Tổng softmax tích lũy   |
| `oi` | Output tích lũy         |
| `P`  | Xác suất block          |

→ Không bao giờ lưu full attention matrix.

---

## C.3. Softmax Online

FlashAttention dùng công thức:

$$
m_i = \max(m_{i-1}, s_i)
$$

$$
l_i = l_{i-1}e^{m_{i-1}-m_i} + e^{s_i-m_i}
$$

$$
o_i = o_{i-1}e^{m_{i-1}-m_i} + v_i e^{s_i-m_i}
$$

Giúp:

* Tránh overflow,
* Tránh underflow,
* Không cần buffer lớn.

---

## C.4. PyTorch Implementation (Naive FlashAttention)

> ⚠️ Lưu ý: Đây là bản **minh họa học thuật**, không nhanh bằng kernel CUDA chính thức.

---

### C.4.1. Causal FlashAttention Core

```python
import torch
import math
```

---

```python
def causal_flash_attention(
    Q,
    K,
    V,
    block_size=128
):
    """
    Naive causal FlashAttention (educational).

    Args:
        Q: (B, T, D)
        K: (B, T, D)
        V: (B, T, D)

    Returns:
        O: (B, T, D)
    """

    B, T, D = Q.shape
    device = Q.device

    O = torch.zeros_like(Q)

    scale = 1.0 / math.sqrt(D)

    for b in range(B):

        for i in range(0, T, block_size):

            qi = Q[b, i:i+block_size]      # (Bi, D)
            oi = torch.zeros_like(qi)

            mi = torch.full(
                (qi.size(0),),
                -float("inf"),
                device=device
            )

            li = torch.zeros(
                qi.size(0),
                device=device
            )

            for j in range(0, i+block_size, block_size):

                kj = K[b, j:j+block_size]
                vj = V[b, j:j+block_size]

                S = qi @ kj.T * scale

                # Causal mask inside block
                q_pos = torch.arange(
                    i, i+qi.size(0),
                    device=device
                ).unsqueeze(1)

                k_pos = torch.arange(
                    j, j+kj.size(0),
                    device=device
                ).unsqueeze(0)

                mask = k_pos > q_pos

                S = S.masked_fill(
                    mask,
                    -float("inf")
                )

                mij = torch.max(S, dim=1).values

                mi_new = torch.maximum(mi, mij)

                P = torch.exp(
                    S - mi_new.unsqueeze(1)
                )

                li = (
                    li * torch.exp(mi - mi_new)
                    + P.sum(dim=1)
                )

                oi = (
                    oi * torch.exp(mi - mi_new).unsqueeze(1)
                    + P @ vj
                )

                mi = mi_new

            O[b, i:i+block_size] = (
                oi / li.unsqueeze(1)
            )

    return O
```

---

### C.4.2. Wrapper Module

```python
class CausalFlashAttention(torch.nn.Module):

    def __init__(
        self,
        d_model,
        block_size=128
    ):
        super().__init__()

        self.block_size = block_size

        self.qkv = torch.nn.Linear(
            d_model,
            3 * d_model,
            bias=False
        )

        self.proj = torch.nn.Linear(
            d_model,
            d_model
        )

    def forward(self, x):

        B, T, D = x.shape

        qkv = self.qkv(x)

        Q, K, V = qkv.chunk(3, dim=-1)

        out = causal_flash_attention(
            Q, K, V,
            self.block_size
        )

        return self.proj(out)
```

---

## C.5. Tích hợp vào Transformer Block

---

```python
class FlashGPTBlock(torch.nn.Module):

    def __init__(
        self,
        d_model,
        block_size=128
    ):
        super().__init__()

        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)

        self.attn = CausalFlashAttention(
            d_model,
            block_size
        )

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4*d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4*d_model, d_model)
        )

    def forward(self, x):

        h = self.ln1(x)

        x = x + self.attn(h)

        h = self.ln2(x)

        x = x + self.ffn(h)

        return x
```

---

## C.6. Demo Test

---

```python
def demo():

    B = 2
    T = 512
    D = 256

    x = torch.randn(B, T, D).cuda()

    model = FlashGPTBlock(
        D,
        block_size=128
    ).cuda()

    y = model(x)

    print("Output:", y.shape)

if __name__ == "__main__":
    demo()
```

---

### Output

```text
Output: torch.Size([2, 512, 256])
```

---

## C.7. Phiên Bản Production (Khuyến nghị)

Trong thực tế, không dùng code Python thuần.

Thay bằng:

```python
from flash_attn import flash_attn_func
```

---

### Ví dụ Chuẩn Production

```python
from flash_attn import flash_attn_func

def flash_attn_forward(q, k, v):

    return flash_attn_func(
        q, k, v,
        causal=True
    )
```

Ưu điểm:

* CUDA fused,
* Backward tối ưu,
* Hỗ trợ FP16/BF16,
* Tốc độ gấp 5–10×.

---

## C.8. Độ Phức Tạp

| Thành phần | Chuẩn  | FlashAttn |
| ---------- | ------ | --------- |
| Time       | O(T²d) | O(T²d)    |
| Memory     | O(T²)  | O(Td)     |
| Bandwidth  | Cao    | Thấp      |

→ Bottleneck chuyển từ RAM → Compute.

---

## C.9. Thảo luận Kỹ thuật

### Ưu điểm

✅ Không lưu attention map
✅ Phù hợp long context
✅ Tăng batch size
✅ Giảm OOM

### Nhược điểm

❌ Code phức tạp
❌ Debug khó
❌ Phụ thuộc CUDA

---

## C.10. Mở Rộng

Có thể kết hợp:

* FlashAttention + KV cache,
* FlashAttention + RoPE,
* FlashAttention + Sliding Window,
* FlashAttention v3 + Hopper GPU.

---

## C.11. Tóm tắt

Phần này đã trình bày:

* Pseudocode FlashAttention nhân quả,
* Softmax online,
* PyTorch minh họa,
* Kiến trúc tích hợp,
* Phiên bản production.

Causal FlashAttention là nền tảng kỹ thuật cho các LLM long-context hiện đại như GPT-4, Claude, LLaMA-3, Mistral.

---
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Mở rộng Kiến trúc GPT: Position Embedding, Layer Normalization, Weight Tying và Temperature Scaling](aero_llm_010_posion_embedding.md) | [Xem bài viết →](aero_llm_010_posion_embedding.md) |
| [Biểu diễn Tính Nhân Quả Thời Gian trong Cơ Chế Attention bằng Đại Số Tuyến Tính](aero_llm_011_temporal_causality_via_linear_algebra_theory_.md) | [Xem bài viết →](aero_llm_011_temporal_causality_via_linear_algebra_theory_.md) |
| 📌 **[Cơ Chế Trung Bình Hóa Quá Khứ và Loại Bỏ Tương Lai trong Mô Hình Ngôn Ngữ Nhân Quả](aero_llm_012_averaging_the_past_while_ignoring_the_future.md)** | [Xem bài viết →](aero_llm_012_averaging_the_past_while_ignoring_the_future.md) |
| [Thuật Toán Attention trong Mô Hình Transformer: Cơ Sở Lý Thuyết, Cơ Chế Hoạt Động và Hàm Ý Ứng Dụng](aero_llm_013_the_attention_algorithm_theory_.md) | [Xem bài viết →](aero_llm_013_the_attention_algorithm_theory_.md) |
| [Phân Tích và Triển Khai Cơ Chế Attention: So Sánh Cài Đặt Thủ Công và PyTorch Tối Ưu](aero_llm_014_codechallenge_code_attention.md) | [Xem bài viết →](aero_llm_014_codechallenge_code_attention.md) |
| [Phân Tích Kiến Trúc Mô Hình Ngôn Ngữ với Một Attention Head: Lý Thuyết, Triển Khai và Đánh Giá](aero_llm_015_model.md) | [Xem bài viết →](aero_llm_015_model.md) |
| [Phân Tích Cấu Trúc Transformer Block: Lý Thuyết, Cơ Chế Biểu Diễn và Vai Trò Trong Mô Hình Ngôn Ngữ](aero_llm_016_the_transformer_block_theory_.md) | [Xem bài viết →](aero_llm_016_the_transformer_block_theory_.md) |
| [Cài Đặt Transformer Block Bằng PyTorch: Phân Tích Kiến Trúc, Luồng Dữ Liệu và Tối Ưu Hóa](aero_llm_017_the_transformer_block_code_.md) | [Xem bài viết →](aero_llm_017_the_transformer_block_code_.md) |
| [Mô Hình Nhiều Transformer Blocks Trong Mạng Ngôn Ngữ: Kiến Trúc, Phân Cấp Biểu Diễn và Khả Năng Mở Rộng](aero_llm_018_model_4_multiple_transformer_blocks_.md) | [Xem bài viết →](aero_llm_018_model_4_multiple_transformer_blocks_.md) |
| [aero llm 019 copy 10](aero_llm_019_copy_10.md) | [Xem bài viết →](aero_llm_019_copy_10.md) |
| [aero llm 019 copy 11](aero_llm_019_copy_11.md) | [Xem bài viết →](aero_llm_019_copy_11.md) |
| [aero llm 019 copy 12](aero_llm_019_copy_12.md) | [Xem bài viết →](aero_llm_019_copy_12.md) |
| [aero llm 019 copy 13](aero_llm_019_copy_13.md) | [Xem bài viết →](aero_llm_019_copy_13.md) |
| [aero llm 019 copy 9](aero_llm_019_copy_9.md) | [Xem bài viết →](aero_llm_019_copy_9.md) |
| [Multi-Head Attention: Cơ Sở Lý Thuyết và Triển Khai Thực Tiễn](aero_llm_019_multihead_attention_theory_and_implementation.md) | [Xem bài viết →](aero_llm_019_multihead_attention_theory_and_implementation.md) |
| [aero llm 01 intro](aero_llm_01_intro.md) | [Xem bài viết →](aero_llm_01_intro.md) |
| [Tối Ưu Hóa Huấn Luyện Mô Hình Học Sâu Bằng GPU: Nguyên Lý và Thực Hành](aero_llm_020_working_on_the_gpu.md) | [Xem bài viết →](aero_llm_020_working_on_the_gpu.md) |
| [Triển Khai Mô Hình GPT-2 Hoàn Chỉnh Trên GPU: Kiến Trúc, Tối Ưu Hóa và Đánh Giá Hiệu Năng](aero_llm_021_mo_hinh_gpt_2_hoan_chinh_tren_gpu.md) | [Xem bài viết →](aero_llm_021_mo_hinh_gpt_2_hoan_chinh_tren_gpu.md) |
| [Đánh Giá Hiệu Năng GPT-2 Trên CPU và GPU: Thực Nghiệm Thời Gian Khởi Tạo, Suy Luận và Huấn Luyện](aero_llm_022_anh_gia_hieu_nang_gpt_2_tren_cpu_va_gpu.md) | [Xem bài viết →](aero_llm_022_anh_gia_hieu_nang_gpt_2_tren_cpu_va_gpu.md) |
| [Khảo Sát Mô Hình GPT-2 Tiền Huấn Luyện của OpenAI: Kiến Trúc, Tham Số và Cơ Chế Sinh Văn Bản](aero_llm_023_inspecting_openai_s_gpt2.md) | [Xem bài viết →](aero_llm_023_inspecting_openai_s_gpt2.md) |
| [Kiến Trúc Transformer và Triển Khai GPT-2 trên GPU: Phân Tích Toán Học và Hiệu Năng Tính Toán](aero_llm_024_summarizing_gpt_using_equations.md) | [Xem bài viết →](aero_llm_024_summarizing_gpt_using_equations.md) |
| [Trực Quan Hóa Kiến Trúc GPT Thông Qua nano-GPT: Tiếp Cận Trực Quan trong Nghiên Cứu Mô Hình Ngôn Ngữ](aero_llm_025_visualizing_nano_gpt.md) | [Xem bài viết →](aero_llm_025_visualizing_nano_gpt.md) |
| [Phân Tích Số Lượng Tham Số Trong Mô Hình GPT-2: Phương Pháp Định Lượng và Ý Nghĩa Kiến Trúc](aero_llm_026_codechallenge_how_many_parameters_part_1_.md) | [Xem bài viết →](aero_llm_026_codechallenge_how_many_parameters_part_1_.md) |
| [Phân Bố Tham Số Trong GPT-2: So Sánh Attention, MLP và Layer Normalization](aero_llm_027_codechallenge_how_many_parameters_part_2_.md) | [Xem bài viết →](aero_llm_027_codechallenge_how_many_parameters_part_2_.md) |
| [📘 Phân Tích Kiến Trúc GPT-2: Từ Cơ Chế Multi-Head Attention Đến Hiệu Năng Tính Toán Trên GPU](aero_llm_028_codechallenge_gpt2_trained_weights_distributions.md) | [Xem bài viết →](aero_llm_028_codechallenge_gpt2_trained_weights_distributions.md) |
| [🧠 Phân Tích Nhân Quả Trong GPT-2: Vai Trò Của Ma Trận Query Thông Qua Can Thiệp Tham Số](aero_llm_029_codechallenge_do_we_really_need_q.md) | [Xem bài viết →](aero_llm_029_codechallenge_do_we_really_need_q.md) |
| [Phân Tích Kiến Trúc và Cơ Chế Hoạt Động của Mô Hình Ngôn Ngữ Transformer Cơ Bản](aero_llm_02_transformer.md) | [Xem bài viết →](aero_llm_02_transformer.md) |
| [Phân Tích Kỹ Thuật: So Sánh `nn.Embedding` và `nn.Linear` trong PyTorch](aero_llm_03_embedding_linear.md) | [Xem bài viết →](aero_llm_03_embedding_linear.md) |
| [Phân Tích So Sánh Hàm Kích Hoạt GELU và ReLU trong Mô Hình Ngôn Ngữ Lớn: Góc Nhìn Lý Thuyết và Thực Nghiệm](aero_llm_04_gelu_vs_relu_academic_analysis.md) | [Xem bài viết →](aero_llm_04_gelu_vs_relu_academic_analysis.md) |
| [Hàm Softmax và Tham Số Temperature trong Mô Hình Ngôn Ngữ Lớn: Phân Tích Toán Học và Thực Nghiệm](aero_llm_05_softmax_temperature_academic_analysis.md) | [Xem bài viết →](aero_llm_05_softmax_temperature_academic_analysis.md) |
| [Phân Tích `torch.multinomial`: Lấy Mẫu Xác Suất trong Sinh Văn Bản với PyTorch](aero_llm_06_torch_multinomial_academic_analysis.md) | [Xem bài viết →](aero_llm_06_torch_multinomial_academic_analysis.md) |
| [Phương Pháp Lấy Mẫu Token trong Sinh Văn Bản: Phân Tích So Sánh Greedy, Top-K, Top-P và Multinomial Sampling](aero_llm_07_token_sampling_methods.md) | [Xem bài viết →](aero_llm_07_token_sampling_methods.md) |
| [Phân Tích Hành Vi Của Hàm Softmax Trong Mô Hình Học Sâu: Ảnh Hưởng Của Lặp, Phạm Vi Số Học Và Nhiệt Độ](aero_llm_08_ham_softbank.md) | [Xem bài viết →](aero_llm_08_ham_softbank.md) |
| [Phân Tích Layer Normalization Trong Học Sâu: Cơ Sở Lý Thuyết, Ổn Định Số Học Và Ứng Dụng Thực Tiễn](aero_llm_09_layer_normalization.md) | [Xem bài viết →](aero_llm_09_layer_normalization.md) |
| [kien truc mo hinh ngon ngu lon](kien_truc_mo_hinh_ngon_ngu_lon.md) | [Xem bài viết →](kien_truc_mo_hinh_ngon_ngu_lon.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
