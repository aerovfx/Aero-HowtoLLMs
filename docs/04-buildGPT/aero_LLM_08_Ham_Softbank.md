
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [04 buildGPT](../index.md)

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
Dưới đây là bài viết khoa học dưới dạng **Markdown**, dựa trên nội dung bạn cung cấp từ tài liệu *“More Softmax Explorations”* , kết hợp phân tích lý thuyết và tham khảo học thuật.

---

# Phân Tích Hành Vi Của Hàm Softmax Trong Mô Hình Học Sâu: Ảnh Hưởng Của Lặp, Phạm Vi Số Học Và Nhiệt Độ

## Tóm tắt (Abstract)

Hàm Softmax là một thành phần cốt lõi trong các mô hình học sâu, đặc biệt trong xử lý ngôn ngữ tự nhiên và thị giác máy tính. Bài viết này phân tích hành vi của Softmax thông qua hai thí nghiệm: (1) áp dụng Softmax lặp nhiều lần lên cùng một phân phối, và (2) khảo sát ảnh hưởng của phạm vi giá trị logits và tham số nhiệt độ (temperature). Kết quả cho thấy Softmax có xu hướng làm phẳng phân phối khi được lặp lại, đồng thời rất nhạy cảm với miền giá trị số và nhiệt độ. Những phát hiện này nhấn mạnh vai trò của chuẩn hóa và kiểm soát độ ổn định số trong các mô hình học sâu hiện đại.

---

## 1. Giới thiệu

Trong học sâu, Softmax thường được sử dụng để chuyển đổi vector logits thành phân phối xác suất. Cho vector đầu vào ( x = (x_1, x_2, ..., x_n) ), Softmax được định nghĩa như sau:

[
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
]

Hàm này đảm bảo rằng:

* Mỗi giá trị đầu ra nằm trong khoảng (0,1),
* Tổng các giá trị bằng 1.

Mặc dù công thức đơn giản, hành vi thực tế của Softmax trong môi trường số học và huấn luyện mô hình phức tạp hơn nhiều. Tài liệu thực nghiệm được cung cấp  cho thấy nhiều hiện tượng phi trực giác, đặc biệt khi Softmax được áp dụng lặp lại hoặc kết hợp với tham số nhiệt độ.

---

## 2. Cơ sở lý thuyết

### 2.1. Softmax và phân phối xác suất

Softmax biến đổi các giá trị logits thành xác suất bằng hàm mũ. Do tính chất tăng nhanh của hàm mũ, những giá trị lớn sẽ được khuếch đại, trong khi giá trị nhỏ bị suy giảm mạnh.

### 2.2. Softmax với tham số nhiệt độ

Phiên bản mở rộng của Softmax có dạng:

[
\text{Softmax}*T(x_i) = \frac{e^{x_i/T}}{\sum*{j=1}^{n} e^{x_j/T}}
]

Trong đó (T) là nhiệt độ:

* (T < 1): Phân phối sắc nét (sharp), tập trung vào phần tử lớn nhất.
* (T = 1): Softmax chuẩn.
* (T > 1): Phân phối phẳng (smooth), tăng tính đa dạng.

### 2.3. Ổn định số học

Việc tính toán hàm mũ trên các giá trị lớn hoặc nhỏ có thể gây:

* Tràn số (overflow),
* Mất độ chính xác (underflow),
* Gradient biến mất hoặc bùng nổ.

Do đó, các kỹ thuật chuẩn hóa (normalization) là cần thiết trong mạng sâu.

---

## 3. Phương pháp nghiên cứu

Nghiên cứu dựa trên hai thí nghiệm chính được mô tả trong tài liệu gốc .

### 3.1. Thí nghiệm 1: Softmax lặp

#### Mô tả

* Tạo 20 số tuyến tính trong khoảng [0,1].
* Áp dụng Softmax.
* Lặp lại quá trình Softmax trên chính đầu ra nhiều lần (8 lần).
* Tính độ lệch chuẩn của phân phối sau mỗi lần lặp.

#### Mục tiêu

Khảo sát việc Softmax lặp có làm phân phối trở nên “sắc nét” hơn hay không.

---

### 3.2. Thí nghiệm 2: Phạm vi logits và nhiệt độ

#### Mô tả

* Sinh 100 logits trong các khoảng:

  * [-0.4, 0.4]
  * [-1, 1]
  * [-5, 5]
* Thêm một giá trị ngoại lai: 6.
* Áp dụng Softmax với các nhiệt độ: 0.5, 1, 3.
* Phân tích xác suất đầu ra.

#### Mục tiêu

Đánh giá ảnh hưởng của:

* Miền giá trị logits,
* Nhiệt độ,
* Giá trị ngoại lai.

---

## 4. Kết quả thực nghiệm

### 4.1. Hiệu ứng của Softmax lặp

Kết quả cho thấy:

* Sau vài lần lặp, phân phối hội tụ về dạng gần như đồng đều.
* Với 20 phần tử, mỗi giá trị tiến gần đến 0.05.
* Độ lệch chuẩn giảm nhanh theo cấp số nhân.

Điều này cho thấy Softmax lặp không làm nổi bật phần tử lớn nhất, mà ngược lại làm mất tính phân biệt.

### 4.2. Vai trò của số lượng phần tử

Số lượng phần tử ảnh hưởng trực tiếp đến giá trị trung bình:

| Số phần tử | Giá trị trung bình |
| ---------- | ------------------ |
| 4          | ≈ 0.25             |
| 20         | ≈ 0.05             |
| 100        | ≈ 0.01             |

Càng nhiều phần tử, xác suất riêng lẻ càng nhỏ.

### 4.3. Ảnh hưởng của nhiệt độ

| Nhiệt độ | Đặc điểm               |
| -------- | ---------------------- |
| T < 1    | Tập trung mạnh vào max |
| T = 1    | Cân bằng               |
| T > 1    | Phân tán               |

Ở T = 0.5, phần tử có logit = 6 chiếm gần như toàn bộ xác suất.
Ở T = 3, phân phối trở nên mềm hơn, tăng tính ngẫu nhiên.

### 4.4. Ảnh hưởng của phạm vi logits

Khi miền giá trị hẹp ([-0.4, 0.4]):

* Giá trị 6 trở nên vượt trội tuyệt đối.

Khi miền rộng ([-5, 5]):

* Sự khác biệt tương đối giảm.
* Phân phối cân bằng hơn.

Điều này chứng minh rằng Softmax phụ thuộc mạnh vào độ chênh lệch tương đối, không chỉ giá trị tuyệt đối.

---

## 5. Thảo luận

### 5.1. Vì sao Softmax lặp làm phẳng phân phối?

Do đầu ra Softmax đã nằm trong [0,1] và có tổng bằng 1. Khi tiếp tục áp dụng hàm mũ trên miền hẹp, hàm mũ trở nên gần tuyến tính, làm mất hiệu ứng khuếch đại.

### 5.2. Hệ quả trong mô hình ngôn ngữ

Trong mô hình ngôn ngữ lớn:

* Vocabulary có thể > 100,000 token.
* Softmax sẽ nén hầu hết xác suất về gần 0.
* Chỉ vài token chiếm ưu thế.

Do đó:

* Nhiệt độ thấp → mô hình lặp, ít sáng tạo.
* Nhiệt độ cao → đa dạng nhưng giảm độ chính xác.

### 5.3. Liên hệ với chuẩn hóa

Các kết quả cho thấy:

* Logits quá nhỏ → Softmax mất hiệu quả.
* Logits quá lớn → mất ổn định.

Vì vậy, các kỹ thuật như:

* Layer Normalization,
* Batch Normalization,
* Weight Regularization,

là cần thiết để duy trì miền giá trị hợp lý.

---

## 6. Ứng dụng thực tiễn

### 6.1. Điều chỉnh nhiệt độ sinh văn bản

| Mục tiêu  | Nhiệt độ  |
| --------- | --------- |
| Chính xác | 0.2 – 0.7 |
| Cân bằng  | ≈ 1       |
| Sáng tạo  | 1.2 – 2   |

### 6.2. Thiết kế mô hình

* Tránh để logits quá hẹp.
* Áp dụng normalization phù hợp.
* Kiểm soát scale khi huấn luyện.

### 6.3. Debug mô hình

Hiện tượng phân phối “chết” (collapse) có thể xuất phát từ:

* Logits bị nén,
* Temperature quá thấp,
* Lỗi chuẩn hóa.

---

## 7. Kết luận

Bài viết đã phân tích chi tiết hành vi của Softmax thông qua hai thí nghiệm thực nghiệm. Các kết luận chính bao gồm:

1. Softmax lặp làm phẳng phân phối thay vì làm sắc nét.
2. Số lượng phần tử ảnh hưởng trực tiếp đến xác suất.
3. Nhiệt độ kiểm soát mức độ tập trung.
4. Phạm vi logits quyết định mức độ phân biệt.
5. Chuẩn hóa là yếu tố sống còn trong mạng sâu.

Những kết quả này giúp hiểu rõ hơn về cách Softmax hoạt động trong các mô hình học sâu hiện đại và cung cấp cơ sở cho việc điều chỉnh mô hình hiệu quả.

---

## Tài liệu tham khảo

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
3. Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Springer.
4. Tài liệu thực nghiệm về Softmax 

---
**phiên bản Python/PyTorch chuẩn, đầy đủ, có chú thích khoa học**, tương ứng trực tiếp với phần *pseudocode* ở trên. Mã này có thể dùng để:

✅ Tái lập thí nghiệm
✅ Viết báo cáo / paper
✅ Làm supplementary material
✅ Thực hành nghiên cứu Softmax

Bạn có thể copy và chạy trực tiếp trong môi trường có `torch` và `matplotlib`.

---

# 📌 1. Cài đặt thư viện

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
```

---

# 📌 2. Hàm Softmax chuẩn (có nhiệt độ)

```python
def softmax_t(x, temperature=1.0):
    """
    Temperature-scaled softmax

    Args:
        x (Tensor): Logits
        temperature (float): Temperature parameter

    Returns:
        Tensor: Probability distribution
    """
    x_scaled = x / temperature
    exp_x = torch.exp(x_scaled - torch.max(x_scaled))  # numerical stability
    return exp_x / torch.sum(exp_x)
```

👉 Trừ `max(x)` để tránh overflow (chuẩn nghiên cứu).

---

# 📌 3. Thí nghiệm 1: Softmax lặp

## 3.1. Hàm thực nghiệm

```python
def iterative_softmax_experiment(
    n_points=20,
    n_iters=8,
    min_val=0.0,
    max_val=1.0
):
    """
    Iterative Softmax Experiment

    Returns:
        probs (list): List of distributions
        stds (list): Standard deviations
    """

    # Generate linear data
    x = torch.linspace(min_val, max_val, n_points)

    probs = []
    stds = []

    # Initial softmax
    p = softmax_t(x)

    probs.append(p.clone())

    for i in range(n_iters):

        std = torch.std(p)
        stds.append(std.item())

        # Apply softmax again
        p = softmax_t(p)

        probs.append(p.clone())

    return probs, stds, x
```

---

## 3.2. Chạy thí nghiệm

```python
probs, stds, x = iterative_softmax_experiment()
```

---

## 3.3. Vẽ kết quả

### Phân phối theo vòng lặp

```python
plt.figure(figsize=(8, 6))

for i, p in enumerate(probs):
    plt.scatter(x, p, label=f"Iter {i}", s=30)

plt.xlabel("Input values")
plt.ylabel("Softmax probability")
plt.title("Iterative Softmax Behavior")
plt.legend()
plt.grid(True)
plt.show()
```

---

### Log độ lệch chuẩn

```python
log_stds = np.log(np.array(stds) + 1e-12)

plt.figure(figsize=(6, 5))

plt.plot(range(len(log_stds)), log_stds, marker='o')

plt.xlabel("Iteration")
plt.ylabel("Log(Standard Deviation)")
plt.title("Convergence of Iterative Softmax")
plt.grid(True)
plt.show()
```

---

# 📌 4. Thí nghiệm 2: Phạm vi logits & Nhiệt độ

---

## 4.1. Hàm thực nghiệm

```python
def temperature_range_experiment(
    ranges=[0.4, 1, 5],
    temperatures=[0.5, 1.0, 3.0],
    n_points=100,
    outlier=6.0
):
    """
    Temperature & Logit Range Experiment

    Returns:
        results (dict): Nested results
    """

    results = {}

    for r in ranges:

        # Generate logits
        x = torch.linspace(-r, r, n_points)

        # Append outlier
        out = torch.tensor([outlier])
        logits = torch.cat([x, out])

        results[r] = {}

        for t in temperatures:

            probs = softmax_t(logits, t)

            results[r][t] = {
                "logits": logits,
                "probs": probs
            }

    return results
```

---

## 4.2. Chạy thí nghiệm

```python
results = temperature_range_experiment()
```

---

## 4.3. Vẽ kết quả

### Zoom-in (xác suất nhỏ)

```python
def plot_zoomed(results):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, r in enumerate(results):

        ax = axes[idx]

        for t in results[r]:

            data = results[r][t]

            logits = data["logits"][:-1]
            probs = data["probs"][:-1]

            ax.scatter(
                logits.numpy(),
                probs.numpy(),
                label=f"T={t}",
                s=20
            )

        ax.set_title(f"Range [-{r}, {r}]")
        ax.set_ylim(0, 0.03)
        ax.set_xlabel("Logits")
        ax.set_ylabel("Probability")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
```

---

### Zoom-out (toàn bộ phân phối)

```python
def plot_full(results):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, r in enumerate(results):

        ax = axes[idx]

        for t in results[r]:

            data = results[r][t]

            logits = data["logits"]
            probs = data["probs"]

            ax.scatter(
                logits.numpy(),
                probs.numpy(),
                label=f"T={t}",
                s=20
            )

        ax.set_title(f"Range [-{r}, {r}]")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Logits")
        ax.set_ylabel("Probability")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
```

---

## 4.4. Vẽ đồ thị

```python
plot_zoomed(results)
plot_full(results)
```

---

# 📌 5. Pipeline tổng hợp (chuẩn nghiên cứu)

```python
def run_full_pipeline():

    print("Running Iterative Softmax...")
    probs, stds, x = iterative_softmax_experiment()

    print("Running Temperature Experiment...")
    results = temperature_range_experiment()

    plot_zoomed(results)
    plot_full(results)

    return probs, stds, results
```

---

```python
probs, stds, results = run_full_pipeline()
```

---

# 📌 6. Đặc điểm chuẩn học thuật của mã này

Mã trên đáp ứng tiêu chuẩn:

✅ Numerical stability
✅ Reproducibility
✅ Modular design
✅ Research-friendly
✅ Dễ mở rộng

Áp dụng được cho:

* Paper supplementary
* Replication study
* Thesis
* Research demo

---

# 📌 7. Gợi ý nâng cao (nếu bạn làm nghiên cứu sâu hơn)

Bạn có thể mở rộng thêm:

### 🔹 Phân tích độ phức tạp

```python
# O(N) per softmax
# O(KN) for iterative
```

### 🔹 Seed cố định

```python
torch.manual_seed(42)
np.random.seed(42)
```

### 🔹 GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
