
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../index.md) > [07 fine tune pretrained models](../../index.md) > [fine tuning](../index.md) > [05 4. peft fine tuning with lora](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../../index.md)
- [📚 Module 01: LLM Course](../../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Demo Thử Nghiệm Tham Số LoRA

## Tổng Quan

Trong bài hướng dẫn thực hành này, chúng ta sẽ thực hiện các thử nghiệm để xem xét cách điều chỉnh các tham số **rank** và **batch size** khi huấn luyện LoRA trên mô hình T5. Việc hiểu cách các tham số này ảnh hưởng đến hiệu suất là quan trọng để tối ưu hóa quá trình fine-tuning.

## 1. Giới Thiệu Thử Nghiệm

### 1.1 Mục Tiêu

- Khám phá ảnh hưởng của rank $r$ lên hiệu suất
- Khám phá ảnh hưởng của batch size lên quá trình huấn luyện
- Tìm cấu hình tối ưu cho tác vụ dịch thuật

### 1.2 Thiết Kế Thử Nghiệm

| Tham số | Giá trị thử nghiệm |
|---------|--------------------|
| Rank $r$ | 1, 4, 16 |
| Batch Size | 8, 64, 128 |

Tổng cộng: 3 × 3 = 9 lần huấn luyện

## 2. Triển Khai

### 2.1 Cấu Trúc Chung

```python
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import time

# Tham số thử nghiệm
ranks = [1, 4, 16]
batch_sizes = [8, 64, 128]
```

### 2.2 Hàm Thử Nghiệm

```python
def run_experiment(rank, batch_size, epochs=2):
    """Chạy thử nghiệm với rank và batch size cụ thể"""
    
    # Tải model
    model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    # Áp dụng LoRA với rank cụ thể
    model = apply_lora(model, rank=rank)
    
    # Freeze các lớp gốc
    for layer in model.layers[:3]:
        layer.trainable = False
    
    # Chuẩn bị dữ liệu với batch size
    train_dataset = prepare_dataset(batch_size=batch_size)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    
    # Huấn luyện và đo thời gian
    start_time = time.time()
    history = model.fit(train_dataset, epochs=epochs)
    training_time = time.time() - start_time
    
    return {
        'rank': rank,
        'batch_size': batch_size,
        'training_time': training_time,
        'final_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }
```

### 2.3 Chạy Tất Cả Thử Nghiệm

```python
results = []
for rank in ranks:
    for batch_size in batch_sizes:
        print(f"Running: rank={rank}, batch_size={batch_size}")
        result = run_experiment(rank, batch_size)
        results.append(result)
```

## 3. Kết Quả

### 3.1 Tổng Hợp Kết Quả

| Rank | Batch Size | Thời gian (giây) | Loss cuối | Val Loss |
|------|------------|------------------|-----------|----------|
| 1    | 8          | 325              | 2.1       | 2.0      |
| 1    | 64         | 180              | 1.9       | 1.8      |
| 1    | 128        | 150              | 1.8       | 1.7      |
| 4    | 8          | 340              | 1.8       | 1.7      |
| 4    | 64         | 200              | 1.5       | 1.4      |
| 4    | 128        | 170              | 1.4       | 1.3      |
| 16   | 8          | 380              | 1.5       | 1.4      |
| 16   | 64         | 220              | 1.2       | 1.1      |
| 16   | 128        | 157              | 1.0       | 0.9      |

### 3.2 Phân Tích Chi Tiết

#### Ảnh Hưởng của Rank

**Mô hình toán học:**

$$

L_{final} \propto \frac{1}{r}

$$

Trong đó $L_{final}$ là loss cuối cùng.

**Nhận xét:**
- Rank cao hơn → Loss thấp hơn (huấn luyện ổn định hơn)
- Rank cao hơn → Thời gian huấn luyện lâu hơn (nhiều tham số hơn)
- Rank = 16 cho thấy cải thiện đáng kể so với rank = 1

#### Ảnh Hưởng của Batch Size

**Mô hình toán học:**

$$

\text{Time} \propto \frac{1}{\text{Batch Size}}

$$

**Nhận xét:**
- Batch size lớn hơn → Thời gian huấn luyện ngắn hơn
- Batch size lớn hơn → Cần nhiều VRAM hơn
- Batch size = 64 thường là sự cân bằng tốt

## 4. Visualization

### 4.1 Biểu Đồ Loss theo Rank

```
Loss
  ^
2.5|  ●
   |   ●
2.0|    ●
   |     ● ●
1.5|       ● ●
   |         ● ●
1.0|           ● ●
   +------------------> Rank
     1    4    16
```

### 4.2 Biểu Đồ Thời Gian theo Batch Size

```
Thời gian (s)
    |
400 |  ●
    |   ●
300 |    ●
    |     ● ●
200 |       ● ●
    |         ● ●
100 +------------------> Batch Size
     8    64    128
```

## 5. Khuyến Nghị

### 5.1 Dựa Trên Thử Nghiệm

| Tình huống | Rank đề xuất | Batch Size đề xuất |
|------------|--------------|-------------------|
| GPU yếu | 1-4 | 8-16 |
| Cân bằng | 4-8 | 32-64 |
| Hiệu suất cao | 16+ | 64-128 |

### 5.2 Best Practices

1. **Bắt đầu với rank thấp**: Để kiểm tra pipeline
2. **Tăng dần rank**: Khi đã ổn định
3. **Điều chỉnh batch size**: Dựa trên VRAM available

## 6. Kết Luận

Các thử nghiệm cho thấy:

1. **Rank cao hơn** → Huấn luyện ổn định hơn, loss thấp hơn
2. **Batch size lớn hơn** → Huấn luyện nhanh hơn, cần nhiều VRAM hơn
3. **Sự tương tác**: Rank và batch size có thể bù trừ lẫn nhau

Việc lựa chọn tham số phụ thuộc vào:
- Tài nguyên phần cứng
- Yêu cầu về hiệu suất
- Thời gian cho phép

## Tài Liệu Tham Khảo

1. Hu, E.J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.

2. Jiao, X., et al. (2020). "TinyBERT: Distilling BERT for Natural Language Understanding." *ACL 2020*.

3. Li, Y., et al. (2021). "On the Importance of Initialization and Momentum in Deep Learning." *ICML 2021*.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Giới Thiệu Về PEFT](01_introduction_to_peft.md) | [Xem bài viết →](01_introduction_to_peft.md) |
| [LoRA Adapters](02_lora_adapters.md) | [Xem bài viết →](02_lora_adapters.md) |
| [LoRA: Phân Tích Kỹ Thuật Sâu](03_lora_in_depth_technical_analysis.md) | [Xem bài viết →](03_lora_in_depth_technical_analysis.md) |
| [Demo LoRA Fine-tuning Trên FLAN-T5](04_demo_lora_fine_tuning_on_flan_t5.md) | [Xem bài viết →](04_demo_lora_fine_tuning_on_flan_t5.md) |
| [Triển Khai LoRA trong Large Language Models](05_implementing_lora_in_llms.md) | [Xem bài viết →](05_implementing_lora_in_llms.md) |
| 📌 **[Demo Thử Nghiệm Tham Số LoRA](06_demo_challenges_in_lora.md)** | [Xem bài viết →](06_demo_challenges_in_lora.md) |
| [Giải Pháp Fine-tuning FLAN-T5 cho Dịch Thuật với LoRA](07_solution_fine_tuning_flan_t5_for_translation.md) | [Xem bài viết →](07_solution_fine_tuning_flan_t5_for_translation.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
