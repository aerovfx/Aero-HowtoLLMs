
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

$$
Tổng cộng: 3 × 3 = 9 lần huấn luyện
$$

## 2. Triển Khai

### 2.1 Cấu Trúc Chung

```python
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import time

# Tham số thử nghiệm

$$
ranks = [1, 4, 16] batch_sizes = [8, 64, 128] ### 2.2 Hàm Thử Nghiệm ```python def run_experiment(rank, batch_size, epochs=2): """Chạy thử nghiệm với rank và batch size cụ thể""" # Tải model model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small") # Áp dụng LoRA với rank cụ thể model = apply_lora(model, rank=rank) # Freeze các lớp gốc for layer in model.layers[:3]: layer.trainable = False # Chuẩn bị dữ liệu với batch size train_dataset = prepare_dataset(batch_size=batch_size) # Compile model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
$$

loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    )
    
    # Huấn luyện và đo thời gian

start_time = time.time()

$$
history = model.fit(train_dataset, epochs=epochs)
$$

training_time = time.time() - start_time

    
    return {
        'rank': rank,
        'batch_size': batch_size,
        'training_time': training_time,
        'final_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }

### 2.3 Chạy Tất Cả Thử Nghiệm

```python

$$
results = []
$$

for rank in ranks:
    for batch_size in batch_sizes:

print(f"Running: rank={rank}, batch_size={batch_size}")

$$
result = run_experiment(rank, batch_size) results.append(result) ## 3. Kết Quả ### 3.1 Tổng Hợp Kết Quả | Rank | Batch Size | Thời gian (giây) | Loss cuối | Val Loss | |------|------------|------------------|-----------|----------| | 1    | 8          | 325              | 2.1       | 2.0      | | 1    | 64         | 180              | 1.9       | 1.8      | | 1    | 128        | 150              | 1.8       | 1.7      | | 4    | 8          | 340              | 1.8       | 1.7      | | 4    | 64         | 200              | 1.5       | 1.4      | | 4    | 128        | 170              | 1.4       | 1.3      | | 16   | 8          | 380              | 1.5       | 1.4      | | 16   | 64         | 220              | 1.2       | 1.1      | | 16   | 128        | 157              | 1.0       | 0.9      | ### 3.2 Phân Tích Chi Tiết #### Ảnh Hưởng của Rank **Mô hình toán học:** L_{final} \propto \frac{1}{r} Trong đó L_{final} là loss cuối cùng. **Nhận xét:** - Rank cao hơn → Loss thấp hơn (huấn luyện ổn định hơn) - Rank cao hơn → Thời gian huấn luyện lâu hơn (nhiều tham số hơn) - Rank = 16 cho thấy cải thiện đáng kể so với rank = 1 #### Ảnh Hưởng của Batch Size **Mô hình toán học:** \text{Time} \propto \frac{1}{\text{Batch Size}}
$$

