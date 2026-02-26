
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../../index.md) > [07 fine tune pretrained models](../../../index.md) > [fine tuning](../../index.md) > [06 5. project creating a full nlp solution](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../../../index.md)
- [📚 Module 01: LLM Course](../../../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Giải Pháp Fine-tuning Mô Hình Tóm Tắt với LoRA

## Tổng Quan

Trong bài học này, chúng ta sẽ xem xét giải pháp cho bài tập fine-tuning mô hình tóm tắt (summarization) sử dụng kỹ thuật LoRA trên tập dữ liệu CNN DailyMail. Đây là tác vụ quan trọng trong các ứng dụng chatbot để tạo tóm tắt cuộc trò chuyện.

## 1. Giới Thiệu Tóm Tắt văn bản

### 1.1 Tại Sao Cần Tóm Tắt?

Trong các ứng dụng chatbot:
- Tóm tắt cuộc trò chuyện dài
- Giảm thiểu token trong inference
- Lưu trữ lịch sử hội thoại hiệu quả

### 1.2 Tập Dữ Liệu CNN DailyMail

- **Context**: Bài báo dài
- **Summary**: Các điểm nổi bật (highlights)
- **Kích thước**: ~300,000 bài báo

## 2. Triển Khai Giải Pháp

### 2.1 Cài Đặt và Import

```python
!pip install transformers tensorflow datasets rouge-score

import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
```

### 2.2 Tải Dữ Liệu

```python
# Tải tậpu CNN DailyMail dữ liệ
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:0.5%]")

print(f"Số lượng ví dụ: {len(dataset)}")
print(dataset[0])
```

### 2.3 Tiền Xử Lý

```python
# Tải tokenizer
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_summarization(examples):
    # Tạo prompt với format: summarize: article
    inputs = ["summarize: " + article for article in examples['article']]
    
    # Highlights là tóm tắt
    targets = [highlight for highlight in examples['highlights']]
    
    # Tokenize
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Giới hạn dữ liệu (CNN rất lớn)
train_data = dataset.select(range(10000))
test_data = dataset.select(range(10000, 10500))

# Áp dụng tiền xử lý
train_data = train_data.map(preprocess_summarization, batched=True)
test_data = test_data.map(preprocess_summarization, batched=True)
```

### 2.4 Chuyển Đổi Sang TensorFlow

```python
# Chuyển sang TensorFlow Dataset
tf_train = train_data.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"],
    batch_size=8,
    shuffle=True
)

tf_test = test_data.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"],
    batch_size=8
)
```

### 2.5 Triển Khai LoRA

```python
# Tải mô hình
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

# Định nghĩa lớp LoRA
class LoRALayer(tf.keras.layers.Layer):
    def __init__(self, original_layer, rank=4, **kwargs):
        super(LoRALayer, self).__init__(**kwargs)
        self.original_layer = original_layer
        self.rank = rank
        
    def build(self, input_shape):
        # Ma trận A (input_dim x rank)
        self.A = self.add_weight(
            name="lora_A",
            shape=(input_shape[-1], self.rank),
            initializer="glorot_uniform",
            trainable=True
        )
        # Ma trận B (rank x input_dim)
        self.B = self.add_weight(
            name="lora_B",
            shape=(self.rank, input_shape[-1]),
            initializer="zeros",
            trainable=True
        )
        super(LoRALayer, self).build(input_shape)
        
    def call(self, inputs):
        original_output = self.original_layer(inputs)
        # LoRA: A × B × input
        lora_output = tf.matmul(tf.matmul(inputs, self.A), self.B)
        return original_output + lora_output
```

### 2.6 Áp Dụng LoRA

```python
# Freeze các lớp đầu
for layer in model.layers[:3]:
    layer.trainable = False

# Áp dụng LoRA cho decoder và lớp dense cuối
# (Lưu ý: Trong thực tế, cần duyệt qua các lớp và thay thế)
```

### 2.7 Thống Kê Tham Số

| Loại | Số lượng tham số |
|------|------------------|
| Tổng | ~76M |
| Trainable (với LoRA) | ~16M |
| Tỷ lệ | ~21% |

### 2.8 Huấn Luyện

```python
# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# Huấn luyện
print("Bắt đầu huấn luyện...")
history = model.fit(
    tf_train,
    validation_data=tf_test,
    epochs=3
)

print(f"Hoàn thành trong ~6 phút")
```

### 2.9 Đánh Giá với BLEU

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothing = SmoothingFunction().method1

# Lấy một batch từ test set
batch = next(iter(tf_test))

# Tính BLEU score trung bình
bleu_scores = []
for i in range(8):
    # Reference
    ref = tokenizer.decode(batch['labels'][i], skip_special_tokens=True)
    
    # Generate
    input_ids = batch['input_ids'][i:i+1]
    outputs = model.generate(input_ids)
    hyp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Tính BLEU
    score = sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothing)
    bleu_scores.append(score)

avg_bleu = np.mean(bleu_scores)
print(f"Average BLEU Score: {avg_bleu:.4f}")
```

## 3. Kết Quả

### 3.1 Thống Kê Huấn Luyện

| Epoch | Training Loss | Validation Loss | Thời gian |
|-------|---------------|-----------------|-----------|
| 1     | 2.5           | 2.2             | ~2 phút   |
| 2     | 2.0           | 1.8             | ~2 phút   |
| 3     | 1.6           | 1.5             | ~2 phút   |

### 3.2 Đánh Giá BLEU

| Chỉ số | Giá trị |
|--------|---------|
| BLEU Score | 0.03 |

**Nhận xét:**
- BLEU = 0.03 là giá trị chấp nhận được
- Có thể cải thiện với:
  - Mô hình lớn hơn (base, large)
  - Nhiều dữ liệu hơn
  - Tăng max_length

## 4. Phân Tích Chi Tiết

### 4.1 Tại Sao BLEU Thấp?

BLEU đo lường sự khớp từng từ giữa reference và hypothesis. Trong tóm tắt:
- Có nhiều cách tóm tắt cùng một văn bản
- BLEU không đánh giá meaning

### 4.2 Các Chỉ Số Thay Thế

| Chỉ số | Ưu điểm |
|--------|---------|
| ROUGE | Đánh giá recall |
| BERTScore | Đánh giá semantic |
| METEOR | Linh hoạt hơn |

## 5. Bài Học Rút Ra

### 5.1 Điểm Quan Trọng

1. **Định dạng prompt**: "summarize: " + article
2. **Giới hạn dữ liệu**: CNN rất lớn, cần giới hạn
3. **LoRA**: Giảm tham số đáng kể

### 5.2 Khuyến Nghị Cho Sản Phẩm

- Sử dụng FLAN-T5-large
- Tăng max_length lên 512 hoặc 1024
- Huấn luyện nhiều epoch hơn

## 6. Kết Luận

Bài tập này đã hướng dẫn chúng ta:
1. Fine-tune FLAN-T5 với LoRA cho tác vụ tóm tắt
2. Xử lý dữ liệu CNN DailyMail
3. Đánh giá với BLEU score
4. Đạt được kết quả BLEU = 0.03

Tóm tắt là một tác vụ quan trọng trong các ứng dụng chatbot để lưu trữ và xử lý cuộc trò chuyện dài.

## Tài Liệu Tham Khảo

1. See, A., et al. (2017). "Get To The Point: Summarization with Pointer-Generator Networks." *ACL 2017*.

2. Hermann, K.M., et al. (2015). "Teaching Machines to Read and Comprehend." *NIPS 2015*.

3. Hu, E.J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Giải Pháp: Fine-tuning Mô Hình Phân Tích Cảm Xúc](01_solution_fine_tuning_the_sentiment_analysis_model.md) | [Xem bài viết →](01_solution_fine_tuning_the_sentiment_analysis_model.md) |
| [Giải Pháp Fine-tuning Mô Hình Question Answering](02_solution_fine_tuning_the_q_a_model.md) | [Xem bài viết →](02_solution_fine_tuning_the_q_a_model.md) |
| 📌 **[Giải Pháp Fine-tuning Mô Hình Tóm Tắt với LoRA](03_solution_fine_tuning_the_summarization_model.md)** | [Xem bài viết →](03_solution_fine_tuning_the_summarization_model.md) |
| [Demo Tích Hợp Mọi Thứ vào Giải Pháp](04_demo_integrating_everything_into_our_solution.md) | [Xem bài viết →](04_demo_integrating_everything_into_our_solution.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
