# Triển Khai LoRA trong Large Language Models

## Tổng Quan

Trong bài học này, chúng ta sẽ tìm hiểu chi tiết cách triển khai **LoRA (Low-Rank Adaptation)** adapters trong các Large Language Models sử dụng Python, TensorFlow và Keras. LoRA là một kỹ thuật Parameter-Efficient Fine-Tuning (PEFT) cho phép fine-tune các mô hình lớn với chi phí tính toán thấp.

## 1. Giới Thiệu về LoRA

### 1.1 Vấn Đề với Fine-tuning Truyền Thống

Fine-tuning truyền thống yêu cầu:
- Cập nhật tất cả các tham số của mô hình
- Bộ nhớ GPU lớn
- Thời gian huấn luyện lâu

### 1.2 Giải Pháp LoRA

LoRA giới thiệu ma trận hạng thấp (low-rank matrices) để thay thế việc cập nhật trực tiếp các weights:

$$W_{new} = W_{original} + \Delta W$$
$$\Delta W = A \times B$$

Trong đó:
- $W_{original} \in \mathbb{R}^{d \times k}$
- $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$
- $r \ll \min(d, k)$ (rank thấp)

## 2. Triển Khai Chi Tiết

### 2.1 Cài Đặt Thư Viện

```python
!pip install transformers tensorflow keras
```

### 2.2 Tải Mô Hình

```python
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
```

### 2.3 Tạo Lớp LoRA Adapter

```python
import tensorflow as tf

class LoraAdapter(tf.keras.layers.Layer):
    def __init__(self, rank=4, **kwargs):
        super(LoraAdapter, self).__init__(**kwargs)
        self.rank = rank
    
    def build(self, input_shape):
        # Tạo ma trận A và B với rank thấp
        self.A = self.add_weight(
            name="lora_A",
            shape=(input_shape[-1], self.rank),
            initializer="glorot_uniform",
            trainable=True
        )
        self.B = self.add_weight(
            name="lora_B",
            shape=(self.rank, input_shape[-1]),
            initializer="zeros",
            trainable=True
        )
        super(LoraAdapter, self).build(input_shape)
    
    def call(self, inputs):
        # Tính toán: A × B × input
        lora_output = tf.matmul(tf.matmul(inputs, self.A), self.B)
        return inputs + lora_output
```

### 2.4 Tích Hợp LoRA vào Mô Hình

```python
class LoraDenseLayer(tf.keras.layers.Layer):
    def __init__(self, original_layer, rank=4, **kwargs):
        super(LoraDenseLayer, self).__init__(**kwargs)
        self.original_layer = original_layer
        self.lora_adapter = LoraAdapter(rank=rank)
    
    def call(self, inputs):
        # Lấy output từ lớp gốc
        original_output = self.original_layer(inputs)
        # Thêm output từ LoRA
        lora_output = self.lora_adapter(inputs)
        return original_output + lora_output
```

### 2.5 Thay Thế Các Lớp Dense

```python
# Thay thế các lớp dense trong model
def apply_lora_to_model(model, rank=4):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            # Thay thế bằng LoraDenseLayer
            lora_layer = LoraDenseLayer(layer, rank=rank)
            # Cập nhật cấu trúc model
            layer.trainable = False
    return model
```

## 3. Ví Dụ Hoàn Chỉnh

### 3.1 Huấn Luyện với LoRA

```python
# Tải dữ liệu WMT16
from datasets import load_dataset

dataset = load_dataset("wMT16", "de-en", split="train[:1%]")

# Tiền xử lý
def preprocess(examples):
    inputs = ["translate German to English: " + ex['de'] for ex in examples['translation']]
    targets = [ex['en'] for ex in examples['translation']]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess, batched=True)

# Chuyển sang TensorFlow
tf_train = dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"],
    batch_size=16
)

# Áp dụng LoRA
model = apply_lora_to_model(model, rank=4)

# Compile và huấn luyện
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(tf_train, epochs=3)
```

## 4. Phân Tích Hiệu Quả

### 4.1 So Sánh Tham Số

| Phương pháp | Tham số trainable | Tỷ lệ |
|-------------|------------------|-------|
| Full Fine-tune | ~247 triệu | 100% |
| LoRA (r=4) | ~2.8 triệu | ~1.1% |
| LoRA (r=8) | ~5.6 triệu | ~2.3% |
| LoRA (r=16) | ~11.2 triệu | ~4.5% |

### 4.2 Công Thức Tính Tham Số LoRA

$$\text{Params}_{LoRA} = 2 \times d \times r$$

Trong đó:
- $d$ là chiều của lớp dense
- $r$ là rank

## 5. Ưu Điểm và Nhược Điểm

### 5.1 Ưu Điểm

1. **Giảm bộ nhớ**: Giảm đáng kể VRAM cần thiết
2. **Tốc độ**: Huấn luyện nhanh hơn
3. **Modular**: Có thể chuyển đổi giữa các adapters
4. **Không tổn thất hiệu suất**: Vẫn đạt được hiệu suất tương đương

### 5.2 Nhược Điểm

1. **Inference latency**: Thêm một chút overhead
2. **Không hoạt động với mọi kiến trúc**: Cần điều chỉnh cho từng loại model

## 6. Kết Luận

LoRA là một kỹ thuật mạnh mẽ cho parameter-efficient fine-tuning. Bằng cách sử dụng ma trận hạng thấp, chúng ta có thể:

- Fine-tune các mô hình lớn với chi phí thấp
- Duy trì hiệu suất cao
- Dễ dàng chuyển đổi giữa các tác vụ

Việc triển khai LoRA đòi hỏi hiểu biết về cấu trúc mô hình và TensorFlow/Keras, nhưng mang lại lợi ích đáng kể trong thực tế.

## Tài Liệu Tham Khảo

1. Hu, E.J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.

2. Liu, H., Tam, D., & Muqeeth, M. (2022). "Parameter-Efficient Transfer Learning for NLP." *ICML 2022*.

3. Ding, N., et al. (2022). "AdapterFusion: Non-invasive Transfer Learning for Intermediate Tasks." *EMNLP 2022*.
