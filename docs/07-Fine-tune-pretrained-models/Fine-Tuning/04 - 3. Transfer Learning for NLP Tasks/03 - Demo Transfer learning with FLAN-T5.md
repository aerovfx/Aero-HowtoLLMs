# Demo Transfer Learning với FLAN-T5

## Tổng Quan

Trong bài hướng dẫn thực hành này, chúng ta sẽ thực hiện transfer learning với mô hình FLAN-T5 để dịch thuật từ tiếng Anh sang tiếng Tây Ban Nha trên tập dữ liệu OPUS-100. Đây là một trong những tác vụ phổ biến nhất trong xử lý ngôn ngữ tự nhiên.

## 1. Giới Thiệu Transfer Learning

### 1.1 Khái Niệm

Transfer learning là kỹ thuật cho phép sử dụng kiến thức từ một tác vụ đã được huấn luyện để giải quyết một tác vụ mới có liên quan. Trong ngữ cảnh của LLMs:

- Mô hình được pre-train trên một lượng lớn dữ liệu văn bản
- Sau đó fine-tune cho một tác vụ cụ thể
- Kiến thức từ pre-training được "chuyển giao" sang tác vụ mới

### 1.2 Lợi Ích

$$\text{Efficiency} \propto \frac{\text{Pre-trained Knowledge}}{\text{New Task Data}}$$

- Giảm thời gian huấn luyện
- Giảm nhu cầu dữ liệu có nhãn
- Cải thiện hiệu suất trên các tác vụ ít dữ liệu

## 2. Triển Khai Chi Tiết

### 2.1 Cài Đặt và Tải Mô Hình

```python
# Cài đặt thư viện
!pip install transformers tensorflow datasets

# Tải tokenizer và mô hình
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
```

### 2.2 Tải Dữ Liệu

```python
from datasets import load_dataset

# Tải tập dữ liệu OPUS-100 (Anh - Tây Ban Nha)
dataset = load_dataset("Helsinki-NLP/opus-100", "en-es")
```

### 2.3 Tiền Xử Lý Dữ Liệu

```python
# Tạo prompt cho tác vụ dịch thuật
def preprocess_function(examples):
    # Thêm prefix "translate English to Spanish"
    inputs = ["translate English to Spanish: " + ex for ex in examples["en"]]
    targets = [ex for ex in examples["es"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    # Tokenize targets
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs
```

### 2.4 Chuyển Đổi Sang TensorFlow Dataset

```python
# Chuyển đổi sang TensorFlow
tf_train = train_dataset.to_tf_dataset(
    columns=["input_ids", "decoder_input_ids", "attention_mask"],
    label_cols=["labels"],
    batch_size=64,
    shuffle=True
)
```

## 3. Transfer Learning trong Thực Tế

### 3.1 Freeze Các Lớp

```python
# Freeze embedding, encoder, decoder
for layer in model.layers[:3]:
    layer.trainable = False

# Chỉ train lớp dense cuối
model.summary()
```

**Kết quả:**
- Tổng tham số: ~247 triệu
- Tham số không train: ~222 triệu
- Tham số trainable: ~24 triệu

### 3.2 Huấn Luyện

```python
# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# Huấn luyện
model.fit(
    tf_train,
    validation_data=tf_test,
    epochs=3
)
```

### 3.3 Đánh Giá

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 2.5          | 1.8             |
| 2     | 1.5          | 1.2             |
| 3     | 0.77         | 0.9             |

**Nhận xét:**
- Loss giảm đều qua các epoch
- Validation loss không tăng (không overfitting)
- Mô hình học tốt tác vụ dịch thuật

## 4. Dịch Thuật Với Mô Hình Đã Fine-tune

### 4.1 Inference

```python
def translate(text):
    # Tạo prompt
    prompt = f"translate English to Spanish: {text}"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="tf")
    
    # Generate
    outputs = model.generate(**inputs, max_length=128)
    
    # Decode
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ví dụ
input_text = "Bird, you don't have to be super brave all the time."
translation = translate(input_text)
print(f"Input: {input_text}")
print(f"Translation: {translation}")
```

### 4.2 Kết Quả

**Input:** "Bird, you don't have to be super brave all the time."

**Output:** "Pájaro, no tienes que ser tan valiente todo el tiempo."

## 5. Phân Tích Toán Học

### 5.1 Kiến Trúc Seq2Seq

Mô hình FLAN-T5 sử dụng kiến trúc Encoder-Decoder:

$$\text{Output} = \text{Decoder}(\text{Encoder}(X), Y_{<t})$$

Trong đó:
- $X$ là chuỗi đầu vào
- $Y_{<t}$ là các token đã được sinh ra trước đó

### 5.2 Transfer Learning Efficiency

Hiệu quả của transfer learning có thể được biểu diễn:

$$\eta_{TL} = \frac{||\theta^*_{new} - \theta_{pre}||}{||\theta_{new}||} \times 100\%$$

Với LoRA, $\eta_{TL}$ thường < 5%, cho thấy chỉ một phần nhỏ tham số cần được điều chỉnh.

## 6. Kết Luận

Trong bài hướng dẫn này, chúng ta đã:

1. Sử dụng FLAN-T5-base cho transfer learning
2. Fine-tune trên tập dữ liệu OPUS-100
3. Freeze các lớp đầu và chỉ train lớp cuối
4. Đạt được kết quả dịch thuật tốt

Transfer learning là một kỹ thuật mạnh mẽ cho phép tận dụng kiến thức từ các mô hình lớn cho các tác vụ cụ thể với chi phí tính toán hợp lý.

## Tài Liệu Tham Khảo

1. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR*, 21(140), 1-67.

2. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." *ACL 2018*.

3. Wei, J., et al. (2021). "Finetuned Language Models are Zero-Shot Learners." *ICLR 2022*.
