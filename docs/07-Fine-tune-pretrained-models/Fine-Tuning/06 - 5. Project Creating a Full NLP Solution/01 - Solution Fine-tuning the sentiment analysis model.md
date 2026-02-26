# Giải Pháp: Fine-tuning Mô Hình Phân Tích Cảm Xúc

## Giới Thiệu

Chào mừng đến với giải pháp fine-tuning mô hình phân tích cảm xúc. Trong phần này, chúng ta sẽ xây dựng các mô hình LLMs mà chúng ta sẽ cắm vào cuối cùng vào một giải pháp chatbot đầy đủ.

## Tập Dữ Liệu

Chúng ta sử dụng tập dữ liệu SST2 từ Stanford NLP Lab, là nguồn mở.

## Tokenization

```python
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize dữ liệu
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        padding='max_length',
        truncation=True,
        max_length=128
    )
```

## Tạo TensorFlow Dataset

```python
# Tạo dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Chuyển thành TensorFlow dataset
tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"],
    batch_size=64,
    shuffle=True
)
```

## Tải Mô Hình

```python
from transformers import TFDistilBertForSequenceClassification

model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2  # Positive/Negative
)

# Đông cứng base model để transfer learning
model.distilbert.trainable = False
```

## Huấn Luyện

```python
# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Huấn luyện
model.fit(tf_train_dataset, epochs=3)
```

## Kết Quả

- **Training Accuracy:** 84%
- **Validation Accuracy:** 83.3%
- **Loss:** Giảm qua mỗi epoch
- **Thời gian huấn luyện:** ~4 phút

Độ chính xác validation gần như bằng với training accuracy, có nghĩa là mô hình không bị overfit.

## Lưu Mô Hình

```python
model.save_pretrained('sentiment_model')
```

Bây giờ bạn có thư mục chứa tất cả cấu hình model và weight.

## Kết Luận

Chúng ta đã thành công trong việc transfer learning trên DistilBERT, trên tập dữ liệu SST2 để thực hiện phân tích cảm xúc. Điều quan trọng là nhớ rằng điều này có thể được áp dụng cho bất kỳ tác vụ phân loại nào - từ phân loại tin tức đến phân loại intent trong chatbot.

---

*Nguồn: File subtitle 01 - Solution Fine-tuning the sentiment analysis model.vtt*
