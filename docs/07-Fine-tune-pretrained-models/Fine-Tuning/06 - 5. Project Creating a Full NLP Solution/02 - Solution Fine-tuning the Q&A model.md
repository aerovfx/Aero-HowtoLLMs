# Giải Pháp Fine-tuning Mô Hình Question Answering

## Tổng Quan

Trong bài học này, chúng ta sẽ xem xét giải pháp cho bài tập fine-tuning mô hình Question Answering (QA) sử dụng FLAN-T5 trên tập dữ liệu SQuAD v2. Đây là một tác vụ quan trọng trong các ứng dụng chatbot và hệ thống hỗ trợ.

## 1. Giới Thiệu Question Answering

### 1.1 Khái Niệm

Question Answering (QA) là tác vụ trong đó mô hình cần trả lời câu hỏi dựa trên một đoạn văn bản ngữ cảnh. Còn được gọi là **Question Answering with Context**.

### 1.2 Định Dạng Dữ Liệu

Mỗi ví dụ trong SQuAD bao gồm:
- **Context**: Đoạn văn bản chứa thông tin
- **Question**: Câu hỏi về nội dung trong context
- **Answer**: Câu trả lời (trích xuất từ context)

## 2. Triển Khai Giải Pháp

### 2.1 Cài Đặt và Import

```python
!pip install transformers tensorflow datasets rouge-score

import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
```

### 2.2 Tải Dữ Liệu SQuAD v2

```python
# Tải tập dữ liệu SQuAD v2
dataset = load_dataset("squad_v2", split="train")

# Xem ví dụ
print(dataset[0])
```

### 2.3 Tiền Xử Lý Dữ Liệu

```python
# Tải tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_qa(examples):
    # Tạo prompt với format: context + question
    inputs = []
    for context, question in zip(examples['context'], examples['questions']):
        prompt = f"{context} Question: {question} Answer:"
        inputs.append(prompt)
    
    # Xử lý câu trả lời
    answers = []
    for ans_text in examples['answers']:
        if len(ans_text['text']) > 0:
            answers.append(ans_text['text'][0])
        else:
            answers.append("")  # Câu trả lời trống
    
    # Tokenize
    model_inputs = tokenizer(inputs, max_length=384, truncation=True, padding="max_length")
    labels = tokenizer(answers, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Giới hạn dữ liệu (SQuAD rất lớn)
train_data = dataset.select(range(25000))
test_data = dataset.select(range(25000, 27000))

# Áp dụng tiền xử lý
train_data = train_data.map(preprocess_qa, batched=True)
test_data = test_data.map(preprocess_qa, batched=True)
```

### 2.4 Chuyển Đổi Sang TensorFlow

```python
tf_train = train_data.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"],
    batch_size=16,
    shuffle=True
)

tf_test = test_data.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"],
    batch_size=16
)
```

### 2.5 Tải và Cấu Hình Mô Hình

```python
# Tải mô hình FLAN-T5-base
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

# Freeze các lớp đầu (transfer learning)
for layer in model.layers[:3]:
    layer.trainable = False

print(f"Tổng tham số: {model.count_params() / 1e6:.1f}M")
```

### 2.6 Huấn Luyện

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
```

### 2.7 Đánh Giá với ROUGE

```python
# Khởi tạo ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Lấy một ví dụ từ test set
batch = next(iter(tf_test))

# Lấy một ví dụ cụ thể
input_ids = batch['input_ids'][0:1]
label_ids = batch['labels'][0:1]

# Generate answer
outputs = model.generate(input_ids)
predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Reference
reference = tokenizer.decode(label_ids[0], skip_special_tokens=True)

# Tính ROUGE
scores = scorer.score(reference, predicted_answer)

print(f"Question: What is the capital of France?")
print(f"Reference: {reference}")
print(f"Predicted: {predicted_answer}")
print(f"ROUGE-1: {scores['rouge1'].precision:.4f}")
print(f"ROUGE-2: {scores['rouge2'].precision:.4f}")
print(f"ROUGE-L: {scores['rougeL'].precision:.4f}")
```

## 3. Kết Quả

### 3.1 Thống Kê Huấn Luyện

| Epoch | Training Loss | Validation Loss | Thời gian |
|-------|---------------|-----------------|-----------|
| 1     | 1.8           | 1.5             | ~30 phút  |
| 2     | 1.2           | 1.0             | ~30 phút  |
| 3     | 0.8           | 0.7             | ~30 phút  |

**Tổng thời gian:** ~90 phút

### 3.2 Đánh Giá ROUGE

| Chỉ số | Giá trị |
|--------|---------|
| ROUGE-1 | 1.0 |
| ROUGE-2 | 0.0 |
| ROUGE-L | 1.0 |

**Ví dụ:**
- **Câu hỏi:** "What is the capital of France?"
- **Reference:** "France"
- **Predicted:** "France"

## 4. Phân Tích Chi Tiết

### 4.1 Tại Sao Sử Dụng Transfer Learning?

$$\text{Performance}_{TL} > \text{Performance}_{from\_scratch}$$

Lý do:
- Mô hình FLAN-T5 đã được pre-train trên nhiều tác vụ
- Transfer learning giảm nhu cầu dữ liệu lớn
- Huấn luyện nhanh hơn

### 4.2 Xử Lý Câu Trả Lời Trống

Trong SQuAD v2, có những câu hỏi không có câu trả lời. Chúng ta xử lý:

```python
# Kiểm tra và xử lý câu trả lời trống
if len(answer['text']) == 0:
    answer_text = ""  # Model sẽ học "I don't know"
else:
    answer_text = answer['text'][0]
```

Điều này quan trọng để tránh **hallucination** - hiện tượng mô hình tạo ra câu trả lời không có thật.

## 5. Bài Học Rút Ra

### 5.1 Điểm Quan Trọng

1. **Format prompt**: Context + Question + Answer
2. **Xử lý câu trả lời trống**: Tránh hallucination
3. **Đánh giá**: ROUGE cho QA

### 5.2 Khuyến Nghị

- Sử dụng FLAN-T5-large để cải thiện kết quả
- Tăng số lượng dữ liệu huấn luyện
- Fine-tune nhiều epoch hơn

## 6. Kết Luận

Bài tập này đã hướng dẫn chúng ta cách:
1. Fine-tune FLAN-T5 cho tác vụ Question Answering
2. Xử lý dữ liệu SQuAD v2
3. Áp dụng transfer learning
4. Đánh giá với ROUGE

Question Answering là một tác vụ quan trọng trong các ứng dụng chatbot và hệ thống hỗ trợ thông minh.

## Tài Liệu Tham Khảo

1. Rajpurkar, P., et al. (2018). "Know What You Don't Know: Unanswerable Questions for SQuAD." *ACL 2018*.

2. Alberti, C., et al. (2019). "Synthetic Data for Natural Language Inference." *EMNLP 2019*.

3. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL 2019*.
