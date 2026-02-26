# Demo Đánh Giá Bản Dịch

## Tổng Quan

Trong bài hướng dẫn thực hành này, chúng ta sẽ mở rộng từ bài trước về transfer learning với FLAN-T5 để thực hiện đánh giá bản dịch. Chúng ta sẽ sử dụng hai chỉ số phổ biến: **ROUGE** và **BLEU** để đo lường chất lượng dịch thuật.

## 1. Giới Thiệu Các Chỉ Số Đánh Giá

### 1.1 Tại Sao Cần Đánh Giá?

Đánh giá tự động là cần thiết để:
- Đo lường hiệu suất mô hình
- So sánh các phương pháp khác nhau
- Tối ưu hóa hyperparameter

### 1.2 Các Chỉ Số Phổ Biến

| Chỉ số | Mô tả | Ứng dụng |
|--------|-------|-----------|
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation | Tóm tắt |
| **BLEU** | Bilingual Evaluation Understudy | Dịch thuật |

## 2. Triển Khai Chi Tiết

### 2.1 Cài Đặt Thư Viện

```python
!pip install rouge-score nltk

import nltk
nltk.download('punkt')
```

### 2.2 Tải Mô Hình Đã Huấn Luyện

```python
# Giả định mô hình đã được huấn luyện từ bài trước
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained("path/to/model")
```

### 2.3 Hàm Dịch Thuật

```python
def translate(text):
    prompt = f"translate English to Spanish: {text}"
    inputs = tokenizer(prompt, return_tensors="tf", max_length=128, truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 3. Tính Toán ROUGE Score

### 3.1 Giới Thiệu về ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) là một nhóm các chỉ số để đánh giá tóm tắt tự động. Các biến thể phổ biến:

- **ROUGE-1**: Đơn vị unigram
- **ROUGE-2**: Bigram
- **ROUGE-L**: Longest common subsequence

### 3.2 Công Thức Toán Học

**ROUGE-N:**

$$\text{ROUGE-N} = \frac{\sum_{s \in \text{Reference}} \sum_{\text{n-gram} \in s} \min(\text{Count}_{hypothesis}(n\text{-gram}), \text{Count}_{reference}(n\text{-gram}))}{\sum_{s \in \text{Reference}} \sum_{\text{n-gram} \in s} \text{Count}_{reference}(n\text{-gram})}$$

### 3.3 Triển Khai

```python
from rouge_score import rouge_scorer

# Khởi tạo ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Tính ROUGE cho một cặp dịch
def calculate_rouge(reference, hypothesis):
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1': scores['rouge1'].precision,
        'rouge2': scores['rouge2'].precision,
        'rougeL': scores['rougeL'].precision
    }
```

## 4. Tính Toán BLEU Score

### 4.1 Giới Thiệu về BLEU

BLEU (Bilingual Evaluation Understudy) đo lường sự tương đồng giữa bản dịch máy và bản dịch tham chiếu của con người.

### 4.2 Công Thức Toán Học

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Trong đó:
- $p_n$ là precision cho n-gram
- $w_n$ là trọng số (thường bằng 1/N)
- BP là brevity penalty

**Brevity Penalty:**

$$\text{BP} = \begin{cases} 1 & \text{nếu } c > r \\ e^{(1-r/c)} & \text{nếu } c \leq r \end{cases}$$

### 4.3 Triển Khai

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Sử dụng smoothing để xử lý câu ngắn
smoothing = SmoothingFunction().method1

def calculate_bleu(reference, hypothesis):
    # Tokenize
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    # Tính BLEU
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
    return score
```

## 5. Đánh Giá Trên Tập Dữ Liệu

### 5.1 Quy Trình

```python
# Lấy một batch từ test dataset
batch = next(iter(test_dataset))

# Lấy reference từ labels
references = tokenizer.decode(batch['labels'][0], skip_special_tokens=True)

# Dịch input
inputs = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
hypothesis = translate(inputs)

# Tính các chỉ số
rouge_scores = calculate_rouge(references, hypothesis)
bleu_score = calculate_bleu(references, hypothesis)

print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
print(f"BLEU: {bleu_score:.4f}")
```

### 5.2 Kết Quả Mẫu

**Ví dụ:**
- **Input:** "I was cleaning"
- **Reference:** "Estaba limpiando"
- **Hypothesis:** "Estaba limpiando"

| Chỉ số | Giá trị |
|--------|---------|
| ROUGE-1 | 1.0 |
| ROUGE-2 | 1.0 |
| ROUGE-L | 1.0 |
| BLEU | 0.4 |

**Nhận xét:**
- ROUGE = 1.0 cho thấy mọi từ trong hypothesis đều có trong reference
- BLEU = 0.4 là giá trị cao, cho thấy dịch thuật tốt

## 6. Phân Tích Chi Tiết

### 6.1 So Sánh Precision và Recall

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

Trong ngữ cảnh dịch thuật:
- **Precision**: Tỷ lệ từ đúng trong hypothesis / tổng từ trong hypothesis
- **Recall**: Tỷ lệ từ đúng trong hypothesis / tổng từ trong reference

### 6.2 Ưu và Nhược Điểm

| Chỉ số | Ưu điểm | Nhược điểm |
|--------|----------|------------|
| ROUGE | Đo lường recall, tốt cho tóm tắt | Không đánh giá ngữ pháp |
| BLEU | Phổ biến, dễ so sánh | Không đánh giá meaning |

## 7. Kết Luận

Trong bài hướng dẫn này, chúng ta đã:

1. Tìm hiểu về các chỉ số ROUGE và BLEU
2. Triển khai hàm tính toán các chỉ số
3. Đánh giá mô hình dịch thuật
4. Phân tích kết quả

Các chỉ số này cung cấp đánh giá tự động, nhưng cần kết hợp với đánh giá của con người để có đánh giá toàn diện.

## Tài Liệu Tham Khảo

1. Papineni, K., et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation." *ACL 2002*.

2. Lin, C.Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." *ACL 2004*.

3. Post, M. (2018). "A Call for Clarity in Reporting BLEU Scores." *WMT 2018*.
