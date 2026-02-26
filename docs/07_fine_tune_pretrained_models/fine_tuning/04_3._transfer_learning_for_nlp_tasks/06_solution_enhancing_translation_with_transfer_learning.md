
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../index.md) > [07 fine tune pretrained models](../../index.md) > [fine tuning](../index.md) > [04 3. transfer learning for nlp tasks](index.md)

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
# Giải Pháp Nâng Cao Dịch Thuật với Transfer Learning

## Tổng Quan

Trong bài học này, chúng ta sẽ xem xét giải pháp cho bài tập nâng cao về transfer learning trong dịch thuật. Bài tập yêu cầu sử dụng tập dữ liệu WMT16 để dịch từ tiếng Đức sang tiếng Anh, một tác vụ khó hơn so với các bài tập trước.

## 1. Giới Thiệu Bài Toán

### 1.1 Mục Tiêu

- Sử dụng tập dữ liệu WMT16
- Dịch từ tiếng Đức sang tiếng Anh
- Áp dụng transfer learning với FLAN-T5

### 1.2 Thách Thức

- Tập dữ liệu mới (chưa quen thuộc)
- Ngôn ngữ nguồn khác với các bài tập trước
- Cần xử lý định dạng dữ liệu đặc biệt

## 2. Triển Khai Giải Pháp

### 2.1 Bước 1: Cài Đặt và Tải Dữ Liệu

```python
# Cài đặt thư viện
!pip install transformers tensorflow datasets

# Tải tập dữ liệu WMT16 (Đức - Anh)
from datasets import load_dataset

$$
dataset = load_dataset("wmt16", "de-en", split="train[:1%]")
$$

### 2.2 Bước 2: Tiền Xử Lý

```python
from transformers import AutoTokenizer

# Tải tokenizer

$$
model_name = "google/flan-t5-base"
$$

$$
tokenizer = AutoTokenizer.from_pretrained(model_name)
$$

def preprocess_function(examples):
    # Lấy câu tiếng Đức và tiếng Anh
    inputs = ["translate German to English: " + ex['de'] for ex in examples['translation']]
    targets = [ex['en'] for ex in examples['translation']]
    
    # Tokenize

$$
model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
$$

$$
labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
$$

    
$$
model_inputs["labels"] = labels["input_ids"]
$$

    
    return model_inputs

# Áp dụng tiền xử lý

$$
processed_dataset = dataset.map(preprocess_function, batched=True)
$$

### 2.3 Bước 3: Tạo TensorFlow Dataset

```python
# Chuyển đổi sang TensorFlow

$$
tf_train = processed_dataset["train"].to_tf_dataset(
$$

$$
columns=["input_ids", "decoder_input_ids", "attention_mask"],
$$

$$
label_cols=["labels"],
$$

$$
batch_size=32,
$$

$$
shuffle=True
$$

)

### 2.4 Bước 4: Tải và Cấu Hình Mô Hình

```python
from transformers import TFAutoModelForSeq2SeqLM

# Tải mô hình

$$
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
$$

# Freeze các lớp đầu
for layer in model.layers[:3]:

$$
layer.trainable = False
$$

### 2.5 Bước 5: Huấn Luyện

```python
# Compile
model.compile(

$$
optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
$$

$$
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
$$

)

# Huấn luyện
model.fit(
    tf_train,

$$
validation_data=tf_test,
$$

    epochs=3
)

## 3. Đánh Giá Kết Quả

### 3.1 Tính BLEU Score

```python
from nltk.translate.bleu_score import sentence_bleu

# Lấy một batch từ test set

$$
batch = next(iter(test_dataset))
$$

# Dịch và đánh giá

$$
bleu_scores = []
$$

for i in range(batch_size):
    # Decode reference

$$
reference = tokenizer.decode(batch['labels'][i], skip_special_tokens=True)
$$

    
    # Generate translation

$$
inputs = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
$$

$$
outputs = model.generate(tokenizer(inputs, return_tensors="tf")["input_ids"])
$$

$$
hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)
$$

    
    # Tính BLEU

$$
score = sentence_bleu([reference.split()], hypothesis.split())
$$

    bleu_scores.append(score)

# Trung bình BLEU

$$
avg_bleu = sum(bleu_scores) / len(bleu_scores)
$$

print(f"Average BLEU Score: {avg_bleu:.4f}")

### 3.2 Kết Quả

| Chỉ số | Giá trị |
|--------|---------|
| BLEU Score | 0.12 |

**Nhận xét:**
- BLEU = 0.12 cho thấy mô hình đã học được cơ bản
- Giá trị này có thể được cải thiện với:
  - Nhiều dữ liệu hơn
  - Nhiều epoch hơn
  - Mô hình lớn hơn

## 4. Phân Tích Chi Tiết

### 4.1 Transfer Learning cho Ngôn Ngữ Mới

Một điểm quan trọng trong bài tập này là chúng ta đang dịch từ tiếng Đức sang tiếng Anh - một ngôn ngữ mà FLAN-T5 không được huấn luyện trực tiếp. Điều này thể hiện:

1. **Khả năng tổng quát hóa** của mô hình pre-trained
2. **Transfer learning** hoạt động ngay cả với ngôn ngữ chưa từng thấy
3. **Hạn chế** của zero-shot cho tác vụ phức tạp

### 4.2 Mô Hình Toán Học

$$

$$

\text{BLEU}_{\text{avg}} = \frac{1}{N} $\sum$_{i=1}^{N} \text{BLEU}(ref_i, hyp_i)

$$

$$

Trong đó:
- $N$ là số lượng ví dụ
- $ref_i$ là bản dịch tham chiếu
- $hyp_i$ là bản dịch của mô hình

## 5. Bài Học Rút Ra

### 5.1 Điểm Quan Trọng

1. **Xử lý dữ liệu mới**: Cần hiểu định dạng dữ liệu trước khi xử lý
2. **Transfer learning**: Có thể áp dụng cho các ngôn ngữ khác nhau
3. **Đánh giá**: BLEU score cung cấp đánh giá định lượng

### 5.2 Khuyến Nghị

- Sử dụng nhiều dữ liệu hơn để cải thiện
- Thử nghiệm với các mô hình lớn hơn
- Điều chỉnh hyperparameters

## 6. Kết Luận

Bài tập này đã chứng minh khả năng của transfer learning trong việc:
- Mở rộng khả năng của mô hình sang ngôn ngữ mới
- Xử lý các tập dữ liệu mới
- Đánh giá hiệu suất bằng các chỉ số tiêu chuẩn

Với kết quả BLEU = 0.12, mô hình đã thể hiện khả năng học dịch thuật cơ bản và có thể được cải thiện thêm với nhiều tài nguyên hơn.

## Tài Liệu Tham Khảo

1. Bojar, O., et al. (2016). "Findings of the 2016 Conference on Machine Translation." *WMT 2016*.

2. Ott, M., et al. (2018). "Scaling Neural Machine Translation." *ACL 2018*.

3. Vaswani, A., et al. (2017). "Attention Is All You Need." *NIPS 2017*.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Transfer Learning Trong LLMs](01_transfer_learning_in_llms.md) | [Xem bài viết →](01_transfer_learning_in_llms.md) |
| [Chọn Mô Hình Cho Transfer Learning](02_choosing_models_for_transfer_learning.md) | [Xem bài viết →](02_choosing_models_for_transfer_learning.md) |
| [Demo Transfer Learning với FLAN-T5](03_demo_transfer_learning_with_flan_t5.md) | [Xem bài viết →](03_demo_transfer_learning_with_flan_t5.md) |
| [Đánh Giá Kết Quả Transfer Learning](04_evaluating_transfer_learning_outcomes.md) | [Xem bài viết →](04_evaluating_transfer_learning_outcomes.md) |
| [Demo Đánh Giá Bản Dịch](05_demo_evaluating_translations.md) | [Xem bài viết →](05_demo_evaluating_translations.md) |
| 📌 **[Giải Pháp Nâng Cao Dịch Thuật với Transfer Learning](06_solution_enhancing_translation_with_transfer_learning.md)** | [Xem bài viết →](06_solution_enhancing_translation_with_transfer_learning.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
