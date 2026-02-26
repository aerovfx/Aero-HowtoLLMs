
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../../index.md) > [07 fine tune pretrained models](../../../index.md) > [fine tuning](../../index.md) > [05 4. peft fine tuning with lora](../index.md)

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
# Demo LoRA Fine-tuning Trên FLAN-T5

## Giới Thiệu

Trong demo này, chúng ta sẽ đến với phần tuyệt vời của LoRA Fine-tuning. Chúng ta sẽ triển khai LoRA cuối cùng, một trong những kỹ thuật tiên tiến và thú vị nhất trong PEFT - parameter efficient fine-tuning.

Tính đến thời điểm ghi hình năm 2024, LoRA chưa đầy hai năm tuổi. Điều này có nghĩa là bạn sẽ học điều không chỉ là state-of-the-art mà còn sẽ thấy rằng việc triển khai nó sẽ hơi phức tạp vì chưa có các gói hỗ trợ LoRA cho Hugging Face, TensorFlow hoặc PyTorch một cách native như làm một cái gì đó như LoRA.apply(). Chúng ta chưa có điều đó. Đó là mức độ state-of-the-art của chúng ta ngay bây giờ.

## Cài Đặt Môi Trường

Để làm LoRA hiệu quả, gói duy nhất chúng ta cần thêm là tensorflow_addons, mà chúng ta sẽ sử dụng để thêm lower adapter.

## Tải Dữ Liệu

Chúng ta sử dụng tập dữ liệu dịch WMT16 từ tiếng Đức sang tiếng Anh.

## Xử Lý Văn Bản

Chúng ta tải tokenizer của mô hình:
- Đầu vào: Thêm prompt "translate English to German" cho phần tiếng Anh
- Target: Bản dịch tiếng Đức
- Sử dụng `return_tensors="tf"` để trả về TensorFlow tensors

## Triển Khai LoRA

### Tạo Lớp LoRA

```python
class LoraLayer(tf.keras.layers.Layer):
    def __init__(self, rank=8, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        
    def build(self, shape):
        # Tạo ma trận A và B
        self.A = self.add_weight(
            name="A",
            shape=(shape[0], self.rank),
            initializer="random_normal",
            trainable=True
        )
        self.B = self.add_weight(
            name="B", 
            shape=(self.rank, shape[1]),
            initializer="random_normal", 
            trainable=True
        )
        
    def call(self, inputs):
        # W' = W + A × B
        return tf.matmul(tf.matmul(inputs, self.A), self.B)
```

### Thay Thế Lớp Dense

Thay thế mỗi lớp Dense trong mô hình bằng lớp LoRA:
- Đặt lớp Dense gốc là non-trainable
- Thêm output của LoRA vào output của Dense gốc

## Kết Quả

Sau khi áp dụng LoRA:
- Tổng tham số: 247 triệu
- Tham số non-trainable: 222 triệu
- **Chỉ train 9% tổng tham số!**

### Hiệu Quả Tính Toán
- GPU RAM sử dụng: Giảm từ ~30GB xuống còn 8GB
- Thời gian huấn luyện mỗi epoch: Giảm đáng kể
- Có thể huấn luyện trên các instance GPU rẻ nhất (như GeForce)

## Kết Luận

Đây là cách triển khai LoRA. Điểm quan trọng là chúng ta chỉ huấn luyện 9% các tham số nhưng vẫn đạt được hiệu suất tốt. Điều này đặc biệt quan trọng nếu bạn muốn huấn luyện một tập dữ liệu lớn hoặc huấn luyện trong nhiều epoch.

---

*Nguồn: File subtitle 04 - Demo LoRA fine-tuning on FLAN-T5.vtt*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Giới Thiệu Về PEFT](01_introduction_to_peft.md) | [Xem bài viết →](01_introduction_to_peft.md) |
| [LoRA Adapters](02_lora_adapters.md) | [Xem bài viết →](02_lora_adapters.md) |
| [LoRA: Phân Tích Kỹ Thuật Sâu](03_lora_in_depth_technical_analysis.md) | [Xem bài viết →](03_lora_in_depth_technical_analysis.md) |
| 📌 **[Demo LoRA Fine-tuning Trên FLAN-T5](04_demo_lora_fine_tuning_on_flan_t5.md)** | [Xem bài viết →](04_demo_lora_fine_tuning_on_flan_t5.md) |
| [Triển Khai LoRA trong Large Language Models](05_implementing_lora_in_llms.md) | [Xem bài viết →](05_implementing_lora_in_llms.md) |
| [Demo Thử Nghiệm Tham Số LoRA](06_demo_challenges_in_lora.md) | [Xem bài viết →](06_demo_challenges_in_lora.md) |
| [Giải Pháp Fine-tuning FLAN-T5 cho Dịch Thuật với LoRA](07_solution_fine_tuning_flan_t5_for_translation.md) | [Xem bài viết →](07_solution_fine_tuning_flan_t5_for_translation.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
