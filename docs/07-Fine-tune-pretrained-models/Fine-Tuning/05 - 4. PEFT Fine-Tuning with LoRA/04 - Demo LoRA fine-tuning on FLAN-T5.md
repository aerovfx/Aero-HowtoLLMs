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
