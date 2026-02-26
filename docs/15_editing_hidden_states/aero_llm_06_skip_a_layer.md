
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [15 editing hidden states](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Bỏ qua một Tầng Transformer (Skip a Layer)

## Tóm tắt (Abstract)
Báo cáo này trình bày kỹ thuật bỏ qua hoàn toàn một Transformer Block trong residual stream của mô hình ngôn ngữ lớn (LLM). Bằng cách sử dụng cơ chế Forward Hook cực kỳ đơn giản để gán trực tiếp giá trị đầu vào (input) làm đầu ra (output), chúng ta có thể làm vô hiệu hóa mọi phép tính toán (Attention và MLP) bên trong tầng đó. Nghiên cứu thực hiện kiểm chứng thông qua chỉ số chuẩn ma trận (Matrix Norm), xác nhận sự triệt tiêu biến đổi tín hiệu tại tầng mục tiêu. Mặc dù đây là một kỹ thuật can thiệp thô (Ablation), nó giúp củng cố hiểu biết về luồng dữ liệu liên tục giữa các khối Transformer.

---

## 1. Mở Đầu (Introduction)
Trong kiến trúc Transformer, mỗi khối tính toán đóng vai trò tinh chỉnh các vector Embeddings từ tầng trước đó. Thông thường, đầu ra của khối $T$ là đầu vào của khối $T+1$. Thí nghiệm này đặt mục tiêu tạo ra một "đường tắt nhân quả" (Causal shortcut), nơi khối $T+1$ vẫn thực hiện tính toán nhưng kết quả của nó bị ghi đè hoàn toàn bởi giá trị nguyên bản của khối $T$. Điều này tương đương với việc "cắt bỏ" một phần bộ não của mô hình để quan sát sự đứt gãy luồng thông tin.

---

## 2. Thiết Lập Kỹ Thuật (Methodology)

### 2.1. Hàm Hook Tối Giản (The Minimalist Hook)
Sự can thiệp được thực hiện thông qua một hàm Hook không chứa logic phức tạp:
```python
def skip_layer_hook(module, input, output):
    return input
- **Cơ chế:** Hàm này bỏ qua tham số `output` (vốn chứa các kết quả tính toán của Attention/MLP) và trả về chính tham số `input`. Kết quả là khối tiếp theo sẽ nhận được dữ liệu y hệt như khối trước đó, như thể khối hiện tại chưa bao giờ tồn tại.

### 2.2. Chỉ số Kiểm chứng (Verification Metric)
Để xác nhận tầng đã bị bỏ qua, chúng ta đo lường chuẩn Frobenius của hiệu số Hidden States giữa các tầng liên tiếp:

$$

$$

\Delta_{norm} = \|\mathbf{H}_{L} - \mathbf{H}_{L-1}\|_F

$$

$$

Nếu \Delta_{norm} = 0 tại tầng L, điều đó có nghĩa là vector không hề thay đổi khi đi qua Transformer Block đó.