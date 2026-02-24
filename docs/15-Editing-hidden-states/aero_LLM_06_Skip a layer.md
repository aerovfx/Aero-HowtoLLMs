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
```
- **Cơ chế:** Hàm này bỏ qua tham số `output` (vốn chứa các kết quả tính toán của Attention/MLP) và trả về chính tham số `input`. Kết quả là khối tiếp theo sẽ nhận được dữ liệu y hệt như khối trước đó, như thể khối hiện tại chưa bao giờ tồn tại.

### 2.2. Chỉ số Kiểm chứng (Verification Metric)
Để xác nhận tầng đã bị bỏ qua, chúng ta đo lường chuẩn Frobenius của hiệu số Hidden States giữa các tầng liên tiếp:
$$ \Delta_{norm} = \|\mathbf{H}_{L} - \mathbf{H}_{L-1}\|_F $$
Nếu $\Delta_{norm} = 0$ tại tầng $L$, điều đó có nghĩa là vector không hề thay đổi khi đi qua Transformer Block đó.

---

## 3. Kết Quả Thực Nghiệm (Results & Analysis)

### 3.1. Triệt tiêu Trung thực (Faithful Ablation)
Kết quả đo lường trên GPT-2 với câu mẫu: "There is a lot of liquid water on planet Earth" cho thấy tại tầng được cài Hook (ví dụ Layer 5), giá trị chênh lệch Norm rơi về chính xác bằng 0. 
- **Lưu ý về Indexing:** Cần ghi nhớ sự khác biệt giữa chỉ số Transformer Block (bắt đầu từ 0) và chỉ số Hidden States (bắt đầu từ 1, do tầng 0 là Embedding nguyên bản). Sự lệch pha này phản ánh cấu trúc nội tại của thư viện Hugging Face.

### 3.2. Can thiệp Thô vs. Phẫu thuật (Chainsaw vs. Surgical Knife)
Báo cáo thừa nhận rằng việc cắt bỏ toàn bộ một tầng Transformer là một can thiệp mang tính "hủy diệt" diện rộng (Chainsaw ablation). Nó không tinh vi bằng việc vá hoạt hóa (Patching) hay can thiệp vào từng Attention Head cụ thể. Tuy nhiên, phương pháp này rất hữu ích để:
1. Kiểm tra tính dư thừa (Redundancy) của một số tầng cụ thể.
2. Xác nhận luồng residual stream hoạt động đúng như thiết kế logic.

---

## 4. Kết Luận
Việc "bỏ qua một tầng" minh chứng cho tính linh hoạt của Forward Hooks trong nghiên cứu Diễn giải cơ học nhân quả. Mặc dù hiếm khi được sử dụng như một giải pháp hiệu chỉnh mô hình trong thực tế, kỹ thuật này cung cấp một công cụ mạnh mẽ để hiểu về sự tích tụ thông tin dọc theo residual stream và vai trò không thể thay thế (hoặc có thể thay thế) của từng khối Transformer riêng lẻ.

---

## Tài liệu tham khảo (Citations)
1. Thử nghiệm Skip Layer trên GPT-2 dựa trên tài liệu `aero_LLM_06_Skip a layer.md`. Phân tích Norm difference để xác nhận sự triệt tiêu biến đổi tín hiệu.
