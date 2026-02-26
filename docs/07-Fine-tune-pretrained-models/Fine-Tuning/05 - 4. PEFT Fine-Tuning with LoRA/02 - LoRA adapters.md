# LoRA Adapters

## Giới Thiệu

Hãy khám phá LoRA adapters, một tập hợp con mạnh mẽ của parameter-efficient fine-tuning, nơi chúng ta sẽ bắt đầu với tổng quan cấp cao sử dụng phép so sánh nấu ăn của chúng ta, sau đó đi sâu hơn vào chi tiết kỹ thuật.

Hãy tưởng tượng bạn có một công thức tuyệt vời. Bạn muốn cải thiện món ăn mà không cần thay đổi toàn bộ quy trình nấu nướng. Bạn mang đến một công cụ chuyên biệt như một microplane để bào vỏ chanh. Công cụ này tạo ra tác động lớn với nỗ lực tối thiểu.

Trong thế giới machine learning, LoRA adapters đóng vai trò tương tự.

## LoRA Là Gì?

LoRA viết tắt của Low-Rank Adaptation. Các adapters này được thiết kế để fine-tune các mô hình pre-trained một cách hiệu quả bằng cách tập trung vào một tập hợp nhỏ các tham số. Chúng đặc biệt hiệu quả khi bạn cần thích nghi một mô hình với các tác vụ mới với dữ liệu hạn chế.

## Cơ Sở Kỹ Thuật

### Ma Trận Trọng Số

Trong một lớp neural network điển hình, trọng số được biểu diễn bởi một ma trận lớn. Trong fine-tuning truyền thống, ma trận này được điều chỉnh để cải thiện hiệu suất mô hình. Tuy nhiên, quá trình này có thể tốn kém về tính toán và đòi hỏi nhiều dữ liệu.

### Giải Pháp LoRA

Với kích thước ma trận n = 512 và rank r = 1:
- Số tham số cần fine-tune trong LoRA: 512 × 1 × 2 = 1,024 tham số
- Số tham số trong ma trận gốc: 512² = 262,144 tham số
- **Giảm khoảng 256 lần!**

## Lợi Ích Của LoRA

So với GPT-3 175B fine-tuned với Adam, LoRA có thể:
- Giảm số lượng tham số có thể huấn luyện xuống **10,000 lần**
- Giảm yêu cầu bộ nhớ GPU xuống **3 lần**

LoRA thực hiện tương đương hoặc tốt hơn so với fine-tuning về chất lượng mô hình trên RoBERTa, DeBERTa, GPT-2, và GPT-3.

## Công Thức LoRA

LoRA đề xuất sử dụng phân rã hạng thấp:

$$W' = W + \Delta W = W + BA$$

Trong đó:
- $W$: Ma trận trọng số pre-trained (đông cứng)
- $B \in \mathbb{R}^{d \times r}$: Ma trận hạng thấp thứ nhất
- $A \in \mathbb{R}^{r \times d}$: Ma trận hạng thấp thứ hai
- $r \ll d$: Rank của ma trận thích nghi

## Kết Luận

Tóm lại, LoRA adapters là một tập hợp con của PEFT sử dụng các ma trận hạng thấp để fine-tune các mô hình một cách hiệu quả. Bằng cách cập nhật chỉ một số nhỏ các tham số, chúng cung cấp cải thiện đáng kể với chi phí tính toán tối thiểu. Điều này làm cho chúng trở thành một công cụ vô giá để thích nghi các mô hình pre-trained cho các tác vụ mới.

---

*Nguồn: File subtitle 02 - LoRA adapters.vtt*
