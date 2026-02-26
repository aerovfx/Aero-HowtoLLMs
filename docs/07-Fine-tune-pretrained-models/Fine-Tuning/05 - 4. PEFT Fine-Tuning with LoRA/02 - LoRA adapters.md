
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../../index.md) > [07 Fine tune pretrained models](../../../index.md) > [Fine Tuning](../../index.md) > [05   4. PEFT Fine Tuning with LoRA](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../../../index.md)
- [📚 Module 01: LLM Course](../../../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
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
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Giới Thiệu Về PEFT](01 - Introduction to PEFT.md) | [Xem bài viết →](01 - Introduction to PEFT.md) |
| 📌 **[LoRA Adapters](02 - LoRA adapters.md)** | [Xem bài viết →](02 - LoRA adapters.md) |
| [LoRA: Phân Tích Kỹ Thuật Sâu](03 - LoRA in depth Technical analysis.md) | [Xem bài viết →](03 - LoRA in depth Technical analysis.md) |
| [Demo LoRA Fine-tuning Trên FLAN-T5](04 - Demo LoRA fine-tuning on FLAN-T5.md) | [Xem bài viết →](04 - Demo LoRA fine-tuning on FLAN-T5.md) |
| [Triển Khai LoRA trong Large Language Models](05 - Implementing LoRA in LLMs.md) | [Xem bài viết →](05 - Implementing LoRA in LLMs.md) |
| [Demo Thử Nghiệm Tham Số LoRA](06 - Demo Challenges in LoRA.md) | [Xem bài viết →](06 - Demo Challenges in LoRA.md) |
| [Giải Pháp Fine-tuning FLAN-T5 cho Dịch Thuật với LoRA](07 - Solution Fine-tuning FLAN-T5 for translation.md) | [Xem bài viết →](07 - Solution Fine-tuning FLAN-T5 for translation.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
