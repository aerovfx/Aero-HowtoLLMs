
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [16 interfering with attention](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Thử thách Lập trình: GPT-2 có thực sự thích Pizza Dứa? (Một nghiên cứu về can thiệp Attention chính xác)

## Tóm tắt (Abstract)
Báo cáo này trình bày một thử thách lập trình mang tính minh họa cao về kỹ thuật can thiệp Attention chính xác (Precision Intervention). Thay vì tác động lên toàn bộ chuỗi token, chúng ta cô lập việc "tắt tiếng" (silencing) một Head cụ thể tại một Layer cụ thể cho duy nhất một Token mục tiêu. Thí nghiệm sử dụng câu mẫu "Peanut butter and pineapple taste great on pizza" để kiểm chứng mối quan hệ ngữ nghĩa giữa "Dứa" (Pineapple) và "Pizza" thông qua Độ tương đồng Cosine. Kết quả thực nghiệm cho thấy các can thiệp siêu nhỏ này không đủ để làm lung lay các liên kết ngữ nghĩa mạnh mẽ đã được mô hình học từ Internet, qua đó thảo luận về các quy mô can thiệp khác nhau trong diễn giải học cơ học.

---

## 1. Mở Đầu (Introduction)
Dù các mô hình ngôn ngữ không có sở thích cá nhân, chúng mang theo những thiên kiến ngữ nghĩa (semantic priors) từ dữ liệu huấn luyện. Thử thách này đặt ra câu hỏi kỹ thuật: Liệu việc vô hiệu hóa một Attention Head tại đúng vị trí token "pineapple" có làm giảm sự liên kết của nó với "pizza" trong residual stream? Đây là một bước tiến từ can thiệp thô (toàn bộ chuỗi) sang can thiệp vi phẫu (surgical interaction).

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Phẫu thuật Token Đơn lẻ (Token-specific Silencing)
Sử dụng mô hình GPT-2 Large và cơ chế Forward Pre-hook được nâng cấp:
- **Biến điều khiển:** `layer_to_silence`, `head_to_silence`, và đặc biệt là `token_to_silence`.
- **Cơ chế:** Chỉ khi Forward Pass đang xử lý đúng token mục tiêu (ví dụ: token cuối của từ "pineapple") tại layer chỉ định, giá trị của head mục tiêu mới bị gán bằng 0.

### 2.2. Xử lý Token đa tầng (Handling Multi-token Words)
Các từ như "peanut butter" thường bị chia thành nhiều tokens (ví dụ: `peanut`, ` butter`).
- **Chiến lược:** Luôn phân tích token cuối cùng của một từ phức vì nó mang đầy đủ nhất ngữ cảnh tích lũy từ các token phía trước.

---

## 3. Kết Quả Thực Nghiệm (Results & Analysis)

### 3.1. Phân tích Baseline (Trạng thái sạch)
- **Quan sát:** Độ tương đồng Cosine giữa "Pineapple" và "Pizza" cao hơn đáng kể so với "Butter" và "Pizza" trên hầu hết các tầng.
- **Giải thích khoa học:** Điều này phản ánh tần suất xuất hiện cùng nhau của các khái niệm này trên Internet (tranh luận về pizza dứa cực kỳ phổ biến) hơn là một "sở thích" thực sự của AI.

### 3.2. Hiệu ứng của Can thiệp Siêu vi (Precision Ablation)
Khi thực hiện vòng lặp kép qua mọi layer và mọi head (36 layers $\times$ 20 heads = 720 kịch bản):
- **Phân tán:** Các điểm dữ liệu (cosine similarity sau can thiệp) cụm lại rất sát đường baseline của mô hình sạch.
- **Thống kê:** Phép thử T-test (đã hiệu chỉnh Bonferroni) cho thấy hầu như không có sự thay đổi nào có ý nghĩa thống kê. 
- **Ý nghĩa:** Mối liên kết giữa "Dứa" và "Pizza" trong GPT-2 Large bền vững đến mức một can thiệp đơn lẻ vào một Head tại một Token không đủ để tạo ra sự dịch chuyển đáng kể. Điều này tương phản với các can thiệp "búa tạ" (như scaling toàn bộ layer) đã thấy ở các bài học trước.

---

## 4. Thảo Luận: Các Cấp độ Tác động (Scales of Interference)
Nghiên cứu chỉ ra sự tồn tại của nhiều quy mô can thiệp:
1. **Macroscopic:** Tác động lên toàn bộ Block hoặc Hidden States (Dễ quan sát, nhưng khó cô lập nguyên nhân).
2. **Microscopic:** Can thiệp vào từng Head x Token (Chính xác tuyệt đối, nhưng hiệu ứng có thể quá nhỏ để đo lường bằng logit output).
Một hướng đi triển vọng là kết hợp các cấp độ này để kiểm chứng các giả thuyết tinh vi về cách mô hình mã hóa các khái niệm trừu tượng.

---

## 5. Kết Luận
Thử thách "Pizza Dứa" minh chứng rằng việc thực thi methodological đúng đắn quan trọng hơn việc diễn giải các kết quả mang tính suy diễn. GPT-2 Large duy trì một cấu trúc tri thức cực kỳ ổn định. Để thay đổi hành vi của nó trong các tác vụ quan trọng, chúng ta có thể cần những can thiệp đa điểm (multi-point interventions) thay vì chỉ nhắm vào một thành phần đơn độc.

---

## Tài liệu tham khảo (Citations)
1. Thử thách Precision Ablation trên GPT-2 Large dựa trên `aero_LLM_05_CodeChallenge Does GPT2 like pineapple pizza.md`. Phân tích sự bền vững của liên kết ngữ nghĩa Internet.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Cắt bỏ Attention Head và Dự đoán Token (Head Ablation and Token Prediction)](aero_llm_01_head_ablation_and_token_prediction.md) | [Xem bài viết →](aero_llm_01_head_ablation_and_token_prediction.md) |
| [Thử thách Lập trình: Dự đoán Token sau khi Cắt bỏ Head (Phần 1)](aero_llm_02_codechallenge_token_prediction_after_head_ablations_part_1_.md) | [Xem bài viết →](aero_llm_02_codechallenge_token_prediction_after_head_ablations_part_1_.md) |
| [Thử thách Lập trình: Dự đoán Token sau khi Cắt bỏ Head (Phần 2)](aero_llm_03_codechallenge_token_prediction_after_head_ablations_part_2_.md) | [Xem bài viết →](aero_llm_03_codechallenge_token_prediction_after_head_ablations_part_2_.md) |
| [Tác động của việc "Tắt tiếng" Head lên Độ tương đồng Cosine (Impact of Head-Silencing on Cosine Similarity)](aero_llm_04_impact_of_head_silencing_on_cosine_similarity.md) | [Xem bài viết →](aero_llm_04_impact_of_head_silencing_on_cosine_similarity.md) |
| 📌 **[Thử thách Lập trình: GPT-2 có thực sự thích Pizza Dứa? (Một nghiên cứu về can thiệp Attention chính xác)](aero_llm_05_codechallenge_does_gpt2_like_pineapple_pizza.md)** | [Xem bài viết →](aero_llm_05_codechallenge_does_gpt2_like_pineapple_pizza.md) |
| [Vá lỗi Attention Head trong tác vụ Nhận dạng Tân ngữ Gián tiếp (Attention Head Patching in IOI)](aero_llm_06_attention_head_patching_in_ioi.md) | [Xem bài viết →](aero_llm_06_attention_head_patching_in_ioi.md) |
| [Thử thách Lập trình: Vá lỗi Head và Token trong tác vụ IOI (Head and Token Patching in IOI)](aero_llm_07_codechallenge_head_and_token_patching_in_ioi.md) | [Xem bài viết →](aero_llm_07_codechallenge_head_and_token_patching_in_ioi.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
