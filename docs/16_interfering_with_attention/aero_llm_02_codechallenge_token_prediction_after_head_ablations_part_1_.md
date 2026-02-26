
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [16 interfering with attention](index.md)

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
# Thử thách Lập trình: Dự đoán Token sau khi Cắt bỏ Head (Phần 1)

## Tóm tắt (Abstract)
Báo cáo này mở rộng nghiên cứu về can thiệp nhân quả lên Attention Heads bằng cách thực hiện quét toàn bộ các tầng và các đầu trong mô hình GPT-2 Small. Thử thách tập trung vào hai kịch bản thay thế: (1) Thay thế hoạt hóa của head bằng số không (Zeroing) và (2) Thay thế bằng giá trị trung bình thực nghiệm của chính head đó (Mean imputation). Sử dụng cơ chế `Forward Pre-hook` linh hoạt kết hợp với biến toàn cục để điều khiển thực nghiệm, nghiên cứu phân tích sự biến thiên của xác suất Softmax đối với các token mục tiêu. Kết quả cho thấy sự nhạy cảm của mô hình đối với can thiệp tại các head là rất khác nhau và không có quy luật không gian rõ rệt, đồng thời xác nhận tính bền vững của dự đoán từ khóa ngay cả khi cấu trúc attention bị xáo trộn cục bộ.

---

## 1. Mở Đầu (Introduction)
Tiếp nối các kỹ thuật cô lập Attention Head, thử thách này đặt ra ba mục tiêu chính: (1) Xây dựng hàm Hook động có khả năng can thiệp vào bất kỳ Head nào tại bất kỳ Layer nào; (2) So sánh tác động của việc triệt tiêu tín hiệu so với việc duy trì mức năng lượng trung bình; (3) Trực quan hóa bản đồ nhạy cảm (Sensitivity map) của toàn bộ mô hình thông qua Heatmap.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Cấu trúc Hook Linh hoạt và Biến Toàn cục
Để tránh việc hard-code cho từng tầng, chúng ta sử dụng một hàm Hook duy nhất được cài vào tất cả các Transformer Blocks. 
- **Cơ chế điều khiển:** Sử dụng các biến toàn cục (global variables) như `layer_to_ablate`, `head_to_ablate`, và `replace_zero` (Boolean).
- **Phạm vi cục bộ:** `if current_layer == layer_to_ablate: ...`. Điều này cho phép thực hiện vòng lặp kép (double for loop) qua 144 tổ hợp (12 layers $\times$ 12 heads) một cách tự động.

### 2.2. Kỹ thuật Imputation (Gán giá trị thay thế)
Nghiên cứu so sánh hai phương pháp:
- **Zero Imputation:** Gán toàn bộ tensor của head mục tiêu bằng 0.
- **Mean Imputation:** Tính toán trung bình cộng của toàn bộ các giá trị hoạt hóa trong head (`head.mean()`) và gán hằng số này cho mọi vị trí trong head đó. Phương pháp này giúp duy trì "mức độ hoạt động" trung bình nhưng triệt tiêu các biến thiên thông tin cụ thể.

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Bản đồ Nhạy cảm Softmax (Exercise 3)
Thông qua Heatmap so sánh sai lệch xác suất ($P_{manipulated} - P_{clean}$):
- **Token Mục tiêu ("Germany"):** Quan sát thấy sự thay đổi phân tán. Một số head khi bị cắt bỏ làm giảm mạnh xác suất đúng, trong khi một số khác lại làm tăng nhẹ. Không có mô hình "vùng chức năng" tập trung rõ rệt.
- **Token Đối chứng ("France"):** Tác động là tối thiểu và gần như bằng không trên toàn bộ bản đồ. Điều này chứng minh các head can thiệp có tính đặc hiệu cao đối với logic dẫn đến đáp án đúng.

### 3.2. Tính Bền Vững của Dự đoán (Argmax Analysis)
Trong 144 lần can thiệp, mô hình vẫn dự đoán đúng "Germany" trong 143 trường hợp. Chỉ có một trường hợp duy nhất mô hình chuyển sang dự đoán từ "the". Điều này củng cố quan điểm rằng LLM có cơ chế bù trừ lỗi cực kỳ mạnh mẽ dọc theo residual stream.

### 3.3. Phân tích Drift của Hoạt hóa (Exercise 4)
Việc trực quan hóa giá trị trung bình thực nghiệm (`observed_head_mean`) tiết lộ:
- Các giá trị trung bình thường rất nhỏ và tập trung quanh 0.
- Không có xu hướng tăng hay giảm (drift) rõ rệt khi đi sâu vào các tầng cuối.
- Khác biệt giữa kết quả Zero Imputation và Mean Imputation là không đáng kể về mặt định tính đối với tác vụ này.

---

## 4. Kết Luận
Thử thách này minh chứng rằng mặc dù Attention Heads là các đơn vị tính toán độc lập, vai trò của chúng trong việc lưu trữ tri thức thế giới được phân bổ theo mạng lưới phức tạp thay vì khu trú tại các tầng cụ thể. Sự tương đồng giữa việc gán bằng không và gán trung bình gợi ý rằng thông tin quan trọng nằm ở "các biến động" (fluctuations) xung quanh mức nền hơn là ở chính mức năng lượng tuyệt đối của Head.

---

## Tài liệu tham khảo (Citations)
1. Thử thách Head Ablated Quét toàn bộ layers trên GPT-2 Small dựa trên `aero_LLM_02_CodeChallenge Token prediction after head ablations (part 1).md`. Phân tích 144 kịch bản can thiệp nhân quả.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Cắt bỏ Attention Head và Dự đoán Token (Head Ablation and Token Prediction)](aero_llm_01_head_ablation_and_token_prediction.md) | [Xem bài viết →](aero_llm_01_head_ablation_and_token_prediction.md) |
| 📌 **[Thử thách Lập trình: Dự đoán Token sau khi Cắt bỏ Head (Phần 1)](aero_llm_02_codechallenge_token_prediction_after_head_ablations_part_1_.md)** | [Xem bài viết →](aero_llm_02_codechallenge_token_prediction_after_head_ablations_part_1_.md) |
| [Thử thách Lập trình: Dự đoán Token sau khi Cắt bỏ Head (Phần 2)](aero_llm_03_codechallenge_token_prediction_after_head_ablations_part_2_.md) | [Xem bài viết →](aero_llm_03_codechallenge_token_prediction_after_head_ablations_part_2_.md) |
| [Tác động của việc "Tắt tiếng" Head lên Độ tương đồng Cosine (Impact of Head-Silencing on Cosine Similarity)](aero_llm_04_impact_of_head_silencing_on_cosine_similarity.md) | [Xem bài viết →](aero_llm_04_impact_of_head_silencing_on_cosine_similarity.md) |
| [Thử thách Lập trình: GPT-2 có thực sự thích Pizza Dứa? (Một nghiên cứu về can thiệp Attention chính xác)](aero_llm_05_codechallenge_does_gpt2_like_pineapple_pizza.md) | [Xem bài viết →](aero_llm_05_codechallenge_does_gpt2_like_pineapple_pizza.md) |
| [Vá lỗi Attention Head trong tác vụ Nhận dạng Tân ngữ Gián tiếp (Attention Head Patching in IOI)](aero_llm_06_attention_head_patching_in_ioi.md) | [Xem bài viết →](aero_llm_06_attention_head_patching_in_ioi.md) |
| [Thử thách Lập trình: Vá lỗi Head và Token trong tác vụ IOI (Head and Token Patching in IOI)](aero_llm_07_codechallenge_head_and_token_patching_in_ioi.md) | [Xem bài viết →](aero_llm_07_codechallenge_head_and_token_patching_in_ioi.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
