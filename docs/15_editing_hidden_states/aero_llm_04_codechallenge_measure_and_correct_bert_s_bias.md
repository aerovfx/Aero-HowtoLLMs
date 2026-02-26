
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
# Thử thách Lập trình: Đo lường và Hiệu chỉnh Định kiến Giới trong BERT

## Tóm tắt (Abstract)
Báo cáo này thực hiện một "cuộc phẫu thuật" nhân quả nhằm phát hiện, định lượng và triệt tiêu định kiến giới trong mô hình BERT Large. Thí nghiệm tập trung vào kịch bản dự đoán đại từ cho danh từ chỉ nghề nghiệp ("engineer"). Kết quả ban đầu xác nhận định kiến mạnh mẽ nghiêng về đại từ nam giới ("he"). Bằng phương pháp trộn vector (Vector Mixing) có trọng số giữa biểu diễn của từ bị che khuất ([MASK]) và đại từ mong muốn ("she"), nghiên cứu chứng minh khả năng đảo ngược hoàn toàn định kiến của mô hình chỉ bằng can thiệp tại một tầng duy nhất. Phân tích quét toàn bộ các tầng tiết lộ rằng tác động của việc hiệu chỉnh tăng dần theo độ sâu của mô hình, đồng thời đặt ra giả thuyết về tính tuyến tính của không gian Embeddings.

---

## 1. Mở Đầu (Introduction)
Định kiến (Bias) trong các mô hình ngôn ngữ lớn là hệ quả tất yếu của việc huấn luyện trên dữ liệu khổng lồ do con người tạo ra. Trong khi việc loại bỏ định kiến trên toàn hệ thống là một thách thức vĩ mô, chúng ta có thể sử dụng Diễn giải cơ học (Mechanistic Interpretability) để thực hiện các can thiệp cục bộ. Nghiên cứu này kết hợp kỹ thuật đánh giá định kiến với phương pháp hiệu chỉnh Hidden States nhằm mục tiêu: (1) Chứng minh sự tồn tại của định kiến giới; (2) Thực hiện hiệu chỉnh nhân quả chính xác (Surgical correction).

---

## 2. Tiết Thiết Lập Can Thiệp (Methodology)

### 2.1. Kỹ thuật Trộn Vector (Weighted Vector Mixing)
Thay vì triệt tiêu hay bơm nhiễu, chúng ta sử dụng một hàm Hook để thực hiện phép tổ hợp tuyến tính giữa hai vector:

$$

\mathbf{v}_{new} = w_1 \cdot \mathbf{v}_{natural} + w_2 \cdot \mathbf{v}_{external}

$$


Trong đó $\mathbf{v}_{natural}$ là vector mô hình tự tính toán cho token [MASK], và $\mathbf{v}_{external}$ là vector thu được từ một lần chạy trước đó có chứa đại từ đích ("she"). 

### 2.2. Chỉ số Định kiến (Bias Score)
Định nghĩa chỉ số định kiến dựa trên sự chênh lệch Log Softmax:

$$

\text{Bias Score} = \log P(\text{"he"}) - \log P(\text{"she"})

$$


- **Dương:** Thiên kiến nam giới.
- **Âm:** Thiên kiến nữ giới.
- **0:** Trung hòa giới tính (Lý tưởng).

---

## 3. Kết Quả Thực Nghiệm (Results & Analysis)

### 3.1. Xác nhận Định kiến Nguyên bản
Với câu mẫu: "The engineer informed the client that [MASK] would need more time", mô hình BERT Large dự đoán xác suất cho "he" đạt hơn 60%, trong khi "she" chỉ chiếm khoảng 2%. 
- **Quan sát thú vị:** Nếu đổi "engineer" thành số nhiều "engineers", xác suất cho đại từ trung tính "they" sẽ nhảy vọt lên vị trí dẫn đầu, cho thấy mô hình nhạy cảm với các dấu hiệu ngữ pháp số ít/số nhiều.

### 3.2. Hiệu chỉnh Phẫu thuật (Surgical Debias)
Tại Layer 10, thực hiện trộn vector với tỷ lệ 10% (Mask) và 90% (She-target).
- **Kết quả:** Xảy ra sự đảo ngược cực đoan. Xác suất dự đoán "she" tiến gần 100%, Bias Score chuyển từ +3 (thiên nam) sang -11 (thiên nữ). Điều này khẳng định ta có thể "ép" mô hình thay đổi hành vi thông qua việc bơm biểu diễn ẩn.

### 3.3. Tác động của Độ sâu Tầng (Layer-wise Sweep)
Khi lặp lại phép trộn 50/50 qua tất cả các tầng:
- **Tầng sớm:** Can thiệp hầu như không có tác động đến Logits đầu ra cuối cùng.
- **Tầng sâu:** Tác động tăng dần đồng nhất. Càng tiến về phía output, việc bơm vector "she" càng định hình mạnh mẽ kết luận của mô hình. Điều này chứng minh các tầng cuối cùng là nơi mô hình tích hợp thông tin ngữ cảnh để đưa ra quyết định cuối cùng.

---

## 4. Thảo Luận Và Kết Luận
Việc hiệu chỉnh định kiến bằng phép cộng trung bình tuyến tính đã thành công rực rỡ trong thí nghiệm này, củng cố giả thuyết rằng không gian Embeddings của BERT có tính xấp xỉ tuyến tính (Linear approximation) đối với các khái niệm trừu tượng như giới tính. Tuy nhiên, cần lưu ý:
1. **Tính cục bộ:** Hiệu chỉnh này chỉ áp dụng cho một ví dụ cụ thể, chưa đảm bảo tính tổng quát hóa.
2. **Hình học không gian:** Nếu cấu trúc đại diện của giới tính là các đường cong hoặc mặt phẳng phức tạp, việc cộng trung bình tuyến tính có thể làm hỏng tính logic của các biểu diễn khác.

Báo cáo khẳng định: Can thiệp nhân quả là công cụ mạnh mẽ để "bẻ lái" mô hình, nhưng cần được áp dụng thận trọng trong các hệ thống thực tế.

---

## Tài liệu tham khảo (Citations)
1. Thử nghiệm hiệu chỉnh định kiến giới trên BERT Large dựa trên `aero_LLM_04_CodeChallenge Measure and correct BERT's bias.md`. Phân tích Bias Score theo độ sâu Transformer Blocks.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Tác động Hạ nguồn của việc Thay đổi Quy mô Lớp sớm (Downstream Impact of Early Layer Scaling)](aero_llm_01_downstream_impact_of_early_layer_scaling.md) | [Xem bài viết →](aero_llm_01_downstream_impact_of_early_layer_scaling.md) |
| [Thử thách Lập trình: Thay đổi Quy mô Hidden State và Tổn thất Token](aero_llm_02_codechallenge_hidden_state_scaling_and_token_loss.md) | [Xem bài viết →](aero_llm_02_codechallenge_hidden_state_scaling_and_token_loss.md) |
| [Thử thách Lập trình: Dự đoán BERT với Nhiễu và Hoán vị (Noisy and Shuffled BERT Predictions)](aero_llm_03_codechallenge_noisy_and_shuffled_bert_predictions.md) | [Xem bài viết →](aero_llm_03_codechallenge_noisy_and_shuffled_bert_predictions.md) |
| 📌 **[Thử thách Lập trình: Đo lường và Hiệu chỉnh Định kiến Giới trong BERT](aero_llm_04_codechallenge_measure_and_correct_bert_s_bias.md)** | [Xem bài viết →](aero_llm_04_codechallenge_measure_and_correct_bert_s_bias.md) |
| [Vá Hoạt hóa và Tác vụ Nhận diện Tân ngữ Gián tiếp (Activation Patching and Indirect Object Identification)](aero_llm_05_activation_patching_with_indirect_object_identification.md) | [Xem bài viết →](aero_llm_05_activation_patching_with_indirect_object_identification.md) |
| [Bỏ qua một Tầng Transformer (Skip a Layer)](aero_llm_06_skip_a_layer.md) | [Xem bài viết →](aero_llm_06_skip_a_layer.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
