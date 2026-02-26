
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [15 Editing hidden states](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Thử thách Lập trình: Thay đổi Quy mô Hidden State và Tổn thất Token

## Tóm tắt (Abstract)
Báo cáo này trình bày kết quả thử thách lập trình về tác động của việc can thiệp Hidden State đối với đầu ra của mô hình (Token selection) và giá trị Log Softmax. Sử dụng mô hình GPT-2 Medium, nghiên cứu thực hiện các phép thay đổi quy mô năng động thông qua Dictionary-based Hooks. Thí nghiệm lấy danh ngôn của Einstein làm mẫu thử để quan sát sự biến thiên của Logits và Loss khi một lớp cụ thể bị suy giảm tín hiệu. Kết quả gây ngạc nhiên cho thấy việc giảm quy mô (Scale 0.6) tại một số tầng có thể làm "sắc bén" phân phối xác suất, dẫn đến việc giảm Loss cho token mục tiêu, tương tự như hiệu ứng giảm nhiệt độ (Temperature) trong hàm Softmax.

---

## 1. Mở Đầu (Introduction)
Mục tiêu cốt lõi của Diễn giải học (Interpretability) không chỉ dừng lại ở việc quan sát các vi mạch nội tại mà phải kết nối được các biến động đó với hành vi đầu ra thực tế của mô hình (sinh từ). Thử thách này tập trung vào việc định lượng sự thay đổi của Logits toàn vocab khi ta "bóp" tín hiệu tại một Transformer Block bất kỳ. Chúng ta sẽ kiểm chứng liệu mô hình có còn giữ được khả năng dự đoán chính xác token kế tiếp sau khi bị can thiệp nhân quả hay không.

---

## 2. Tiết Thiết Lập Thử Thách (Methodology)

### 2.1. Cấu trúc Dict-based Hook Linh hoạt
Thay vì hard-code một layer duy nhất, chúng ta xây dựng hệ thống Hook tham chiếu đến một `scaling_dict`.
- **Cơ chế:** `if layer_num in scaling_dict.keys(): output = output[0] * scaling_dict[layer_num]`.
- **Lợi ích:** Cho phép kiểm thử đơn lẻ hoặc đồng thời nhiều lớp với các hệ số scale khác nhau chỉ bằng cách cập nhật Dictionary mà không cần gỡ/cài lại Hook.

### 2.2. Dữ liệu Thử nghiệm và Baseline
- **Prompt:** "I have no special talents. I am only passionately" (Trích Einstein).
- **Target Token:** " curious" (Token ID: 11040).
- **Baseline:** Chạy mô hình ở trạng thái nguyên bản (`pure_logits`) để làm mốc đối chứng cho xác suất và Loss.

---

## 3. Kết Quả Thực Nghiệm (Results & Analysis)

### 3.1. Sự Tĩnh Lặng Toàn Cục (Global Suppression)
Khi scale Layer 2 với hệ số 0.6, đồ thị Logits cho thấy một sự sụt giảm biên độ đồng loạt (Global downward shift) trên toàn bộ dải từ vựng. Mặc dù cường độ tín hiệu giảm mạnh, mối tương quan (Correlation) giữa Logits sạch và Logits bị can thiệp vẫn duy trì ở mức cực cao ($r \approx 0.995$). Điều này chứng tỏ cấu trúc tương đối giữa các từ vẫn được bảo toàn.

### 3.2. Nghịch lý Giảm Loss (The Loss Paradox)
Một phát hiện thú vị là khi scale lớp sớm, vị trí của token " curious" trong danh sách Top-10 dự đoán lại tăng lên so với mô hình gốc. 
- **Giải thích:** Việc giảm quy mô Hidden State tương đương với việc "làm lạnh" (decreasing temperature) hệ thống. Nó giúp loại bỏ bớt các nhiễu nền và làm cho phân phối xác suất tập trung hơn vào các ứng viên hàng đầu. Trong trường hợp này, sự can thiệp nhân quả vô tình lại mang lại kết quả "tốt hơn" về mặt toán học (Loss thấp hơn).

### 3.3. Quét Toàn Bộ Các Lớp (Layer Sweep)
Thực hiện lặp qua 24 lớp của GPT-2 Medium:
- **Tính ổn định:** Hầu hết các lớp khi bị scale 0.6 đều dẫn đến việc giảm Loss cho token mục tiêu.
- **Xu hướng:** Loss có xu hướng tăng dần (mô hình dự đoán kém đi) khi can thiệp xảy ra ở các lớp càng sâu về phía cuối. Điều này củng cố giả thuyết rằng các lớp cuối cùng đóng vai trò quyết định trực tiếp hơn đến việc tinh chỉnh xác suất đầu ra.

---

## 4. Kết Luận
Can thiệp nhân quả bằng cách thay đổi quy mô Hidden State tiết lộ rằng mô hình có tính ổn định cao về mặt cấu trúc tương quan Logits. Tuy nhiên, cường độ tín hiệu có ảnh hưởng trực tiếp đến độ "sắc" của softmax. Việc giảm năng lượng tín hiệu (Scaling down) có thể làm giảm tính ngẫu nhiên (Stochasticity) của mô hình. Bài học rút ra là: khi nghiên cứu nội động lực của mô hình, luôn cần liên kết chúng với lựa chọn Token cuối cùng để đánh giá tác động thực tiễn.

---

## Tài liệu tham khảo (Citations)
1. Thử thách Hidden-state scaling trên GPT-2 Medium dựa trên tài liệu `aero_LLM_02_CodeChallenge Hidden-state scaling and token loss.md`. Phân tích sự tương đồng giữa Scaling và Softmax Temperature.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Tác động Hạ nguồn của việc Thay đổi Quy mô Lớp sớm (Downstream Impact of Early Layer Scaling)](aero_LLM_01_Downstream impact of early layer scaling.md) | [Xem bài viết →](aero_LLM_01_Downstream impact of early layer scaling.md) |
| 📌 **[Thử thách Lập trình: Thay đổi Quy mô Hidden State và Tổn thất Token](aero_LLM_02_CodeChallenge Hidden-state scaling and token loss.md)** | [Xem bài viết →](aero_LLM_02_CodeChallenge Hidden-state scaling and token loss.md) |
| [Thử thách Lập trình: Dự đoán BERT với Nhiễu và Hoán vị (Noisy and Shuffled BERT Predictions)](aero_LLM_03_CodeChallenge Noisy and shuffled BERT predictions.md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Noisy and shuffled BERT predictions.md) |
| [Thử thách Lập trình: Đo lường và Hiệu chỉnh Định kiến Giới trong BERT](aero_LLM_04_CodeChallenge Measure and correct BERT's bias.md) | [Xem bài viết →](aero_LLM_04_CodeChallenge Measure and correct BERT's bias.md) |
| [Vá Hoạt hóa và Tác vụ Nhận diện Tân ngữ Gián tiếp (Activation Patching and Indirect Object Identification)](aero_LLM_05_Activation patching with indirect object identification.md) | [Xem bài viết →](aero_LLM_05_Activation patching with indirect object identification.md) |
| [Bỏ qua một Tầng Transformer (Skip a Layer)](aero_LLM_06_Skip a layer.md) | [Xem bài viết →](aero_LLM_06_Skip a layer.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
