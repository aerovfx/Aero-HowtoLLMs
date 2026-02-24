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
