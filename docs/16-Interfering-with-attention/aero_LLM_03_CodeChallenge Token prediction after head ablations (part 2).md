# Thử thách Lập trình: Dự đoán Token sau khi Cắt bỏ Head (Phần 2)

## Tóm tắt (Abstract)
Báo cáo này tập trung vào việc so sánh định lượng các phương pháp can thiệp Attention Head và khám phá tính đặc hiệu của token đối với dự đoán của mô hình. Bằng cách phân tích phân phối xác suất Softmax thông qua biểu đồ Histogram và Heatmap sai lệch, nghiên cứu chỉ ra rằng việc thay thế bằng giá trị không (Zeroing) và giá trị trung bình (Mean) dẫn đến kết quả gần như tương đương do trung bình nội tại của các head vốn đã gần bằng không. Ngoài ra, thực nghiệm về sự khác biệt giữa các token biến thể (ví dụ: có và không có dấu cách) tiết lộ tính nhạy cảm cực cao của mô hình đối với cấu trúc đầu vào, củng cố giả thuyết về "Sự hỗn loạn tất định" (Deterministic Chaos) trong các hệ thống AI phức tạp.

---

## 1. Mở Đầu (Introduction)
Trong nghiên cứu Diễn giải cơ học, câu hỏi liệu "cắt bỏ" (Ablation) hay "thay thế" (Imputation) là phương pháp tốt hơn vẫn còn gây tranh cãi. Phần 2 của thử thách này đi sâu vào: (1) Trực quan hóa sự khác biệt giữa Zero và Mean Imputation; (2) Phân tích tác động của việc thay đổi token đối chứng (Non-target); (3) Kiểm chứng tính chính xác khi chỉ can thiệp vào duy nhất một token cuối cùng trong chuỗi.

---

## 2. Kết Quả Thực Nghiệm (Exercise 5-7)

### 2.1. So sánh Zero vs. Mean Imputation (Exercise 5)
- **Histogram:** Phân phối sai lệch Softmax của phương pháp Zero (Màu cam) và Mean (Màu xanh) gần như trùng khớp hoàn toàn.
- **Giải thích:** Điều này không gây ngạc nhiên vì các kỹ sư AI thường sử dụng kỹ thuật chuẩn hóa (Normalization) để giữ hoạt hóa xoay quanh 0. Việc gán 0 hay gán một giá trị trung bình cực nhỏ (ví dụ 0.05) không tạo ra sự khác biệt đáng kể về mặt thống kê đối với dự án tiếp theo.
- **Điểm mấu chốt:** Tác động chính đến từ việc triệt tiêu phương sai (Variance) của Head – tức là biến mọi giá trị thành một hằng số – hơn là bản thân giá trị của hằng số đó.

### 2.2. Tác động của Dấu cách (Exercise 6 - Token Spacing)
Thay đổi token đối chứng từ " France" sang "Germany" (không có dấu cách phía trước):
- **Kết quả:** Mặc dù về mặt ngữ nghĩa (Semantics) con người coi chúng là một, mô hình phân biệt rạch ròi. Việc can thiệp vào các attention heads làm thay đổi xác suất của " Germany" (có dấu cách) nhưng không ảnh hưởng đến "Germany" (không dấu cách).
- **Lý do:** Mô hình được huấn luyện trên hàng tỷ văn bản và hiểu rằng sau từ "of" phải là một đơn vị từ vựng có dấu cách ngăn cách. Xác suất cho từ "Germany" viết liền là cực thấp (Infinitesimal) và không bị ảnh hưởng bởi các mạch logic ngữ cảnh thông thường.

### 2.3. Can thiệp Token đơn lẻ (Exercise 7 - Precise Ablation)
Thay vì cắt bỏ head cho toàn bộ chuỗi 11 tokens, chúng ta chỉ thay thế giá trị tại token cuối cùng (`input[:, -1, ...]`).
- **Quan sát:** Phân phối sai lệch trở nên "chặt chẽ" (tighter) hơn. Việc chỉ can thiệp vào token cuối cùng có tác động yếu hơn so với việc cắt bỏ trên toàn bộ chuỗi. Điều này chứng tỏ thông tin ngữ cảnh được tích lũy và duy trì dọc theo toàn bộ quá trình xử lý chuỗi của Head.

---

## 3. Thảo Luận: Sự Hỗn Loạn Tất Định (Deterministic Chaos)
LLM là những hệ thống phi tuyến tính cực kỳ phức tạp. 
- **Định nghĩa:** Một thay đổi nhỏ tại một biến số (một Attention Head ở Layer 5) có thể gây ra những hệ quả khó dự đoán ở đầu ra (Layer 12).
- **Tính chất:** Tuy nhiên, hệ thống này là "tất định" (Deterministic) – cùng một đầu vào và cùng một can thiệp sẽ luôn cho ra cùng một kết quả (trong giới hạn sai số máy tính). Điều này cho phép chúng ta thực hiện các nghiên cứu lặp lại và phẫu thuật sâu vào cấu trúc mô hình.

---

## 4. Kết Luận
Kết quả thực nghiệm cho thấy sự tinh vi của mô hình trong việc xử lý token và sự bền vững đối với các can thiệp lẻ tẻ. Việc mô hình có thể duy trì dự đoán đúng ngay cả khi bị mất phương sai tại một Head quan trọng chứng minh tính dự phòng cao của kiến trúc Transformer. Những phát hiện này thôi thúc chúng ta tìm kiếm những phương pháp can thiệp tinh vi hơn, thay vì chỉ sử dụng "búa tạ" để gán bằng không hoặc trung bình.

---

## Tài liệu tham khảo (Citations)
1. Thử thách Token Prediction và Deterministic Chaos trên GPT-2 dựa trên `aero_LLM_03_CodeChallenge Token prediction after head ablations (part 2).md`. Phân tích sự ảnh hưởng của token spacing và precise ablation.
