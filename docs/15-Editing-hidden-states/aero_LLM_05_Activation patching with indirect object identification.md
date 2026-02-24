# Vá Hoạt hóa và Tác vụ Nhận diện Tân ngữ Gián tiếp (Activation Patching and Indirect Object Identification)

## Tóm tắt (Abstract)
Báo cáo này giới thiệu hai khái niệm quan trọng trong diễn giải học LLM: kỹ thuật Vá hoạt hóa (Activation Patching) và tác vụ Nhận diện Tân ngữ Gián tiếp (Indirect Object Identification - IOI). Thông qua việc sử dụng mô hình GPT-2 XL, nghiên cứu thực hiện việc "cấy ghép" các Hidden States từ một chuỗi văn bản "nguồn" (donor) sang một chuỗi "đích" (recipient) để quan sát sự chuyển dịch quyết định của mô hình. Kết quả thực nghiệm cho thấy một sự chuyển pha (phase transition) rõ rệt: các tầng sớm có khả năng kháng nhiễu cao, trong khi các tầng từ giữa đến cuối mô hình cực kỳ nhạy cảm với việc vá hoạt hóa, dẫn đến việc mô hình thay đổi hoàn toàn dự đoán ngữ pháp theo dữ liệu bị cấy ghép.

---

## 1. Mở Đầu (Introduction)
Vá hoạt hóa (Activation Patching) là một phương pháp can thiệp nhân quả mạnh mẽ, cho phép ta cô lập tác động của một token cụ thể tại một lớp cụ thể. Thay vì thay đổi trọng số mô hình hoặc thêm nhiễu ngẫu nhiên, ta sử dụng đúng các hoạt hóa "sạch" từ một ngữ cảnh khác để "corrupt" (làm sai lệch) quá trình xử lý của mô hình. Tác vụ IOI cung cấp một khung tham chiếu ngữ pháp hoàn hảo để đo lường sự thay đổi này thông qua khả năng nhận diện tân ngữ gián tiếp (ví dụ: xác định ai là người nhận quà trong câu).

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Kỹ thuật Vá Hoạt hóa (Patching)
- **Cơ chế:** Giả sử có hai câu A và B chỉ khác nhau ở tên riêng. Ta trích xuất Hidden State của tên riêng ở câu A và ghi đè nó vào vị trí tương ứng ở câu B trong quá trình Forward Pass.
- **Mục tiêu:** Xác định xem thông tin tại vị trí và tầng đó đóng góp bao nhiêu phần trăm vào quyết định cuối cùng của mô hình.

### 2.2. Tác vụ IOI (Indirect Object Identification)
- **Cấu trúc:** "Bob and Barbara went to the beach. Bob gave the umbrella to [MASK]".
- **Logic:** Mô hình phải hiểu rằng Barbara là người nhận (indirect object).
- **Thí nghiệm:** Ta hoán đổi vị trí chủ ngữ và tân ngữ giữa hai câu để tạo ra cặp donor-recipient hoàn hảo.

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Phân tích Logit Difference (IOI Metric)
Chúng ta đo lường sự chênh lệch logit giữa hai ứng viên tiềm năng (ví dụ: $Logit(\text{Mike}) - Logit(\text{Emma})$). 
- Ở trạng thái sạch, mô hình thể hiện sự tự tin cao (chênh lệch khoảng 2-3 đơn vị logit) vào đáp án đúng ngữ pháp.

### 3.2. Sự Chuyển Pha Của Các Tầng (Layer-wise Phase Transition)
Khi thực hiện vá hoạt hóa lần lượt qua 48 tầng của GPT-2 XL:
- **Tầng 0 - 20:** Mô hình hầu như phớt lờ các vector bị "vá" vào. Dự đoán cuối cùng vẫn chính xác theo ngữ cảnh của câu đích. Điều này cho thấy giai đoạn này mô hình chủ yếu xử lý ở mức độ từ vựng sơ khai.
- **Tầng 25 trở đi:** Xuất hiện một sự sụp đổ đột ngột. Mô hình bắt đầu tin vào thông tin từ vector bị cấy ghép và đưa ra dự đoán sai (nhưng lại đúng theo câu nguồn). Đây chính là nơi cấu trúc ngữ pháp và quan hệ thực thể được tích hợp và quyết định.

### 3.3. Đặc điểm Kỹ thuật của Hugging Face
Một phát hiện quan trọng trong quá trình kiểm chứng (Sanity check): Phép vá hoạt hóa hoạt động hoàn hảo trên mọi tầng trừ tầng cuối cùng. 
- **Lý do:** Trong cấu trúc của Hugging Face, `hidden_states` của tầng cuối đã bao gồm cả bước `LayerNorm` cuối cùng của toàn bộ Encoder/Decoder. Việc can thiệp vào đây tương đương với việc "áp đặt LayerNorm hai lần", gây ra sai số nhỏ trong phép đo nhưng không làm thay đổi xu hướng chung của thí nghiệm.

---

## 4. Kết Luận
Vá hoạt hóa giống như việc sử dụng một "chiếc búa tạ" để thăm dò các mạch thần kinh. Nó tiết lộ rằng kiến thức về ngữ pháp và quan hệ thực thể không phân bổ đều mà tập trung mạnh mẽ ở nửa sau của mạng residual stream. Thí nghiệm này đặt nền móng cho các nghiên cứu sâu hơn về việc can thiệp vào từng Attention Head riêng lẻ để tìm ra chính xác "vị trí" của các mạch logic trong mô hình.

---

## Tài liệu tham khảo (Citations)
1. Thí nghiệm Activation Patching trên GPT-2 XL dựa trên `aero_LLM_05_Activation patching with indirect object identification.md`. Quan sát hiện tượng chuyển pha trong tác vụ IOI.
