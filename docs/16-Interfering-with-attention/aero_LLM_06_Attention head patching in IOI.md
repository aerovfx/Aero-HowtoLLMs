# Vá lỗi Attention Head trong tác vụ Nhận dạng Tân ngữ Gián tiếp (Attention Head Patching in IOI)

## Tóm tắt (Abstract)
Báo cáo này mở rộng kỹ thuật vá hoạt hóa (Activation Patching) từ mức độ Hidden States (toàn bộ transformer block) xuống mức độ tinh vi hơn là các cá thể Attention Head. Sử dụng tác vụ Nhận dạng Tân ngữ Gián tiếp (Indirect Object Identification - IOI), nghiên cứu thực hiện việc "cấy ghép" (transplanting) hoạt hóa của head từ một chuỗi donor sang một chuỗi recipient. Kết quả thực nghiệm cho thấy một sự khác biệt đáng kinh bạt: trong khi việc vá Hidden States tạo ra sự thay đổi hành vi triệt để, việc vá tất cả Attention Heads chỉ tạo ra những tác động mờ nhạt. Phát hiện này dẫn đến một thảo luận sâu sắc về vai trò của tiểu khối Attention như một cơ chế điều chỉnh (tweaking) thay vì là nguồn thay đổi chính cho các vector biểu diễn.

---

## 1. Mở Đầu (Introduction)
Tác vụ IOI (ví dụ: "Sam và Sally đi công viên, Sam tặng quà cho...") yêu cầu mô hình phải xác định chính xác đối tượng không lặp lại (Sally). Trong các nghiên cứu trước, việc vá (patching) toàn bộ Hidden States đã chứng minh được sự tồn tại của "các tầng quan trọng" nơi tri thức ngữ pháp được lưu trữ. Báo cáo này đi sâu vào việc phẫu thuật các Attention Heads để xem liệu chúng ta có thể cô lập hành vi này ở mức độ head hay không.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Phân biệt Hidden States Patching và Head Patching
- **Hidden States (Trạng thái ẩn):** Là đầu ra cuối cùng của Transformer Block, trực tiếp hình thành dự đoán token tiếp theo. Patching ở đây là thay đổi toàn bộ "niềm tin" của mô hình tại tầng đó.
- **Attention Heads:** Chỉ tính toán các điều chỉnh nhỏ (adjustments) dựa trên ngữ cảnh để cộng vào residual stream. Patching ở đây chỉ là thay đổi "cách mô hình nhìn vào ngữ cảnh".

### 2.2. Quy trình Kỹ thuật
- **Bước 1:** Chạy Forward Pass trên chuỗi A (Donor) để lấy và lưu trữ hoạt hóa của tất cả các Heads thông qua Hook.
- **Bước 2:** Chạy Forward Pass trên chuỗi B (Recipient), đồng thời dùng Hook để ghi đè hoạt hóa của các Heads bằng dữ liệu từ chuỗi A.
- **Cấu trúc Hook:** Sử dụng `Forward Pre-hook` trên lớp `c_proj` để can thiệp vào các Heads trước khi chúng bị trộn lẫn bởi ma trận $W_O$.

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Sự chênh lệch về hiệu quả (Discrepancy)
- **Quan sát:** Ngay cả khi vá toàn bộ 12 heads trong một tầng của GPT-2 Small, chỉ số IOI (xác suất của Sally so với Sam) chỉ dịch chuyển nhẹ. Mô hình không bao giờ hoàn toàn bị đánh lừa để chọn Sam (tân ngữ sai) như khi vá Hidden States.
- **Tính nhất quán:** Hiện tượng này lặp lại trên cả GPT-2 Medium, cho thấy đây là đặc tính kiến trúc chứ không phải do quy mô tham số.

### 3.2. Giải thích cơ học
Tại sao việc vá Heads lại yếu hơn nhiều so với Hidden States? 
- **Lý do:** Hidden States bao gồm cả đầu ra của Attention và mạng MLP. Quan quan trọng hơn, Attention chỉ đóng góp một phần nhỏ (Residual) vào vector biểu diễn tổng thể. Khi chúng ta chỉ vá Head, chúng ta chỉ đang thay đổi "phần bổ sung" ngữ cảnh, trong khi "nội dung gốc" (từ các tầng trước đó trong residual stream) vẫn được bảo toàn mạnh mẽ.

---

## 4. Thảo Luận: Deterministic Logic trong Coding
Nghiên cứu nhấn mạnh tầm quan trọng của các sanity checks:
- Việc so sánh trực tiếp các tensor sau khi vá (`head_xb == head_xa`) là bước bắt buộc để xác nhận Hook hoạt động đúng. 
- Sự phức tạp của việc `reshape` (từ embeddings sang heads) là cần thiết để chuẩn bị cho các can thiệp nhắm mục tiêu vào duy nhất một head trong các thử thách tiếp theo.

---

## 5. Kết Luận
Việc vá Attention Heads tiết lộ rằng các heads hoạt động như những bộ tinh chỉnh tinh vi. Mặc dù chúng mang thông tin ngữ cảnh quan trọng, nhưng sức mạnh của chúng bị giới hạn bởi cấu trúc residual stream. Những thảo luận tiếp theo sẽ nhắm vào việc mô tả toán học cho sự khác biệt này và cách cô lập các "Heads chuyên biệt" cho tác vụ IOI.

---

## Tài liệu tham khảo (Citations)
1. Thí nghiệm Head Patching trên tác vụ IOI dựa trên `aero_LLM_06_Attention head patching in IOI.md`. So sánh sự khác biệt giữa can thiệp Hidden States và Attention Subblock.
