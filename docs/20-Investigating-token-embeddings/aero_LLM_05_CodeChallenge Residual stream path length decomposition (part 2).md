# Thử Thách Lập Trình: Phân Rã Độ Dài Đường Dẫn Luồng Số Dư (Phần 2)

## Tóm tắt (Abstract)
Tiếp tục cuộc khảo sát trên GPT-2 Large, ở phần hai, tiêu điểm được dịch chuyển từ góc quay sang phân tích tham số Độ dài đường dẫn (Path Length) phát sinh bởi hai nhánh Attention và MLP, so với luồng số dư tổng (Hidden States). Kết nối biến thiên Euclidean cho thấy cấu trúc đồ thị tương quan rõ rệt: MLP thể hiện ảnh hưởng áp đảo hơn so với Attention trong việc tạo hình dạng vóc vector chốt xuất ra. Đồng thời, nghiên cứu cảnh báo sự nguy hiểm khi xử lý thống kê Pearson do dữ liệu Token mở màn (First Token Outlier) gãy ngoặt mọi phân bố, và giải pháp thay thế thông minh nhất là lược bỏ nó hoặc chuyển qua hàm hạng Spearman.

---

## 1. Mở Đầu (Introduction)
Sau khi khẳng định rằng hai bộ lọc Attention và MLP thực thi các thuật toán tính toán định hướng trực giao (Cosine Similarity $\approx 0$), câu hỏi quan thiết tiếp theo là: Vậy cơ cấu nào mạnh hơn, cơ cấu nào đóng góp phần xác thịt (magnitude) làm giãn vector nhiều hơn trên đường đi qua Transformer Layer?
Để theo vết, phân tích Độ dài đường dẫn (Path Lengths $PL$) qua 3 bộ lọc riêng rẽ (PL Attn, PL MLP và PL Hidden States) được thiết lập. 

---

## 2. Quá Trình Thi Thiết & Phân Tích (Methodology & Analysis)

### 2.1. Phác Đồ Lưới Chênh Lệch Euclide (Path Length Matrix)
Sử dụng công thức khoảng cách Euclide chéo lớp (Layer $i \to i-1$). 
Tại các block đầu của mô hình, sự giãn nở Path length vẫn âm ỉ, tuy nhiên khi tiến sâu vào những cỗ máy block ở đoạn cuối (chặng áp chót trước khi xuất ra Vocab Matrix), độ lớn của vector điều chỉnh từ cả 3 luồng (Attention, MLP, HS) tăng sực nức. Màu biểu đồ trải màu bung sáng mạnh ở khúc đuôi phản ánh "Cú đẩy cuối cùng" (Big Step Change) trước rào cản chọn từ tiếp theo.

### 2.2. Khớp Tương Quan Pearson R & Chỉ Báo Nhỉnh Hơn Của MLP
Sử dụng phép tương quan biến $PL_{attn} \leftrightarrow PL_{h\_states}$ và $PL_{mlp} \leftrightarrow PL_{h\_states}$:
- Hầu hết các Transformer Layers, cường độ bơm tín hiệu của MLP có quan hệ gắn bó cao hơn nhiều (Correlation mạnh và Positive rõ ràng) so với nhánh vất vả hơn của Attention.
- Sự thắng thế này hợp lý theo logic cơ học: Nhiệm vụ của Khối Attention là phóng tầm nhìn đi khắp chuỗi dài tóm bắt Context. Nhiệm vụ của MLP đóng vai trò nhào nặn Không gian chiều (Expand Dimensionality), biến vector thành hình dáng sẵn sàng cho quy trình phân lớp.  
- Hơn nữa, vì MLP nằm đằng sau nút giao Attention, nó đã "ăn" theo phần nội tiết điều chỉnh đó, nghiễm nhiên hệ quả của MLP sẽ lan tỏa đậm hơn khi tới chặng đích kết thúc Transform Block.

### 2.3. Hiệu Ứng Bẻ Gãy Outlier Từ Chữ Đầu Tiên (The First Token Anomaly)
Thực nghiệm tiết lộ một bài học xương máu: Khi để hớ hênh **Token thứ nhất** dập chung vào mẻ phân tích Correlation của Pearson, toàn bộ cấu trúc biểu đồ lập tức nát vụn (Từ mạnh mẽ $0.6$ tụt thảm hại âm $\to -0.2$). Tác nhân cốt lõi là First Token mang trạng thái "khởi động mù" cực đoan, xé toạc đồ thị văng xa khỏi tâm phân phối cả triệu độ lệch chuẩn.
**Cách xử lý:** 
1. Tốt nhất là loại bỏ hoàn toàn Token 0.
2. Hoặc phải sử dụng cơ chế tương quan Spearman R (Tính bằng Rank) vốn dĩ triệt tiêu mọi uy lực của Outliers.

---

## 3. Kết Luận
Sự tương quan khập khiễng giữa Path lengths minh họa rõ mô hình "Công xưởng hai khâu" của Transformer: Attention đóng vai người dò tin (Information gatherer) và MLP gánh vác việc gia công đóng gói khối tính (Information structurer), với sức ép cấu hình kích cỡ Vector phần lớn bị MLP định hình. Bức tranh này bồi đắp kiến thức vi mô về LLMs, đồng thời giương cao lá cờ đỏ nhắc nhở về sự hiện diện rủi ro của Token khởi đầu.

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm chạy hàm Correlation và tính Cumulative sum trên các luồng trích xuất Pytorch dựa theo mã nguồn tại `aero_LLM_05_CodeChallenge Residual stream path length decomposition (part 2).md`.
