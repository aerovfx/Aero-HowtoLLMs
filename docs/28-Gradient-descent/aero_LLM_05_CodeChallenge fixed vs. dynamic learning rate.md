# Học sâu: Thử thách Lập trình – Tốc độ học Cố định vs. Động (Fixed vs. Dynamic Learning Rate)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về tầm quan trọng của việc điều chỉnh tốc độ học (learning rate) một cách linh hoạt trong quá trình huấn luyện mô hình học sâu. chúng ta thực hiện so sánh đối chứng giữa một tốc độ học cố định (fixed) và các phương pháp điều chỉnh động (dynamic) dựa trên thời gian (epochs) và độ dốc (gradients). Nghiên cứu thực hiện các thực nghiệm trên Python để minh chứng rằng việc giảm tốc độ học theo thời gian (learning rate decay) hoặc điều chỉnh theo độ lớn của gradient không chỉ giúp mô hình hội tụ nhanh hơn mà còn đạt được độ chính xác cao hơn tại điểm tối ưu, từ đó đặt nền móng cho các thuật toán tối ưu hóa hiện đại như Adam và RMSprop.

---

## 1. Hạn chế của Tốc độ học Cố định

Trong các mô hình cơ bản, tốc độ học được giữ không đổi suốt toàn bộ vòng lặp. Tuy nhiên, điều này dẫn đến hai nghịch lý:
- **Nguyên lý:** Nếu chọn $\eta$ lớn, mô hình học nhanh lúc đầu nhưng sẽ bị "dao động" (vượt qua cực tiểu) khi về gần đích. Nếu chọn $\eta$ nhỏ, mô hình ổn định nhưng mất quá nhiều thời gian để di chuyển từ điểm khởi tạo.
- **Mục tiêu thử thách:** Tìm kiếm sự cân bằng bằng cách biến $\eta$ thành một biến số thay đổi theo từng bước lặp.

---

## 2. Phương pháp Điều chỉnh Động

Nghiên cứu phân tích hai chiến lược điều chỉnh động phổ biến:

### 2.1. Điều chỉnh dựa trên Thời gian (Learning Rate Decay)
- **Cơ chế:** Càng về sau quá trình huấn luyện, tốc độ học càng giảm dần theo công thức: $\eta_{mới} = \eta_{gốc} \cdot (1 - \frac{i}{N})$, với $i$ là epoch hiện tại và $N$ là tổng số epoch.
- **Ưu điểm:** Giúp mô hình thực hiện các bước đi lớn khi mới bắt đầu (khám phá không gian) và các bước đi cực nhỏ khi đã ở gần cực tiểu (tăng độ chính xác).

### 2.2. Điều chỉnh dựa trên Gradient (Adaptive Learning)
- **Cơ chế:** Tốc độ học được tỷ lệ thuận với độ lớn của gradient tại vị trí hiện tại. 
- **Lý thuyết:** Khi gradient lớn (đang ở dốc đứng/xa đích), $\eta$ sẽ lớn để đẩy nhanh tốc độ. Khi gradient tiến về 0 (đáy thung lũng), $\eta$ tự động giảm xuống để tránh làm văng mô hình khỏi cực tiểu.

---

## 3. Kết quả Thực nghiệm và Phân tích Hội tụ

Thông qua việc chạy đồng thời ba kịch bản huấn luyện:
1. **Time-based Learning Rate (Xanh lá):** Thường giành chiến thắng về tốc độ và độ chính xác. Đạt đến mục tiêu $x=0.5$ chỉ sau 10 epoch thay vì 50.
2. **Gradient-based Learning Rate (Cam):** Thể hiện tính thích nghi tốt với địa hình hàm số nhưng cần sự chuẩn hóa (scaling) phù hợp để tránh làm tăng $\eta$ quá mức.
3. **Fixed Learning Rate (Xanh dương):** Hội tụ chậm và thường dừng lại ở một sai số lớn hơn so với các phương pháp động.

---

## 4. Liên hệ với các Optimizer Hiện đại

Những khái niệm trong thử thách này là tiền thân của các giải pháp công nghiệp:
- **RMSprop và Adam:** Sử dụng các biến thể của việc điều chỉnh $\eta$ dựa trên bình phương trung bình của các gradient trong quá khứ.
- **Scheduler:** Các bộ lập lịch trong PyTorch (như StepLR) thực hiện chính xác phương pháp giảm theo epoch mà chúng ta đã mô phỏng.

---

## 5. Kết luận
Tốc độ học động là một "vũ khí" tối thượng trong việc tối ưu hóa mạng nơ-ron sâu. Việc thấu hiểu cách thức $\eta$ tương tác với thời gian và bề mặt lỗi giúp nhà nghiên cứu không chỉ huấn luyện mô hình thành công mà còn tiết kiệm được đáng kể tài nguyên tính toán. Thử thách này minh chứng rằng khả năng tự thích nghi là yếu tố tiên quyết để các mô hình LLM có thể học tập hiệu quả từ những khối lượng dữ liệu khổng lồ với cấu trúc bề mặt mất mát phức tạp.

---

## Tài liệu tham khảo (Citations)
1. Thử nghiệm so sánh hiệu năng của các chiến lược tốc độ học dựa trên `aero_LL_05_CodeChallenge fixed vs. dynamic learning rate.md`. Phân tích thực nghiệm về learning rate decay, adaptive methods và ứng dụng trong các optimizer hiện đại.
