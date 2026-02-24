# Mạng Mạch Thuật Toán (Circuits) Trong Mô Hình Học Sâu

## Tóm tắt (Abstract)
Báo cáo này mở ra một khái niệm trọng tâm trong nghiên cứu Cơ học Diễn dịch (Mechanistic Interpretability): Mạng mạch (Circuits). Nằm ở cấp độ trừu tượng cao hơn so với việc phân tích các nơ-ron đơn lẻ, nhưng lại thu hẹp hơn so với việc đánh giá toàn bộ Vector nhúng (Embeddings) hay Khối xử lý (Transformer Layers), khái niệm "Circuit" đại diện cho một cụm các chiều không gian (dimensions) hay cụm nơ-ron hợp tác để thực thi một vi tác vụ luận lý cụ thể. Việc truy tìm các mạng mạch này đối diện với nghịch lý giới hạn bởi tính liên tục (Continuous parameters) tự nhiên của Toán học học sâu và sự thay đổi linh hoạt theo ngữ cảnh, biến "Mạch thuật toán" trở thành một khái niệm khó định nghĩa bằng ranh giới vật lý tĩnh, sinh ra một bài toán mở đầy hứa hẹn.

---

## 1. Mở Đầu (Introduction)
Trong thiết kế phần cứng, một bảng vi mạch (Electronic circuit) là hệ thống có ranh giới, kết nối rõ ràng vì do con người hàn nối rập khuôn theo mục đích thiết kế. Tuy nhiên, trong Cơ học hệ thống phức tạp tự sinh (Emergent Complex Systems) – như Não bộ sinh học hay Trí tuệ nhân tạo (Deep Learning Language Models), khái niệm Mạch trở nên trừu tượng và phi ranh giới. 
Nghiên cứu về Circuits hướng đến việc trả lời câu hỏi: Làm thế nào những tổ hợp nhỏ các thông số toán học có thể tự động bện xoắn với nhau tạo thành một cụm chức năng (ví dụ: Cụm nhận diện số nhiều, Cụm phát hiện lỗi chính tả, Cụm chú ý đại từ nhân xưng)? 

---

## 2. Kiến Trúc Của Một Mạch Vi Mô Hình (Circuits Architecture)

### 2.1. Đa Tầng Kết Cấu Sinh Học
Tương tự sự tiến hóa của Giải phẫu não bộ (Neuroanatomy), cấu trúc "mạch" trong một hệ thống lớn chứa hàng tỷ tham số thường chia làm nhiều cấp độ:
1. **Cấp Nơ-ron Đơn (Subcellular):** Ở nhánh sinh học, trên một tế bào có thể chia thành nhiều phân luồng vi mạch rẽ nhánh đuôi gai (dendrites) tính toán riêng. Ở AI, nó là tính năng phi tuyến ở một Head độc lập.
2. **Cấp Cụm Tế Bào (Local Network):** Tổ hợp gồm hàng trăm Neuron liên kết nhau gánh vác một khả năng nhận thức nhỏ. Ở AI, nó là một "Circuit" nằm yên trong một nhánh Attention hay MLP cụ thể.
3. **Cấp Vùng Nhận Thức (Macro-regions):** Liên kết nhiều mô-đun để giải quyết luận lý vĩ mô. Ở AI, đây là các khối Circuit đan chéo vượt thời gian nhiều tầng lớp rễ (Multiple Layers).

### 2.2. Bản Chất Bất Định Của Thuật Toán Sâu (Continuous Activations) 
Bài toán tìm ra Circuit của một mạng LM gặp ba thách thức siêu việt:
- Không giống não bộ đo được dòng xung điện tách biệt, trong Học sâu, các trọng số tham số (Weights) và mức kích hoạt tuyến tính (Activations) mang giá trị số thực liên tục (Continuous Numbers). Trừ khi sử dụng các kỹ thuật ép chuẩn L1 Regularizations hay Dropout liên tục, về cơ bản không bao giờ có một cụm nơ-ron nào hoàn toàn đứt kết nối bằng 0 tĩnh. Sự tính toán của chúng lan tỏa vạn vật, cản trở việc xác định đâu là viền ngoài của một Circuit lõi.
- "Circuit Fluidity": Cấu trúc mạng mạch không đứng yên. Khi chuỗi Token dần dần được nạp thêm vào ngữ cảnh (Context length), các thành viên cấu thành nên mạch hiện tại có thể được thay thế hoặc dạt hướng để phục vụ một Mạch mới động.

---

## 3. Khái Niệm Hoạt Động (Operational Definition)
Trong giới hạn cơ bản, chúng ta xem Circuit trong LLMs bằng lăng kính **Thống kê Cơ học (Statistical techniques)**: Mạch là một tập con các mô-đun (neurons hoặc heads) cho thấy sự biểu thị cùng hành vi (Behaving in similar localized ways) dưới các loại Context xác định, được khai thác thông qua phân rã thành phần vĩ mô hoặc hồi quy thưa thớt (Sparse probing). Quá trình giải phẫu sẽ được triển khai bằng cách bóc tách từng Attention Head trên nền các phép toán tương quan ma trận.

---

## 4. Kết Luận
Việc bọc tách thuật toán Mạch (Circuits tracking) là ngọn giáo sắc bén nhất của Mechanistic Interpretability hiện nay, nhưng cũng ẩn chứa nhược điểm "Diễn dịch quá đà" (Over-interpretation) khi ta cố gò ép máy móc tư duy theo mô hình sinh học tĩnh. Sự nhận thức thấu đáo và linh hoạt sẽ mở đường cho những chuỗi bài tập cô lập từng Head và phân tích biến số ẩn sắp tới.

---

## Tài Liệu Tham Khảo (Citations)
1. Thảo luận lý thuyết cốt lõi về bản thể học của Circuits, sự khác biệt giữa Engineering circuit và Emergent complex system phân tích từ `aero_LLM_01_What is a circuit in a DL model.md`.
