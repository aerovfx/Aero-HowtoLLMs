# Theo Dõi Dòng Chảy Cosine Similarity Trên Trục Văn Bản Chuyên Tuần Tự (Word Sequences)

## Tóm tắt

Trên thực tế, ngôn ngữ giao tiếp không đơn thuần là những cụm từ đơn độc văng lảng vãng trong không gian Embeddings. Ngôn từ thực thụ chỉ có giá trị khi bị trói buộc vào một "Trị số Không thời gian" - Đó là Trình Tự Chữ Viết (Sequences). Báo cáo thực hành này đào cắt không gian Vector tĩnh của mô hình BERT, áp dụng liên hoàn kỹ thuật Vector hóa độ tương quan Cosine từng bước đệm (Sequential Pairs) thông qua hàm lân cận để giải mã cách bộ máy học tự động bóp méo ý nghĩa theo luồng di chuyển từ vựng.

---

## 1. Cơ Chế Kết Tinh Vector Cosine Nối Tiếp Bề Mặt (Sequential Pairs)
Mô tả cho câu lệnh: 
> *My phone is in the kitchen near the cold ice cream.*

Thuật toán không chạy điểm quy nạp tương tự cho toàn câu, mà nó cắt nhỏ từng chặng $t_i$:
$C(t_i, t_{i-1}) = \cos(\vec{v}_i, \vec{v}_{i-1}) = \frac{\vec{v}_i \cdot \vec{v}_{i-1}}{\|\vec{v}_i\| \|\vec{v}_{i-1}\|}$

Khi đặt lên thanh đồ thị Bar plot:
- Lực hút giữa `cold` và `ice` đẩy Cosine vọt lên ngưỡng $\sim 0.6$ (Mối quan hệ nhiệt đại đa cấu trúc).
- Lực hút giữa `ice` và `cream` duy trì $\sim 0.5$ (Cấu trúc danh từ ghép truyền thống).
- Nhưng lực hút giữa `phone` và `is` sụp đổ xuống mức $\sim 0.15$. Mạng lý lẽ của BERT đã học từ hàng triệu trang sách rằng `is` là động từ to be liên kết ngẫu nhiên với vạn vật. `Phone` chả có thuộc tính gì sinh ra lực hấp dẫn với `is`.

Do đó, đồ thị Sequential Cosine này chính là Biểu Đồ Điện Não Đồ (EEG) cho thấy mức độ gắn kết logic liền kề (Logical transition density) của từng chuỗi tư duy.

---

## 2. Đo Khảo Phân Nhánh Nghĩa Bằng Đường Tiệm Cận Biến Đổi (Diverging Sequences)

Lý do vì sao Cosine Cục bộ quan trọng được chứng minh qua hai câu Garden-Path:
A: *The conductor waved his hands as the train departed.*
B: *The conductor waved his hands as the orchestra began.*

Tại thời điểm bộ Tokenizer đi từ đầu đến chữ `The conductor waved his hands as...`: Trí tuệ của BERT lẫn não sinh học chúng ta chưa phân tích được từ "Conductor" này là "Người soát vé tàu tủy" hay "Nhạc trưởng giao hưởng" (Tính mập mờ ý niệm đa nghĩa Polysense). 

Toàn bộ biểu đồ đồ thị Cosine của hai câu văn đè lên nhau trùng khớp đến $\mathbf{100\%}$. Chỉ đến khi đâm sầm vào 2 Tokens biến hóa cuối cùng (`train departed` và `orchestra began`), biểu đồ mới rẽ nhánh đồ thị (Forking transition):
- Tại điểm rẽ $\to$ `train` với độ dốc Cosine cao hơn, kéo ngược tâm nhúng của mạng nội hàm lên một miền vận chuyển giao thông.
- Tại điểm rẽ $\to$ `orchestra`, một hàm phân bổ Vector khác được bẻ gãy kích hoạt. 
Đó chính là lúc sự tái định nghĩa được kiến tạo.

---

## 3. Bản Chất Của Tính Mập Mờ Giải Phẫu

Điều tiết lộ chua xót nhất từ thực nghiệm trên: Các ma trận tĩnh Embedded Matrix thuần túy (như BERT raw vector) **hoàn toàn câm điếc trong việc hiểu văn cảnh ngược**.
- Dù chữ `conductor` sau này đã được làm rạng tỏ là *Nhạc trưởng*. Thế nhưng, tọa độ điểm $\vec{v}_{\text{conductor}}$ khi rút thẳng từ Vocabulary Embeddings $E$, rồi được đối chiếu với $\vec{v}_{\text{waved}}$ là hoàn toàn tĩnh tại. Thống kê khoảng cách sẽ bị cứng ngắc (Frozen logic).

Tuy nhiên mạng ngôn ngữ học sâu BERT lại không chết bởi nguyên lý đó vì Embedded matrix này mới chỉ là "Tầng Trệt". Khi các giá trị này mớm dần qua nhiều Trụ Cột Attention Layers, một cơ chế truy ngược thời gian ngầm định (Backward context flow) sẽ ép cập nhật lại định dạng véc-tơ của từ `conductor` bằng cơ chế Self-attention có trọng số (Weighted dot matrix). Phân tích chuỗi tuần tự chính là tiền đề căn bản nhất để ta mở đường lên phân tích Context Vectors sau này.

---

## Tài liệu tham khảo

1. **Vaswani, A., et al. (2017).** *Attention is all you need.* NIPS. (Đặt ngòi nổ cho chuỗi thời gian phân đoạn ngữ đoạn).
2. **Peters, M. E., et al. (2018).** *Deep contextualized word representations.* NAACL (Mô hình hóa Context Dependency ELMo).
3. Tài liệu mô phỏng logic mạng học sâu *Cosine similarity in word sequences.*
