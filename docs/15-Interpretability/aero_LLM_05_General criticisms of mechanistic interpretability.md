# Những Lời Chỉ Trích Tổng Quát Về Diễn Giải Cơ Chế (Mechanistic Interpretability)

## Tóm tắt

Bất kỳ hệ tư tưởng khoa học nào được sinh ra đều cần thiết phải được mài giũa bằng sư phản biện. Diễn giải Cơ chế (Mechanistic Interpretability) tự hào đóng vai trò tiên phong mổ xẻ "hộp đen" Mô hình Ngôn ngữ Lớn, nhưng nó cũng đối diện với các hạn chế phương pháp luận khắc nghiệt. Bài viết này tổng hợp và phân tích 4 luồng chỉ trích phổ biến mang tính tổng quát nhắm vào lĩnh vực này, giải mã chúng dưới góc độ đại số không gian và lý thuyết tính toán, đồng thời nhấn mạnh vai trò của phản biện khắt khe trong việc xúc tiến tiến trình minh bạch hoá Trí tuệ Nhân tạo.

---

## 1. Vấn Đề "Hái Anh Đào" Thống Kê (Cherry Picking)

"Cherry picking" là một thuật ngữ thống kê chỉ việc chọn lọc thiên kiến. Trong Mech Interp, khi mô hình sở hữu hàng triệu hoặc tỷ tham số (parameters), không gian tìm kiếm là cực kỳ bao la.
- **Rủi ro phi tuyến:** Một nhà nghiên cứu thiết lập Hook và trích xuất lượng dữ liệu kích hoạt khổng lồ ($h_l$), sau đó chạy các thuật toán trích xuất mạch. Họ vô tình hay cố ý chỉ công bố một phần tỷ lệ siêu nhỏ những phân tích "có vẻ hợp lý" hoặc vừa vặn hoàn hảo với giả thuyết nhân quả ban đầu.
- **Hệ quả phân phối giả vụn (Statistical flukes):** Việc chắt lọc này có nguy cơ biến một nhiễu loạn ngẫu nhiên ở bề mặt xác suất (noise) thành một khám phá nền tảng, gây ngộ nhận về cách LLMs mã hóa thông tin.

---

## 2. Thách Thức Trong Khả Năng Tái Phục Hồi (Reproducibility)

Tái lập kết quả lập lại (Reproducibility problem) là cuộc khủng hoảng xảy ra trong rất nhiều nhánh khoa học như Y khoa hay Tâm lý học, và Mech Interp không phải là ngoại lệ.
- Một nguyên tắc toán học hay ma trận đặc trưng (feature representation) có thể hoạt động hoàn hảo trên một bộ dữ liệu từ điển văn bản cấu trúc sẵn, nhưng lại phân rã thành các thông số dị thường (anormal vector mapping) khi được thử nghiệm với một chuỗi tokens phi cấu trúc khác.
- Cùng một phương thức cắt mạch (Circuit ablation) tạo ra biểu hiện A ở mô hình LLaMA, nhưng đem kết quả hoàn toàn chệch hướng trên mô hình GPT hay Claude. Sự thiếu nhất quán này đặt ra dấu hỏi lớn về độ chín (maturity) của quy chuẩn phân tích đo lường học sâu.

---

## 3. Khủng Hoảng Của Chủ Nghĩa Hoàn Nguyên Số Liệu (Reductionism Limit)

Mech Interp áp dụng tư duy chủ nghĩa hoàn nguyên (Bottom-up reductionism) - đi từ các đơn vị đo bé nhất: tham số đơn lẻ, nơ-ron rời rạc, điểm chú ý từng token một. 
- **Thiếu bức tranh tổng thể đa cực:** Nếu thông tin chứa đựng trong LLM thực chất không hề nằm gọn cấu trúc tĩnh (localized), mà được định tuyến phi tuyến trong vùng siêu dữ liệu (Superposition), phân tách đa chiều hoặc các khối tiềm ẩn (Latent distribution), thì việc mổ lớp một cách cơ học sẽ thất bại.
- **Nghịch lý vật lý:** Không thể hiểu được vẻ đẹp toàn phác của âm nhạc hay các đặc tính tính cách xã hội loài người nếu chỉ dựa vào việc mổ các hạt phân tử điện tích cấu thành chất xám một nơ-ron sinh học. Các cấu trúc ngôn ngữ trỗi dậy (Emergent capabilities) ở LLM chịu chi phối bởi mạng đồ thị kết tinh phức tạp, thứ không thể đọc trọn vẹn ở phép tuyến tính bậc thấp.

Tuy vậy, việc đánh giá vội vàng rồi loại bỏ hoàn nguyên chủ nghĩa mang tính rủi ro, vì chính vật lý lượng tử hay di truyền học DNA cũng phải khởi sinh từ những mẫu vật vô cùng vi biên.

---

## 4. Thiếu Sự Phổ Quát (Lack of Universality) Giữ Mô Hình Đồ Chơi và Hệ Thống Khổng Lồ

Nhiều trung tâm cố gắng chứng minh lý thuyết trên các cấu trúc hộp đồ chơi (Toy Models / Sparse Autoencoders lớp nông) chỉ có vài ngàn node mã hóa.
- Sự phê phán tập trung vào giả định: Liệu một động thái (motif) toán học ở mạng con vài ngàn tham số có tỷ lệ thuận hay sao chép thuật toán chính xác lên kiến trúc Transformer đa tầng siêu khổng lồ (với kích thước không gian vector nội bộ, $d_{k}, d_{v}$ và $M_{head}$ cao cấp gấp vạn lần)?
- Điều này tương tự như việc áp dụng Vật lý Học Cổ Điển Newton (Newtonian Mechanics trong một môi trường chân không, không có xung đột lượng tử) vào việc giải thích hạt vũ trụ hố đen. Cơ học tinh gọn ban đầu không sai, nhưng hoàn toàn bất biến trong ứng dụng ở quy mô bất đối xứng.

Tuy nhiên, nếu cơ học Newton bị hủy bỏ thì cơ học Lượng tử cũng không bao giờ có nền móng chào đời. Mô hình đồ chơi chính là bậc thang tiệm cận.

---

## 5. Kết Luận

Có những tiếng gắt gao rằng Mechanistic Interpretability cho đến hiện tại chưa thực sự mang lại một tiến công phòng thủ AI Safety (Safety guardrail) nào có tính trực quan và mang tính tác động lớn tới các luồng thương mại. Nhưng việc đón nhận sự khuyết thiếu (như Cherry picking, chủ nghĩa Hoàn nguyên quá đà hay Tái lập hệ thống) không tạo ra lý do để từ bỏ mảng ghép khó nhất của Trí Tuệ Nhân Tạo. Nhìn nhận thẳng thắn các chỉ trích này là nguyên lý bắt buộc để chuyển hóa lĩnh vực non trẻ này thành một khuôn khổ toán thấu hiểu vững chãi.

---

## Tài liệu tham khảo

1. **Olah, C., et al. (2020).** *Zoom In: An Introduction to Circuits.* Distill.
2. **Ioannidis, J. P. A. (2005).** *Why Most Published Research Findings Are False.* PLoS Medicine.
3. **Elhage, N., et al. (2021).** *A Mathematical Framework for Transformer Circuits.* Anthropic.
4. **Smelser, N. J., & Baltes, P. B. (Eds.). (2001).** *International Encyclopedia of the Social & Behavioral Sciences.* (Reductionism discussions).
