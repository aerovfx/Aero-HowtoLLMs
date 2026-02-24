# Toán học trong Học sâu: Kiểm định T (The T-Test)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về phương pháp kiểm định giả thuyết thống kê (T-test), một công cụ quan trọng để đánh giá tính hiệu quả của các kiến trúc và tham số trong học sâu. chúng ta phân tích cơ chế so sánh giữa giả thuyết không ($H_0$) và giả thuyết đối ($H_a$), công thức toán học dựa trên hiệu số trung bình chuẩn hóa theo độ lệch chuẩn, và ý nghĩa của giá trị $p$ trong việc xác định sự khác biệt có ý nghĩa thống kê. Nghiên cứu thực hiện các thực nghiệm trên thư viện SciPy để minh chứng cách thức quy trình kiểm định T giúp nhà nghiên cứu đưa ra các quyết định có cơ sở khoa học khi lựa chọn giữa các mô hình AI khác nhau.

---

## 1. Mục tiêu của Kiểm định T trong Deep Learning

Trong quá trình phát triển AI, chúng ta thường đặt câu hỏi: "Kiến trúc mô hình A có thực sự tốt hơn kiến trúc B?". Việc chỉ nhìn vào độ chính xác (accuracy) cao hơn ở một vài lượt chạy là chưa đủ để kết luận.
- **Kiểm định T:** Cho phép xác định xem sự khác biệt về hiệu năng giữa hai phân phối dữ liệu (ví dụ: độ chính xác của 20 lượt chạy mô hình A vs 20 lượt chạy mô hình B) là thực tế hay chỉ là kết quả ngẫu nhiên của biến thiên mẫu.
- **Giả thuyết Không ($H_0$):** Giả định rằng hai mô hình có hiệu năng như nhau. Mọi khác biệt quan sát được chỉ là do ngẫu nhiên.
- **Giả thuyết Đối ($H_a$):** Khẳng định có sự khác biệt thực sự và có ý nghĩa giữa hai mô hình.

---

## 2. Công thức và Cơ chế vận hành

Giá trị $t$ được tính toán dựa trên một nguyên lý đơn giản:
$$t = \frac{\bar{x} - \bar{y}}{s / \sqrt{n}}$$
Trong đó:
- **Tử số:** Khoảng cách giữa hai giá trị trung bình.
- **Mẫu số:** Độ lệch chuẩn được chuẩn hóa theo kích thước mẫu (nhiễu).
- **Nguyên lý cốt lõi:** Giá trị $t$ càng lớn khi sự khác biệt giữa các giá trị trung bình càng cao và độ biến thiên (nhiễu) bên trong mỗi nhóm mẫu càng thấp.

---

## 3. Diễn giải Kết quả: Ngưỡng ý nghĩa và Giá trị $p$

Sau khi có giá trị $t$, chúng ta quy đổi nó sang giá trị $p$ (p-value):
- **Ngưỡng 0.05:** Đây là ngưỡng phổ biến nhất trong khoa học. Nếu $p < 0.05$, có ít hơn 5% khả năng sự khác biệt này xảy ra do ngẫu nhiên. Chúng ta bác bỏ $H_0$ và kết luận mô hình có sự cải tiến thực sự.
- **Trường hợp $p \geq 0.05$:** Không đủ bằng chứng để kết luận sự khác biệt. Trong ngữ cảnh học sâu, điều này có nghĩa là kiến trúc mới không mang lại lợi ích thực chất so với kiến trúc cũ, mặc dù con số trung bình có thể trông cao hơn một chút.

---

## 4. Thực thi Kỹ thuật với SciPy

Nghiên cứu sử dụng hàm `stats.ttest_ind()` (Independent Samples T-test) từ thư viện SciPy:
- **Tính độc lập:** Hàm này phù hợp để so sánh hai nhóm dữ liệu không phụ thuộc vào nhau (ví dụ: hai mô hình được huấn luyện hoàn toàn tách biệt).
- **Tính đối xứng:** Dấu của giá trị $t$ (âm hay dương) chỉ phụ thuộc vào thứ tự đưa dữ liệu vào hàm, không ảnh hưởng đến giá trị $p$ và kết luận cuối cùng.
- **Trực quan hóa Dữ liệu:** Sử dụng kỹ thuật "jittering" (thêm nhiễu ngẫu nhiên vào trục X) giúp tách các điểm dữ liệu bị chồng lấp, cho phép quan sát phân phối thực tế một cách trực quan hơn trước khi thực hiện kiểm định.

---

## 5. Kết luận
Kiểm định T là "thanh bảo kiếm" giúp các kỹ sư AI tránh được bẫy của những cải tiến ảo do ngẫu nhiên. Trong thế giới của LLM, nơi mà chi phí huấn luyện cực kỳ đắt đỏ, việc sử dụng các công cụ thống kê như T-test để xác nhận tính hiệu quả của các siêu tham số (hyperparameters) trước khi triển khai quy mô lớn là vô cùng cần thiết. Thấu hiểu T-test là bước đệm để tiến tới những phương pháp so sánh phức tạp hơn như ANOVA hay tính toán kích thước hiệu ứng (effect size).

---

## Tài liệu tham khảo (Citations)
1. Ứng dụng kiểm định T trong so sánh hiệu năng mô hình dựa trên `aero_LL_12_The t-test.md`. Phân tích giả thuyết không, giá trị $p$, công thức thống kê và thực thi kiểm định độc lập trong SciPy.
