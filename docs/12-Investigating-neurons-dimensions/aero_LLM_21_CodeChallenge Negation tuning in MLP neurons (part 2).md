# Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 2)

## Tóm tắt (Abstract)
Báo cáo này tiếp tục quy trình nghiên cứu về "nơ-ron phủ định" bằng cách triển khai phân tích hồi quy logistic diện rộng trên quy mô hàng nghìn đơn vị nơ-ron. Chúng ta tập trung vào việc phát triển bộ phân loại (classifier) để định lượng khả năng phân biệt giữa các khái niệm Phủ định và Khẳng định của từng nơ-ron tại một tầng Transformer cụ thể. Quy trình bao gồm việc xử lý các thách thức về hội tụ số học, đánh giá độ chính xác của mô hình dự báo và trực quan hóa định tính thông qua bản đồ nhiệt văn bản nhằm xác thực tính chọn lọc chức năng của nơ-ron.

---

## 1. Triển khai Hồi quy Logistic diện rộng (Exercise 4)

### 1.1. Thiết lập Mô hình và Nhãn danh mục
Chúng ta xây dựng vector nhãn `category_labels` có kích thước tương ứng với tổng số mẫu:
- **Nhãn 0:** Các từ Khẳng định (Affirmations).
- **Nhãn 1:** Các từ Phủ định (Negations).
Hệ số Beta dương ($\beta > 0$) sẽ trực tiếp đồng nghĩa với việc nơ-ron nhạy cảm hơn với các cấu trúc phủ định.

### 1.2. Kỹ thuật Xử lý Hồi quy trên 5120 nơ-ron
Do tính chất đa dạng của dữ liệu hoạt hóa nơ-ron, một số đơn vị có thể gây lỗi cho thuật toán ước lượng phi tuyến. Các biện pháp kỹ thuật được áp dụng bao gồm:
1. **Tăng cường lặp:** Thiết lập `maxiter=3000` để hỗ trợ hội tụ trong các trường hợp phân tách dữ liệu phức tạp.
2. **Khối ngoại lệ (Try-Except):** Bảo vệ chương trình khỏi bị dừng đột ngột bởi các nơ-ron có dữ liệu quá nhiễu hoặc tách rời hoàn hảo (perfect separability), đồng thời đánh dấu các trường hợp này bằng giá trị `NaN`.
3. **Phân tách Tham số:** Chỉ tập trung vào hệ số góc (slope) của biến nhãn, loại bỏ tham số hằng số (intercept) vì nó chỉ đại diện cho mức hoạt hóa nền của nơ-ron.

### 1.3. Đánh giá Độ chính xác Dựa trên Xác suất
Với nơ-ron có hiệu ứng mạnh nhất (ví dụ: index 2022 tại tầng 13), chúng ta sử dụng hàm `predict()` để thu được xác suất logit. Áp dụng ngưỡng 0.5 để so sánh với nhãn thực tế, từ đó tính toán được **Độ chính xác (Accuracy)**. Kết quả thực nghiệm cho thấy một số nơ-ron đơn lẻ có khả năng phân loại đúng các mẫu phủ định với độ chính xác vượt trội so với mức ngẫu nhiên.

---

## 2. Trực quan hóa Bản đồ nhiệt Văn bản (Exercise 5)

### 2.1. Phân tích Định tính nơ-ron "Vô địch"
Để hiểu rõ hơn về hành vi của nơ-ron có hệ số Beta cao nhất, chúng ta ánh xạ hoạt hóa của nó lên chuỗi từ ngữ thực tế. Quy trình thực hiện:
- **Min-Max Scaling:** Chuẩn hóa biên độ hoạt hóa về dải $[0, 1]$ để phù hợp với thang màu (Colormap).
- **Bản đồ nhiệt (Heatmap):** Các từ phủ định như "not", "won't" thường xuyên kích hoạt mức "sáng" cao nhất trên bản đồ, trong khi các từ như "can", "will" trong cùng một ngữ cảnh lại có mức hoạt hóa thấp.

---

## 3. Thảo luận về Ý nghĩa Thống kê
Mặc dù nơ-ron có hệ số Beta lớn nhất thường có ý nghĩa thống kê cao, nhưng chúng không nhất thiết là nơ-ron có $p$-value nhỏ nhất. Sự khác biệt này đến từ sự cân bằng giữa quy mô hiệu ứng (effect size) và độ biến thiên (variance) của dữ liệu. Hiện tượng này nhấn mạnh tầm quan trọng của việc kết hợp cả chỉ số tham số ($\beta$) và độ tin cậy ($p$) trong Mechanistic Interpretability.

---

## 4. Kết Luận Phần 2
Chúng ta đã chứng minh được rằng lớp MLP chứa các đơn vị chức năng có khả năng hoạt động như "bộ phát hiện phủ định" (negation detectors). Trong giai đoạn tiếp theo, nghiên cứu sẽ mở rộng phạm vi ra toàn bộ 36 tầng của GPT-2 Large để tìm kiếm sự phân bổ của các nơ-ron này trong toàn bộ cấu trúc mạng.

---

## Tài liệu tham khảo (Citations)
1. Hồi quy Logistic xuyên tầng trên GPT-2 Large dựa trên `aero_LLM_21_CodeChallenge Negation tuning in MLP neurons (part 2).md`. Phân tích hệ số Beta và độ chính xác phân loại.
