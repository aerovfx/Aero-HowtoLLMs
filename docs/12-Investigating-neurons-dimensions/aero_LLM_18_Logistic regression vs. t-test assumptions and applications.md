# Đối chiếu Hồi quy Logistic và Kiểm định T-test: Giả định và Ứng dụng

## Tóm tắt (Abstract)
Báo cáo này thực hiện một phân tích đối chiếu giữa Hồi quy Logistic và Kiểm định T-test (T-test) – hai phương pháp thống kê phổ biến nhất trong việc phân tích hành vi nơ-ron. Thông qua việc phân tích cấu trúc toán học và thực nghiệm trên dữ liệu giả lập, chúng ta làm rõ các kịch bản sử dụng tối ưu cho từng phương pháp. Mặc dù có những điểm khác biệt về giả định (đặc biệt là giả định về phân phối chuẩn), kết quả thực nghiệm bộc lộ một sự tương hợp (concordance) mạnh mẽ về ý nghĩa thống kê giữa hai phương pháp khi áp dụng trên cùng một bộ dữ liệu hoạt hóa nơ-ron.

---

## 1. Phân tích Cấu trúc Toán học

### 1.1. Hồi quy Logistic
- **Mục tiêu:** Xây dựng phương trình toán học để tạo ra xác suất một mẫu dữ liệu thuộc về một trong hai danh mục.
- **Kết quả:** Cung cấp các tham số ($\beta$) dùng để xây dựng mô hình dự báo.
- **Ứng dụng:** Thích hợp khi cần dự đoán nhãn (label) của token dựa trên hoạt hóa nơ-ron hoặc khi phân tích đa biến (nhiều nơ-ron cùng lúc).

### 1.2. Kiểm định T-test
- **Mục tiêu:** Đo lường sự khác biệt chuẩn hóa giữa giá trị trung bình của hai nhóm dữ liệu (ví dụ: Nouns vs. Verbs).
- **Kết quả:** Cung cấp trị số thống kê $t$ – một thước đo về quy mô hiệu ứng (effect size).
- **Ứng dụng:** Thích hợp để xác định sự khác biệt có ý nghĩa giữa hai loại token trong một nơ-ron cụ thể.

---

## 2. So sánh Giả định và Đặc tính

| Tiêu chí | Hồi quy Logistic | Kiểm định T-test |
| :--- | :--- | :--- |
| **Giả định Phân phối** | Không yêu cầu phân phối chuẩn. | Yêu cầu dữ liệu có phân phối chuẩn. |
| **Giả định Quần thể** | Mẫu đến từ một quần thể có tỉ lệ thuộc về danh mục thay đổi theo hoạt hóa. | Hai nhóm đến từ hai quần thể thực sự khác biệt. |
| **Khả năng dự báo** | Cho phép dự đoán xác suất ở cấp độ từng mẫu dữ liệu đơn lẻ. | Không thiết kế để dự đoán cho mẫu đơn lẻ. |
| **Tính mở rộng** | Dễ dàng mở rộng cho nhiều biến độc lập. | Giới hạn trong một biến phụ thuộc và hai nhóm. |

---

## 3. Thực nghiệm Đối chiếu trên Dữ liệu Giả lập

### 3.1. Sự Tương hợp về Hệ số và Trị số T
Thực nghiệm mô phỏng dữ liệu với quy mô hiệu ứng thay đổi cho thấy sự tương quan cực kỳ chặt chẽ giữa hệ số $\beta$ của Hồi quy Logistic và trị số $t$.
- **Lưu ý về tính ổn định:** Tại các giá trị cực hạn (nơi dữ liệu hai nhóm tách biệt hoàn toàn), Hồi quy Logistic bộc lộ sự không ổn định về mặt số học (numerical instability) do tính chất phi tuyến của thuật toán ước lượng, trong khi T-test vẫn giữ được tính ổn định tuyến tính.

### 3.2. Sự Tương hợp về Giá trị P (P-values)
Sử dụng giá trị $-\log(p)$ để trực quan hóa toàn bộ dải phân phối. Kết quả khẳng định:
- Khi kết quả không có ý nghĩa thống kê ở phương pháp này, nó cũng thường không có ý nghĩa ở phương pháp kia.
- Khi một nơ-ron được xác định là "có ý nghĩa" ($p < 0.05$), cả hai phương pháp đều đưa ra kết luận đồng nhất.

---

## 4. Các điểm lưu ý Kỹ thuật

1. **Thứ tự Nhãn (Label Ordering):** Việc thay đổi thứ tự nhập dữ liệu (ví dụ: Noun trước Verb hoặc ngược lại) chỉ làm thay đổi dấu (sign) của hệ số $\beta$ hoặc giá trị $t$ mà không làm thay đổi bản chất thống kê.
2. **Tính Tương đương Toán học:** Kiểm định T-test mẫu cặp (Paired samples T-test) về mặt toán học là tương đương với kiểm định T-test một mẫu (One-sample T-test) trên hiệu số giữa các cặp dữ liệu.

---

## 5. Kết Luận
Việc lựa chọn giữa Hồi quy Logistic và T-test phụ thuộc vào câu hỏi nghiên cứu:
- Chọn **Hồi quy Logistic** khi muốn xây dựng "bộ giải mã" (decoder) để dự đoán danh mục từ hoạt hóa.
- Chọn **T-test** khi muốn kiểm chứng nhanh sự khác biệt đặc tính giữa các nhóm nơ-ron hoặc các đầu Attention.

---

## Tài liệu tham khảo (Citations)
1. Đối chiếu Hồi quy Logistic và T-test trên LLM dựa trên `aero_LLM_18_Logistic regression vs. t-test assumptions and applications.md`. Phân tích tương hợp p-value và tính ổn định số học.
