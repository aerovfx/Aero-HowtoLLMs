# Nhập môn Python: Biến Boolean và Logic Nhị phân (Booleans and Logic)

## Tóm tắt (Abstract)
Báo cáo này giới thiệu về "Boolean", một kiểu dữ liệu nền tảng trong lập trình được đặt tên theo nhà logic học George Boole. Kiểu Boolean đại diện cho tính đúng/sai của một mệnh đề, phục vụ như là cơ sở cho các cổng logic và mạch tính toán. Chúng ta sẽ khám phá các từ khóa dành riêng trong Python (`True`, `False`), phân biệt giữa phép gán (`=`) và phép so sánh (`==`), cùng với các quy tắc kết hợp mệnh đề thông qua các toán tử logic `and` và `or`. Nghiên cứu cũng nhấn mạnh tầm quan trọng của việc sử dụng dấu ngoặc đơn để duy trì sự minh bạch trong các cấu trúc so sánh phức tạp.

---

## 1. Bản chất của Biến Boolean
Biến Boolean chỉ có thể nhận một trong hai giá trị duy nhất: **Đúng** (`True`) hoặc **Sai** (`False`). Trong máy tính, chúng thường được mã hóa lần lượt bằng các giá trị số là `1` và `0`.
- **Từ khóa dành riêng (Reserved Keywords):** Python yêu cầu viết hoa chữ cái đầu (`True`, `False`). Các biến thể viết thường hoặc viết hoa toàn bộ sẽ không được nhận diện là kiểu dữ liệu logic.

---

## 2. Các Phép So sánh và Truy vấn Logic

### 2.1. Phép bằng (`==`) vs Phép gán (`=`)
Đây là một trong những điểm gây nhầm lẫn nhất đối với người mới học lập trình:
- **Phép gán (`=`):** Là một phát biểu mang tính khẳng định (ví dụ: gán danh sách cho một biến).
- **Phép bằng (`==`):** Là một câu hỏi truy vấn đối với Python. "Liệu A có bằng B hay không?". Câu trả lời luôn trả về một giá trị Boolean.

### 2.2. Các toán tử so sánh khác
- **So sánh biên:** `<` (nhỏ hơn), `>` (lớn hơn), `<=` (nhỏ hơn hoặc bằng), `>=` (lớn hơn hoặc bằng).
- **Tính linh hoạt về kiểu:** Python đủ linh hoạt để nhận diện `10` (int) bằng với `10.0` (float) khi thực hiện phép so sánh `==`, mặc dù kiểu dữ liệu của chúng khác nhau.

---

## 3. Kết hợp Mệnh đề (Logical Connectors)
Chúng ta có thể thực hiện các so sánh liên hợp (conjunctive comparisons) để kiểm tra nhiều điều kiện đồng thời:
- **Toán tử `and`:** Trả về `True` chỉ khi **tất cả** các mệnh đề thành phần đều đúng.
- **Toán tử `or`:** Trả về `True` nếu có **ít nhất một** mệnh đề thành phần đúng.

---

## 4. Kỹ thuật Lập trình và Khả năng Đọc hiểu

### 4.1. Vai trò của Dấu ngoặc đơn `()`
Trong các biểu thức logic dài, việc sử dụng dấu ngoặc đơn giúp cô lập và nhóm các điều kiện một cách trực quan. Mặc dù Python có thể thực thi mã mà không cần dấu ngoặc (dựa trên thứ tự ưu tiên), nhưng dấu ngoặc giúp giảm bớt gánh nặng nhận thức cho lập trình viên và ngăn ngừa các sai sót logic tiềm ẩn.

### 4.2. Gán kết quả Logic cho Biến
Kết quả của một phép toán phức tạp có thể được lưu trữ vào một biến kiểu `bool`:
*Ví dụ:* `outcome = (x * 2 == y)`.
Việc lưu trữ này rất hữu ích để sử dụng làm điều kiện kiểm soát luồng (flow control) trong các đoạn mã tiếp theo của chương trình.

---

## 5. Kết luận
Booleans là "ngôn ngữ" của các quyết định trong lập trình. Việc nắm vững cách xây dựng và kết hợp các mệnh đề logic là nền tảng để xây dựng các thuật toán có khả năng phản ứng linh hoạt với dữ liệu, đồng thời là bước chuẩn bị quan trọng cho việc nghiên cứu các cơ chế chú ý (attention mechanisms) và các phép toán logic trong transformers.

---

## Tài liệu tham khảo (Citations)
1. Lý thuyết Booleans và logic so sánh trong Python dựa trên `aero_LLM_04_Booleans.md`. Phân tích các toán tử `==`, `and`, `or` và kiểu dữ liệu `bool`.
