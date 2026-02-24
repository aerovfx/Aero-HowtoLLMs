# Nhập môn Python: Câu lệnh Điều kiện If-Else và Logic Nhị phân (If-Else Statements)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về câu lệnh điều kiện `if-else`, cơ chế ra quyết định cơ bản nhất trong lập trình Python. Chúng ta phân tích cách thức Python đánh giá các biểu thức logic để trả về giá trị Boolean (`True` hoặc `False`), vai trò của từ khóa `else` trong việc xử lý các kịch bản thay thế, và kỹ thuật kết hợp nhiều điều kiện thông qua các toán tử `and` (và), `or` (hoặc). Nghiên cứu cũng giới thiệu toán tử Modulus (`%`) và ứng dụng của nó trong việc phân loại số chẵn/lẻ, đồng thời minh chứng khả năng lồng ghép (nesting) các cấu trúc điều khiển để xây dựng các thuật toán phân luồng phức tạp.

---

## 1. Nguyên lý Hoạt động của Câu lệnh If

### 1.1. Cú pháp và Đánh giá Boolean
Câu lệnh `if` kiểm tra tính đúng đắn của một biểu thức logic (conditional):
- **Cấu trúc:** `if điều_kiện:` theo sau là khối mã được thụt lề.
- **Cơ chế:** Nếu điều kiện trả về `True`, khối mã bên dưới sẽ được thực thi. Nếu trả về `False`, Python sẽ bỏ qua khối mã đó và tiếp tục chạy các dòng lệnh phía sau.
- **Ràng buộc:** Giống như hàm và vòng lặp, câu lệnh `if` bắt buộc phải có dấu hai chấm (`:`) và khối mã bên dưới phải được thụt lề đồng nhất.

### 1.2. Mở rộng với Else
Để xử lý trường hợp điều kiện không được thỏa mãn, chúng ta sử dụng `else`. Đây là cấu trúc "Nếu... thì... nếu không thì...", cho phép lập trình viên bao quát mọi kịch bản dữ liệu có thể xảy ra, tránh việc chương trình bị kết thúc đột ngột hoặc bỏ sót thông tin.

---

## 2. Logic Điều kiện Phức hợp (Conjunctive Conditionals)
Trong thực tế nghiên cứu AI, các điều kiện thường không đơn lẻ. Python cung cấp các từ khóa dành riêng để kết hợp logic:
- **`and` (Phép hội):** Toàn bộ biểu thức chỉ đúng khi **tất cả** các điều kiện thành phần đều đúng.
- **`or` (Phép tuyển):** Biểu thức đúng khi có **ít nhất một** điều kiện thành phần đúng.
Việc nắm vững logic này là nền tảng để xây dựng các bộ lọc dữ liệu (filters) và các cơ chế dừng sớm (early stopping) trong huấn luyện mô hình.

---

## 3. Toán tử Modulus (%) và Ứng dụng Phân loại
Toán tử Modulus trả về số dư của một phép chia.
- **Thuật toán:** $A \% B = R$ (Trong đó $R$ là số dư).
- **Phân loại số học:** Đây là phương pháp phổ biến nhất để kiểm tra tính chẵn lẻ của một số nguyên:
    - `n % 2 == 0`: Số chẵn (Even).
    - `n % 2 == 1`: Số lẻ (Odd).
Kỹ thuật này thường được dùng để thực hiện các tác vụ định kỳ trong vòng lặp (như lưu trữ checkpoint sau mỗi $k$ bước huấn luyện).

---

## 4. Cấu trúc Lồng ghép (Nested Flow Control)
Python cho phép lồng các câu lệnh `if-else` vào bên trong vòng lặp `for`.
- **Cơ chế:** Tại mỗi bước lặp, chương trình sẽ thực hiện kiểm tra điều kiện và đưa ra phản hồi tương ứng.
- **Ví dụ thực tiễn:** Duyệt qua một danh sách các tham số của mô hình và chỉ cập nhật (update) những trọng số thỏa mãn một ngưỡng (threshold) nhất định.

---

## 5. Kết luận
Câu lệnh điều kiện là "bộ não" điều phối dòng chảy của chương trình. Việc thấu hiểu logic Boolean cùng khả năng kết hợp các toán tử phức hợp và cấu trúc lồng ghép cho phép nhà nghiên cứu xây dựng các quy trình xử lý dữ liệu thông minh, có khả năng tự thích ứng với các biến số trong môi trường thực nghiệm LLM.

---

## Tài liệu tham khảo (Citations)
1. Cơ sở về câu lệnh điều kiện If-Else trong Python dựa trên `aero_LLM_02_If-else statements.md`. Phân tích logic Boolean, toán tử Modulus và kỹ thuật lồng ghép cấu trúc điều khiển.
