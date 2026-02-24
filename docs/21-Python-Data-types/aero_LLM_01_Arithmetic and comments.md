# Nhập môn Python: Các phép toán cơ bản và Chú thích (Arithmetic and Comments)

## Tóm tắt (Abstract)
Báo cáo này giới thiệu các khối xây dựng cơ bản nhất của ngôn ngữ lập trình Python: các toán tử số học và cơ chế chú thích mã nguồn. Chúng ta sẽ khám phá cách Python xử lý các phép tính từ đơn giản đến phức tạp, quy tắc về khoảng trắng, cơ chế hiển thị kết quả trong môi trường Notebook và thứ tự ưu tiên của các phép toán. Đây là những khái niệm nền tảng để thực hiện các tính toán định lượng trong xử lý ngôn ngữ tự nhiên và học sâu.

---

## 1. Các Phép Toán Cơ Bản
Trong Python, các ký hiệu toán học được diễn giải dựa trên ngữ cảnh và các quy tắc số học chuẩn:

### 1.1. Nhân và Lũy thừa
- **Phép nhân (`*`):** Sử dụng một dấu sao. Ví dụ: `4 * 5 = 20`.
- **Phép lũy thừa (`**`):** Sử dụng hai dấu sao liên tiếp. Ví dụ: `3 ** 2 = 9` (khác với ký hiệu `^` trong một số ngôn ngữ khác hoặc trong LaTeX).

### 1.2. Chia, Cộng và Trừ
- **Phép chia (`/`):** Kết quả luôn được trả về dưới dạng số thập phân (float). Ví dụ: `3 / 4 = 0.75`.
- **Phép cộng (`+`) và Phép trừ (`-`):** Thực hiện theo nguyên tắc toán học thông thường.

---

## 2. Quy tắc về Khoảng trắng và Khả năng Đọc mã
Python có cách tiếp cận linh hoạt đối với khoảng trắng trong các biểu thức toán học:
- **Tính linh hoạt:** `4*5` và `4 * 5` (có dấu cách) đều cho kết quả như nhau. Python hoàn toàn bỏ qua các khoảng trắng này trong quá trình thực thi.
- **Tính thẩm mỹ:** Việc sử dụng khoảng trắng giữa các toán tử và biến (đặc biệt là trong các đầu vào của hàm) giúp mã nguồn trở nên trực quan và dễ đọc hơn đối với con người.

---

## 3. Chú thích trong Mã nguồn (Comments)
Chú thích là những đoạn văn bản được Python bỏ qua khi thực thi nhưng lại cực kỳ quan trọng đối với lập trình viên:
- **Ký hiệu:** Sử dụng dấu thăng (`#`) để bắt đầu một dòng chú thích.
- **Công dụng:** Giải thích ý nghĩa của mã, ghi chú các tham số hoặc tạm thời vô hiệu hóa một đoạn mã (comment out).
- **Phím tắt:** Sử dụng `Command/Control + /` để nhanh chóng bật/tắt chú thích cho một dòng hoặc một khối mã.

---

## 4. Cơ chế Hiển thị và Hàm `print()`

### 4.1. Hiển thị trong Notebook
Trong một ô mã (code cell), mặc dù tất cả các dòng đều được thực thi, nhưng Python chỉ tự động hiển thị kết quả của **dòng cuối cùng**.

### 4.2. Hàm `print()`
Để hiển thị kết quả của nhiều phép tính trong cùng một ô mã, chúng ta sử dụng hàm `print()`:
- **Cấu trúc:** `print( biểu thức )`.
- **Lợi ích:** Cho phép kiểm soát chính xác những thông tin nào cần xuất ra màn hình để theo dõi quá trình tính toán.

---

## 5. Thứ tự Ưu tiên của các Phép toán (Order of Operations)
Python tuân thủ các quy tắc ưu tiên toán học chuẩn (PEMDAS):
1. **Dấu ngoặc đơn `()`:** Luôn được ưu tiên hàng đầu để nhóm các phép tính.
2. **Lũy thừa `**`:** Có ưu tiên cao hơn phép nhân, chia và cộng, trừ.
3. **Nhân/Chia:** Ưu tiên hơn Cộng/Trừ.

*Ví dụ:* `3 ** 2 + 1` sẽ cho kết quả là `10` (tính $3^2$ trước), trong khi `3 ** (2 + 1)` sẽ cho kết quả là `27` (tính $2+1$ trước).

---

## 6. Kết luận
Việc nắm vững các toán tử số học và cách thức Python diễn giải các biểu thức là bước đi đầu tiên nhưng vô cùng quan trọng. Sự thấu hiểu về thứ tự phép toán và cách sử dụng chú thích sẽ giúp lập trình viên viết được những đoạn mã không chỉ chính xác về mặt kỹ thuật mà còn rõ ràng về mặt logic.

---

## Tài liệu tham khảo (Citations)
1. Các phép toán cơ bản và chú thích trong Python dựa trên `aero_LLM_01_Arithmetic and comments.md`. Phân tích cú pháp toán tử và hàm `print()`.
