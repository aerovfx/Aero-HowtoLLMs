
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [03 Python Indexing and slicing](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Nhập môn Python: Kỹ thuật Chỉ mục Nâng cao (Advanced Indexing)

## Tóm tắt (Abstract)
Báo cáo này mở rộng kiến thức về kỹ thuật chỉ mục (indexing) trong Python, một kỹ năng cốt lõi để thao tác với dữ liệu trong học sâu. Chúng ta sẽ nghiên cứu về hàm `range()` và cơ chế "Cận trên loại trừ" (Exclusive upper bound), giải thích lý do tại sao Python bắt đầu đếm từ 0 và kết thúc trước giá trị chỉ định. Nghiên cứu cũng giới thiệu kỹ thuật chỉ mục âm (negative indexing) để truy cập dữ liệu từ cuối danh sách, cách sử dụng biến làm chỉ mục, và phân tích các lỗi phổ biến liên quan đến kiểu dữ liệu và cú pháp. Cuối cùng, một bài thực hành tổng hợp sẽ kiểm chứng khả năng trích xuất dữ liệu từ các cấu trúc lồng ghép phức tạp.

---

## 1. Hàm `range()` và Cơ chế Đếm trong Python

### 1.1. Đối tượng `range`
Hàm `range(n)` tạo ra một đối tượng có thể lặp lại (iterable) đại diện cho một dãy số. Để xem các con số này dưới dạng danh sách, ta cần chuyển đổi bằng hàm `list(range(n))`.

### 1.2. Cận trên loại trừ (Exclusive Upper Bound)
Một đặc điểm quan trọng của Python là khi đếm đến một số $n$, nó sẽ bắt đầu từ $0$ và dừng lại ở $n-1$.
- *Ví dụ:* `list(range(5))` sẽ trả về `[0, 1, 2, 3, 4]`.
Số $5$ được gọi là "cận trên loại trừ" vì nó được dùng làm mốc dừng nhưng không bao hàm trong kết quả cuối cùng. Quy tắc này đảm bảo rằng số lượng phần tử trả về luôn bằng đúng giá trị $n$ truyền vào.

---

## 2. Kỹ thuật Chỉ mục Linh hoạt

### 2.1. Chỉ mục âm (Negative Indexing)
Python cho phép truy cập các phần tử từ phía cuối danh sách bằng cách sử dụng số âm:
- `-1`: Chỉ phần tử cuối cùng.
- `-2`: Chỉ phần tử áp chót.
Kỹ thuật này cực kỳ hữu ích khi chúng ta cần lấy dữ liệu cuối chuỗi mà không biết trước độ dài của danh sách.

### 2.2. Chỉ mục bằng Biến (Variable-based Indexing)
Chúng ta có thể sử dụng giá trị của một biến để làm chỉ mục. Điều này giúp mã nguồn trở nên linh hoạt hơn, cho phép xác định vị trí trích xuất dữ liệu dựa trên các tham số hoặc kết quả tính toán trước đó thay vì viết cứng (hard-coding) một con số cụ thể.

---

## 3. Các lỗi thường gặp và Ràng buộc Kiểu dữ liệu

### 3.1. Lỗi "Object is not callable"
Lỗi này thường xảy ra khi lập trình viên sử dụng dấu ngoặc đơn `()` thay vì dấu ngoặc vuông `[]` để truy cập chỉ mục. Trong Python, `[]` dành cho truy xuất dữ liệu, còn `()` dành cho việc gọi hàm.

### 3.2. Ràng buộc số nguyên (Integer Constraint)
Chỉ mục bắt buộc phải là số nguyên (`int`). Việc sử dụng số thập phân (ví dụ: `3.0`) sẽ gây ra lỗi `TypeError`, ngay cả khi giá trị đó tương đương với một số nguyên. Trong trường hợp này, cần sử dụng hàm `int()` để ép kiểu trước khi truy xuất.

---

## 4. Thực hành: Trích xuất dữ liệu lồng ghép (Pop Quiz)
Thách thức đặt ra là trích xuất một giá trị nằm sâu bên trong một cấu trúc phức tạp: một danh sách chứa chuỗi, số và cả từ điển (dictionary).
- **Quy trình:**
    1. Xác định vị trí của từ điển trong danh sách mẹ (ví dụ: `list[4]`).
    2. Sử dụng khóa (key) của từ điển để truy xuất giá trị mong muốn (ví dụ: `list[4]['key']`).
Việc kết hợp liên tiếp các toán tử trích xuất cho phép ta "nội soi" vào bất kỳ tầng dữ liệu nào của mô hình.

---

## 5. Kết luận
Chỉ mục không chỉ đơn thuần là việc chọn một vị trí, mà còn là công nghệ để điều điều phối dòng dữ liệu. Hiểu rõ về cận trên loại trừ, chỉ mục âm và các ràng buộc về kiểu dữ liệu sẽ giúp lập trình viên viết mã an toàn hơn và xử lý được những cấu trúc dữ liệu đa tầng thường gặp trong kiến trúc Transformer.

---

## Tài liệu tham khảo (Citations)
1. Kỹ thuật chỉ mục nâng cao trong Python dựa trên `aero_LLM_01_Indexing.md`. Phân tích hàm `range()`, cơ chế loại trừ cận trên và trích xuất dữ liệu lồng ghép.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[Nhập môn Python: Kỹ thuật Chỉ mục Nâng cao (Advanced Indexing)](aero_LLM_01_Indexing.md)** | [Xem bài viết →](aero_LLM_01_Indexing.md) |
| [Nhập môn Python: Kỹ thuật Cắt lát Danh sách (Slicing)](aero_LLM_02_Slicing.md) | [Xem bài viết →](aero_LLM_02_Slicing.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
