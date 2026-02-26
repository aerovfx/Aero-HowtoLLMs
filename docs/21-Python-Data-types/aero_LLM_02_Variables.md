
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [21 Python Data types](../index.md)

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
# Nhập môn Python: Biến và các Kiểu dữ liệu Cơ bản (Variables and Data Types)

## Tóm tắt (Abstract)
Báo cáo này trình bày cách thức Python tổ chức và lưu trữ thông tin thông qua khái niệm "biến" (variables). Chúng ta tập trung vào ba kiểu dữ liệu nền tảng: số nguyên (int), số thập phân (float) và chuỗi ký tự (str). Nghiên cứu đi sâu vào các quy tắc đặt tên biến theo tiêu chuẩn công nghiệp (Camel case, Snake case), sự phân biệt tinh tế giữa các kiểu dữ liệu có vẻ giống nhau (ví dụ: `10` vs `10.0`), và cách các toán tử số học thay đổi hành vi tùy thuộc vào kiểu dữ liệu của biến. Đây là những khối kiến thức thiết yếu để quản lý hàng tỷ thông số trong các mô hình ngôn ngữ lớn.

---

## 1. Khái niệm về Biến và Phép gán (Assignment)
Biến đóng vai trò như các thùng chứa thông tin, giúp chúng ta tái sử dụng dữ liệu nhiều lần mà không cần viết lại toàn bộ giá trị.
- **Cú pháp:** `tên_biến = giá_trị`.
- **Đặc điểm:** Khi thực hiện gán giá trị cho một biến, Python sẽ thực thi mã nhưng không hiển thị kết quả ra màn hình (Output). Để xem nội dung bên trong, ta cần gọi tên biến đó hoặc sử dụng hàm `print()`.

---

## 2. Quy tắc và Phong cách đặt tên Biến

### 2.1. Quy tắc bắt buộc
- **Ký tự:** Chỉ bao gồm chữ cái, chữ số và dấu gạch dưới (`_`).
- **Khởi đầu:** Tên biến **không được phép** bắt đầu bằng chữ số.
- **Ký tự đặc biệt:** Không được chứa khoảng trắng hoặc các ký hiệu như `?`, `!`, `@`,... ngoại trừ dấu gạch dưới.
- **Độ nhạy chữ hoa/chữ thường:** `MyVariable` và `myvariable` là hai biến hoàn toàn khác nhau trong Python.

### 2.2. Phong cách (Naming Styles)
- **Camel Case:** Viết hoa chữ cái đầu của mỗi từ mới (ví dụ: `targetTokenIndex`). Thường tạo cảm giác trực quan như các bướu lạc đà.
- **Snake Case:** Sử dụng dấu gạch dưới để phân tách các từ (ví dụ: `target_token_index`).
*Lưu ý:* Việc lựa chọn phong cách nào không ảnh hưởng đến hiệu suất mã nguồn, nhưng tính nhất quán là yếu tố then chốt giúp mã dễ đọc cho cộng đồng.

---

## 3. Các Kiểu dữ liệu Cơ bản

### 3.1. Số nguyên (`int`) và Số thập phân (`float`)
Mặc dù về mặt toán học $10$ và $10.0$ có giá trị bằng nhau, nhưng trong lập trình:
- **`int`:** Các số nguyên hoàn chỉnh. Tiết kiệm bộ nhớ hơn.
- **`float`:** Số có dấu phẩy động (có phần thập phân). Chiếm nhiều bộ nhớ hơn và có các giới hạn về độ chính xác.
Một số hàm trong Python yêu cầu đầu vào bắt buộc là `int` (ví dụ: số lần lặp), và sẽ báo lỗi nếu nhận vào một `float`.

### 3.2. Chuỗi ký tự (`str`)
Đại diện cho văn bản, được bao quanh bởi dấu nháy đơn (`'`) hoặc nháy đôi (`"`).
- **Phép toán trên chuỗi:** 
    - Nhân một chuỗi với một số nguyên (`'abc' * 3`) sẽ tạo ra sự lặp lại (`'abcabcabc'`).
    - Python không cho phép nhân chuỗi với một số thập phân vì khái niệm "lặp lại 2.5 lần" không có ý nghĩa logic trong xử lý chuỗi.
- **Chuyển đổi kiểu:** Sử dụng hàm `float()` hoặc `str()` để chuyển đổi qua lại giữa định dạng số và định dạng văn bản (ví dụ: chuyển `'2.4'` thành `2.4`).

---

## 4. Kiểm tra Kiểu dữ liệu với hàm `type()`
Để xác định bản chất của một biến, chúng ta sử dụng hàm `type()`. Việc kết hợp `print(type(biến))` là một kỹ thuật gỡ lỗi (debugging) quan trọng để đảm bảo dữ liệu đang ở định dạng mong muốn trước khi thực hiện các phép toán phức tạp.

---

## 5. Kết luận
Biến không chỉ là nơi lưu trữ mà còn là công cụ để biểu đạt logic của chương trình một cách rõ ràng. Việc thấu hiểu sự khác biệt giữa các kiểu dữ liệu và tuân thủ các quy tắc đặt tên sẽ giúp xây dựng những hệ thống mã nguồn bền vững, dễ bảo trì và giảm thiểu các lỗi không mong muốn khi làm việc với các tensor dữ liệu quy mô lớn.

---

## Tài liệu tham khảo (Citations)
1. Cơ sở về biến và kiểu dữ liệu trong Python dựa trên `aero_LLM_02_Variables.md`. Phân tích định danh biến, quy tắc đặt tên và các kiểu số học/chuỗi.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
