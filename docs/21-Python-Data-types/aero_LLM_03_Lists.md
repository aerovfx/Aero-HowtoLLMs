
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
# Nhập môn Python: Danh sách và Kỹ thuật Chỉ mục (Lists and Indexing)

## Tóm tắt (Abstract)
Báo cáo này giới thiệu về "Danh sách" (List), một cấu trúc dữ liệu linh hoạt trong Python cho phép tổ chức và vận hành trên các tập hợp thông tin không đồng nhất. Chúng ta sẽ phân tích cơ chế "Chỉ mục dựa trên số 0" (Zero-based indexing), một khái niệm cốt lõi gây ra nhiều nhầm lẫn cho người mới bắt đầu. Nghiên cứu cũng thực hiện các thực nghiệm về việc lồng ghép danh sách (nested lists), sử dụng biến làm phần tử của danh sách và các kỹ thuật trình bày mã nguồn đa dòng để tối ưu hóa khả năng đọc hiểu của con người thông qua chú thích.

---

## 1. Cấu trúc và Định nghĩa Danh sách
Danh sách là một tập hợp các phần tử được bao quanh bởi dấu ngoặc vuông `[]` và ngăn cách nhau bởi dấu phẩy `,`.
- **Tính đa dạng:** Một danh sách không bắt buộc phải chứa các phần tử cùng kiểu. Nó có thể bao gồm số nguyên, số thập phân, chuỗi ký tự và thậm chí là các danh sách khác.
- **Tính linh hoạt:** Chúng ta có thể định nghĩa danh sách trên một dòng hoặc trải dài trên nhiều dòng để dễ dàng thêm các chú thích giải thích cho từng phần tử.

---

## 2. Cơ chế Chỉ mục dựa trên số 0 (Zero-based Indexing)

### 2.1. Cách Python đếm vị trí
Khác với ngôn ngữ tự nhiên, Python bắt đầu đếm các vị trí trong danh sách từ số **0**.
- **Phần tử thứ 1:** Nằm ở chỉ mục (index) `0`.
- **Phần tử thứ 2:** Nằm ở chỉ mục `1`.
- **Phần tử thứ n:** Nằm ở chỉ mục `n-1`.

### 2.2. Trích xuất phần tử
Để truy cập một giá trị cụ thể, chúng ta sử dụng cú pháp: `tên_danh_sách[vị_trí]`. Việc nắm vững quy tắc này là điều kiện tiên quyết để làm việc với các tensor và ma trận trong học sâu, nơi dữ liệu thường được tổ chức theo các chuỗi (sequences).

---

## 3. Danh sách lồng nhau và Trích xuất đa tầng

### 3.1. Danh sách lồng nhau (Nested Lists)
Python cho phép một phần tử trong danh sách bản thân nó lại là một danh sách khác. Điều này cho phép tạo ra các cấu trúc dữ liệu phân cấp phức tạp.

### 3.2. Indexing kép
Để truy cập một phần tử bên trong một danh sách con, chúng ta sử dụng chỉ mục liên tiếp:
*Ví dụ:* `danh_sách[2][0]` sẽ truy cập vào phần tử đầu tiên của danh sách nằm ở vị trí thứ 3 trong danh sách mẹ.

---

## 4. Tối ưu hóa Khả năng Đọc mã (Readability)
Việc khai báo danh sách trên nhiều dòng kết hợp với chú thích (`#`) là một thực hành tốt (best practice) trong lập trình khoa học:
- Giúp giải thích ý nghĩa của các tham số cấu hình (configuration settings).
- Giúp người đọc hiểu nhanh vai trò của từng phần tử trong một tập hợp dữ liệu lớn.
- Python sẽ bỏ qua các ký tự xuống dòng và khoảng trắng bên trong dấu ngoặc vuông, giữ cho logic của danh sách không thay đổi dù cách trình bày khác nhau.

---

## 5. Kết luận
Danh sách là một trong những công cụ mạnh mẽ và được sử dụng rộng rãi nhất trong hệ sinh thái Python. Việc thấu hiểu sự khác biệt giữa "giá trị phần tử" và "vị trí chỉ mục", cùng với khả năng vận hành trên các danh sách lồng nhau, đặt nền móng vững chắc cho việc xử lý các tập dữ liệu phức tạp trong nghiên cứu Trí tuệ nhân tạo.

---

## Tài liệu tham khảo (Citations)
1. Cấu trúc danh sách và kỹ thuật indexing trong Python dựa trên `aero_LLM_03_Lists.md`. Phân tích cơ chế zero-based indexing và lồng ghép danh sách.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
