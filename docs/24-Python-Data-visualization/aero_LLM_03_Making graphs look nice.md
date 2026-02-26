
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [24 Python Data visualization](../index.md)

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
# Nhập môn Python: Kỹ thuật Tinh chỉnh và Thẩm mỹ Biểu đồ (Making Graphs Look Nice)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu các phương pháp nâng cao để tối ưu hóa giao diện biểu đồ trong Matplotlib, chuyển đổi các sơ đồ kỹ thuật thô thành các tài liệu trực quan chuyên nghiệp. Chúng ta phân tích việc sử dụng hệ thống gán nhãn, tích hợp ngôn ngữ soạn thảo toán học LaTeX, và cơ chế thiết lập giới hạn trục tọa độ động. Nghiên cứu cũng giới thiệu hàm `gca().set()`, một công cụ mạnh mẽ để quản lý tập trung các thông số thẩm mỹ, đồng thời khảo sát các hệ thống định nghĩa màu sắc chuyên dụng như RGB, Hex và màu định danh. Đây là những kỹ năng thiết yếu để trình bày các kết quả nghiên cứu LLM một cách thu hút và dễ hiểu.

---

## 1. Hệ thống Nhãn dán và Soạn thảo Toán học

### 1.1. Tiêu đề và Nhãn trục
Việc thiếu nhãn trục là một sai sót nghiêm trọng trong báo cáo khoa học. Matplotlib cung cấp các hàm `plt.xlabel()`, `plt.ylabel()` và `plt.title()` để định nghĩa ngữ cảnh cho dữ liệu.

### 1.2. Tích hợp LaTeX
Python cho phép sử dụng cú pháp LaTeX bằng cách bao bọc chuỗi ký tự trong dấu đô la `$`.
- **Lợi ích:** Cho phép hiển thị các ký hiệu toán học phức tạp như số mũ (`x^3`), chỉ số dưới, hoặc các ký tự Hy Lạp. Điều này giúp các chú giải đồ thị trở nên đồng nhất với các công thức toán học trong bài báo khoa học.

---

## 2. Quản lý Không gian Hiển thị Động

### 2.1. Giới hạn Trục (Axis Limits)
Mặc định, Matplotlib tự động mở rộng trục tọa độ để bao quát toàn bộ dữ liệu, đôi khi tạo ra các khoảng trắng không cần thiết. Hàm `plt.xlim()` và `plt.ylim()` cho phép lập trình viên kiểm soát chính xác phạm vi quan sát.

### 2.2. Kỹ thuật Soft Coding cho Giới hạn
Thay vì nhập các con số cố định, một kỹ thuật chuyên nghiệp là gán giới hạn dựa trên giá trị cực biên của dữ liệu đầu vào:
`plt.xlim([x[0], x[-1]])`
- **Lợi ích:** Khi tập dữ liệu thay đổi quy mô, biểu đồ sẽ tự động điều chỉnh khung nhìn mà không cần can thiệp thủ công vào mã nguồn.

---

## 3. Quản lý Tập trung với `GCA` (`Get Current Axis`)
Hàm `plt.gca().set()` cho phép cấu hình đồng thời nhiều tham số (nhãn, tiêu đề, giới hạn) trong một câu lệnh duy nhất.
- **Tính khoa học:** Việc tổ chức các tham số này theo từng dòng có thụt lề giúp mã nguồn trở nên sạch sẽ, dễ bảo trì và giảm thiểu việc lặp lại tên hàm `plt` nhiều lần.

---

## 4. Hệ màu và Độ dày Đường kẻ (Colors & Line Width)

### 4.1. Định nghĩa Màu sắc Đa dạng
Nghiên cứu chỉ ra ba phương thức chính để tùy biến màu sắc:
1. **Tọa độ RGB:** Một danh sách gồm ba số thực từ 0 đến 1 (ví dụ: `[0.7, 0.3, 0.9]`).
2. **Mã Hex:** Chuỗi ký tự thập lục phân (ví dụ: `"#D6690A"`).
3. **Màu định danh:** Sử dụng các tên tiếng Anh phổ biến được thư viện hỗ trợ (ví dụ: `"sky blue"`).

### 4.2. Độ dày Đường kẻ (`lineWidth`)
Tham số `lineWidth` (hoặc `lw`) cho phép thay đổi độ đậm nhạt của nét vẽ. Kỹ thuật kết hợp `lineWidth` với biến chạy của vòng lặp `for` giúp tạo ra các hiệu ứng thị giác tăng dần, hỗ trợ việc phân cấp thông tin trong các biểu đồ có nhiều đường biểu diễn.

---

## 5. Kết luận
Sự chuyên nghiệp của một nhà nghiên cứu AI không chỉ thể hiện ở thuật toán mà còn ở cách họ trình bày kết quả. Việc làm chủ các kỹ thuật từ LaTeX, GCA đến hệ màu đa dạng giúp biến các con số khô khan thành những câu chuyện trực quan đầy sức thuyết phục, góp phần nâng cao giá trị truyền tải của các công trình thực nghiệm LLM.

---

## Tài liệu tham khảo (Citations)
1. Kỹ thuật tinh chỉnh thẩm mỹ biểu đồ với Matplotlib dựa trên `aero_LL_03_Making graphs look nice.md`. Phân tích gán nhãn LaTeX, kỹ thuật GCA, quản lý màu sắc RGB/Hex và hiệu ứng độ dày đường kẻ.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
