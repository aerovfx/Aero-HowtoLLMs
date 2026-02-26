
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [12 Investigating neurons dimensions](../index.md)

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
# Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 1)

## Tóm tắt (Abstract)
Báo cáo này bắt đầu một thử thách nghiên cứu đa giai đoạn nhằm định lượng mối quan hệ giữa độ dài của token (tính theo số ký tự) và cường độ hoạt hóa của các nơ-ron MLP trên toàn bộ các tầng của mô hình GPT-Neo. Trong phần này, chúng ta tập trung vào việc thiết lập hệ thống Hooks đa tầng, chuẩn bị dữ liệu từ bộ dữ liệu FineWeb và thực hiện phân tích thống kê về phân phối độ dài token. Kết quả thiết lập cho thấy khả năng trích xuất đồng thời hoạt hóa từ 12 khối Transformer với quy mô 3072 nơ-ron mỗi khối, tạo điều kiện cho các phân tích so sánh liên tầng ở các giai đoạn sau.

---

## 1. Mở Đầu (Introduction)
Các token trong LLM không có độ dài đồng nhất: một số chỉ là một ký tự đơn giản, trong khi số khác đại diện cho các từ phức tạp dài nhiều ký tự. Câu hỏi đặt ra là: Liệu mô hình có dành nhiều "năng lượng tính toán" (hoạt hóa nơ-ron) hơn cho các token dài - vốn thường mang nhiều thông tin ngữ nghĩa hơn - hay không? Báo cáo này xây dựng khung thực nghiệm để kiểm chứng giả thuyết này thông qua phân tích dải tần hoạt hóa (histograms).

---

## 2. Thiết lập Hệ thống trích xuất Đa tầng

### 2.1. Hooks Đa mục tiêu
Khác với các thực nghiệm trước chỉ tập trung vào một tầng đơn lẻ, nghiên cứu này yêu cầu quan sát hành vi của mô hình theo chiều sâu. Một vòng lặp `for` được sử dụng để cấy 12 Hooks vào thành phần `c_fc` (MLP expansion) của tất cả các khối Transformer. Mỗi Hook lưu trữ dữ liệu vào một `Dictionary` với key định danh duy nhất (ví dụ: `MLP_0`, `MLP_1`,...), cho phép chụp lại trạng thái toàn cục của mô hình trong một lượt forward-pass duy nhất.

### 2.2. Hiệu năng tính toán (CPU vs. GPU)
Dù việc vận hành mô hình GPT-Neo 125M trên CPU chỉ mất khoảng 1 phút cho 8192 tokens, báo cáo khuyến nghị sử dụng GPU để giảm thời gian xuống mức vài giây. Điều này đặc biệt quan trọng khi mở rộng quy mô sang các mô hình lớn hơn (như GPT-Neo 1.3B) ở các giai đoạn sau của thử thách.

---

## 3. Phân tích Dữ liệu Đầu vào (FineWeb Dataset)

### 3.1. Thu thập và Tokenization
Dữ liệu được lấy từ FineWeb cho đến khi đạt chính xác 8192 tokens. Con số này được chọn để khớp hoàn hảo với cấu trúc batch $16 \times 512$, tối ưu hóa việc sử dụng bộ nhớ và tính toán trên tensor.

### 3.2. Thống kê Độ dài Token
Một phát hiện thú vị từ phân tích thống kê:
- **Phạm vi:** Tokens có độ dài từ 1 đến 16 ký tự.
- **Trung vị (Median):** Độ dài trung vị quan sát được là 4 ký tự.
- **Phân nhóm:** Dựa trên trung vị, dữ liệu được chia thành 3 nhóm: "Ngắn hơn trung vị", "Bằng trung vị" và "Dài hơn trung vị". Do token là các giá trị nguyên, một lượng lớn dữ liệu (khoảng 1/8) tập trung chính xác tại giá trị trung vị, tạo nên một đặc thù thống kê cần lưu ý khi thực hiện các phép so sánh sau này.

---

## 4. Kiểm chứng Trạng thái Hoạt hóa
Sau khi chạy batch dữ liệu qua mô hình, chúng ta thu được 12 ma trận hoạt hóa, mỗi ma trận có kích thước $[16, 512, 3072]$. 
- `16`: Số chuỗi trong batch.
- `512`: Số tokens trong mỗi chuỗi.
- `3072`: Số nơ-ron MLP mở rộng.
Sự đồng nhất về kích thước trên tất cả các tầng xác nhận hệ thống Hooks đã hoạt động chính xác và sẵng sàng cho việc tính toán thống kê cường độ (magnitude) ở Phần 2.

---

## 5. Kết Luận Phần 1
Chúng ta đã hoàn tất việc xây dựng "phòng thí nghiệm nội soi" cho GPT-Neo. Việc phân nhóm token theo độ dài ký tự cung cấp một biến độc lập rõ ràng để nghiên cứu sự tác động lên biến phụ thuộc là hoạt hóa nơ-ron. Giai đoạn tiếp theo sẽ đi sâu vào việc xây dựng các biểu đồ histogram để so sánh trực tiếp các nhóm này, nhằm tìm kiếm các xu hướng chọn lọc độ dài xuyên suốt các tầng của mô hình.

---

## Tài liệu tham khảo (Citations)
1. Thử thách về Activation histograms trên GPT-Neo dựa trên `aero_LLM_11_CodeChallenge Activation histograms by token length (part 1).md`. Thiết lập hệ thống Hooks đa tầng và phân tích thống kê độ dài token từ FineWeb.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
