# Nhập môn Python: Kỹ thuật Nhập và Phân tích Văn bản trực tuyến (Importing Text from the Web)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu phương pháp thu thập dữ liệu văn bản trực tiếp từ môi trường internet thông qua thư viện `requests` trong Python. chúng ta thực hiện quy trình trích xuất nội dung từ Project Gutenberg (một kho lưu trữ văn bản công cộng quy mô lớn), phân tích các thuộc tính của đối tượng phản hồi (response object) và xử lý các ký tự điều khiển định dạng. Nghiên cứu cũng thực hiện các phép đo lường thống kê cơ bản như tổng dung lượng ký tự và quy mô tập từ vựng duy nhất (unique characters) thông qua cấu trúc dữ liệu `set`. Đây là bước khởi đầu quan trọng trong việc xây dựng các tập dữ liệu huấn luyện (training datasets) cho các mô hình ngôn ngữ từ các nguồn tài nguyên trực tuyến.

---

## 1. Thu thập Dữ liệu với Thư viện Requests

### 1.1. Giao thức HTTP Get
Để nhập văn bản từ một trang web, Python sử dụng thư viện `requests`. Hàm `requests.get(url)` thực hiện một yêu cầu truy cập đến máy chủ và tải toàn bộ thông tin tại địa chỉ URL chỉ định về bộ nhớ tạm dưới dạng một đối tượng (object).

### 1.2. Khám phá Đối tượng (Hàm `dir`)
Khi làm việc với các kiểu dữ liệu mới, hàm `dir(biến)` là công cụ thiết yếu để liệt kê toàn bộ các phương thức (methods) và thuộc tính (attributes) khả dụng. Đối với dữ liệu văn bản, thuộc tính `.text` là quan trọng nhất vì nó chứa nội dung thô của tài liệu.

---

## 2. Tiền xử lý và Ký tự đặc biệt
Văn bản thu thập từ web thường chứa các ký hiệu định dạng mà mắt thường không thấy được trong các trình soạn thảo thông thường:
- **`\n` (Newline):** Ký hiệu xuống dòng.
- **`\r` (Carriage Return):** Ký hiệu đầu dòng.
Việc hiểu và xử lý các ký tự này là cần thiết để đảm bảo tính nhất quán của dữ liệu trước khi đưa vào các thuật toán tokenization.

---

## 3. Thống kê Đặc trưng Văn bản

### 3.1. Phân tích Dung lượng
Sử dụng hàm `len(web_text)` để xác định tổng số lượng ký tự có trong văn bản. Đây là chỉ số quan trọng để ước tính tài nguyên tính toán cần thiết.

### 3.2. Tập hợp Duy nhất (`set`)
Hàm `set()` chuyển đổi một chuỗi văn bản thành một tập hợp toán học chỉ chứa các phần tử không trùng lặp.
- **Thực nghiệm:** Một văn bản có thể có hàng trăm nghìn ký tự (`The Odyssey` có hơn 700.000 ký tự), nhưng số lượng ký tự duy nhất (bao gồm chữ cái, dấu câu và ký hiệu định dạng) thường chỉ chiếm một tỷ lệ cực nhỏ (khoảng 150 ký tự). Điều này minh chứng cho tính nén và sự lặp lại của ngôn ngữ tự nhiên.

---

## 4. Trình bày số liệu với F-strings Nâng cao
Đối với các con số lớn, việc đọc hiểu dữ liệu thô (ví dụ: 710323) thường gây khó khăn. F-strings cung cấp định dạng ngăn cách hàng nghìn bằng dấu phẩy:
- **Cú pháp:** `{biến:,}`.
- **Kết quả:** `710,323`. 
Việc chuẩn hóa hiển thị này giúp các báo cáo kết quả thực nghiệm trở nên chuyên nghiệp và dễ thẩm định hơn.

---

## 5. Kết luận
Khả năng nhập văn bản trực tiếp từ web mở ra nguồn tài nguyên dữ liệu vô tận cho nghiên cứu LLM. Việc nắm vững quy trình từ thu thập, khám phá đối tượng đến phân tích thống kê cơ bản là nền tảng để nhà nghiên cứu xây dựng các pipeline xử lý dữ liệu tự động, biến internet thành một thư viện học tập khổng lồ cho các mô hình trí tuệ nhân tạo.

---

## Tài liệu tham khảo (Citations)
1. Phương pháp nhập văn bản từ web và phân tích bộ ký tự duy nhất dựa trên `aero_LL_02_Importing text from the web.md`. Phân tích thư viện requests, hàm set() và định dạng số lớn trong F-strings.
