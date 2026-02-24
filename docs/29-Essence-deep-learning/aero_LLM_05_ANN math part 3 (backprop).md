# Học sâu: Giải tích ANN Phần 3 – Lan truyền ngược (Backpropagation)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về cơ chế lan truyền ngược (backpropagation), "động cơ" cốt lõi giúp mạng nơ-ron nhân tạo học tập từ dữ liệu. chúng ta phân tích quá trình mở rộng từ một bộ phân lớp Perceptron đơn lẻ sang một mạng lưới đa tầng phức tạp, nơi mỗi nút hoạt động như một đơn vị tính toán độc lập. Nghiên cứu giải mã mối liên hệ hữu cơ giữa lan truyền ngược và thuật toán Hạ giang (Gradient Descent), đồng thời thuyết minh về vai trò của quy tắc chuỗi (chain rule) trong việc tính toán đạo hàm của các hàm hợp để điều chỉnh trọng số, từ đó tối ưu hóa hàm mất mát trên toàn bộ kiến trúc mạng.

---

## 1. Từ Perceptron đến Mạng lưới Đa tầng

Trong các kiến trúc phức tạp, chúng ta đơn giản hóa biểu đồ tính toán bằng cách coi mỗi Perceptron là một "nút" (node) duy nhất:
- **Tính độc lập:** Mỗi nút không biết mình nằm trong một mạng lưới khổng lồ; nó chỉ tiếp nhận đầu vào, tính toán tổng có trọng số kèm phi tuyến tính và chuyển đầu ra cho các nút tiếp theo.
- **Dòng chảy dữ liệu:** Dữ liệu thô đi vào các tầng đầu tiên, được biến đổi qua nhiều cấp độ phi tuyến trước khi đưa ra dự đoán cuối cùng ở tầng đầu ra.
- **Tín hiệu sai số:** Tín hiệu này được tính ở tầng cuối cùng và phải "chảy ngược" lại toàn bộ mạng lưới để từng nút biết cần điều chỉnh trọng số của mình như thế nào.

---

## 2. Bản chất của Lan truyền ngược

Lan truyền ngược thực chất chính là **Gradient Descent** được áp dụng cho mọi tầng của mạng nơ-ron:
- **Công thức cập nhật:** Trọng số mới ($w$) được tính bằng cách trừ đi đạo hàm của hàm mất mát nhân với tốc độ học ($\eta$):
  $$w = w - \eta \cdot \frac{\partial L}{\partial w}$$
- **Thách thức:** Vì hàm dự đoán $\hat{y}$ là sự kết hợp của nhiều hàm chồng chéo (tích vô hướng nằm trong hàm kích hoạt, nằm trong hàm mất mát), chúng ta không thể tính đạo hàm trực tiếp một cách đơn giản.

---

## 3. Quy tắc Chuỗi (Chain Rule) và Đạo hàm Hàm hợp

Để giải quyết sự phức tạp của các hàm lồng nhau, lan truyền ngược sử dụng quy tắc chuỗi:
- **Cơ chế:** Đạo hàm tổng thể được chia nhỏ thành tích của các đạo hàm thành phần. Ví dụ, sự thay đổi của hàm mất mát theo trọng số phụ thuộc vào:
    1. Hàm mất mát thay đổi thế nào theo đầu ra của tầng cuối.
    2. Đầu ra tầng cuối thay đổi thế nào theo kết quả tính toán tuyến tính.
    3. Kết quả tuyến tính thay đổi thế nào theo từng trọng số cụ thể.
- **Tính thực tiễn:** Mỗi hàm kích hoạt (Sigmoid, ReLU, Tanh) đều có một công thức đạo hàm riêng, đóng vai trò như một mắt xích trong chuỗi tính toán này.

---

## 4. Tối ưu hóa trong Thực tế (PyTorch và Máy tính)

Mặc dù giải tích về lan truyền ngược có vẻ rất "rối rắm", các thư viện AI hiện đại như PyTorch đã tự động hóa quy trình này:
- **Ổn định số học:** Các thuật toán được tinh chỉnh để tránh hiện tượng gradient biến mất hoặc bùng nổ mà các công thức thuần túy có thể gặp phải.
- **Hiệu năng:** Việc tính toán trung bình trên các "lô" dữ liệu (batches) và sử dụng các thủ thuật lập trình giúp quá trình lan truyền ngược diễn ra cực nhanh trên GPU.
- **Nguyên lý không đổi:** Dù các kỹ thuật mã hóa có thay đổi, khái niệm cốt lõi vẫn là tìm hướng dốc nhất để hạ thấp sai số.

---

## 5. Kết luận
Lan truyền ngược là cầu nối giữa lý thuyết toán học trừu tượng và khả năng học tập thực tế của máy tính. Việc thấu hiểu quy tắc chuỗi và cách thức sai số lan tỏa ngược qua các tầng nơ-ron là chìa khóa để giải thích cách một mô hình LLM với hàng tỷ tham số có thể tự tinh chỉnh để hiểu được ngôn ngữ con người. Kết thúc phần giải tích này, chúng ta đã có đầy đủ các mảnh ghép: từ Perceptron (kiến trúc), Lan truyền xuôi (vận hành), Hàm mất mát (thước đo) đến Lan truyền ngược (học tập).

---

## Tài liệu tham khảo (Citations)
1. Cơ chế lan truyền ngược và quy tắc chuỗi đạo hàm dựa trên `aero_LL_05_ANN math part 3 (backprop).md`. Thuyết minh về sự độc lập của các nút nơ-ron và quy trình cập nhật trọng số tự động trong các thư viện học sâu. village.
