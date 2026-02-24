# Toán học trong Học sâu: Tìm Cực trị bằng Đạo hàm (Minima and Maxima)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về phương pháp xác định các điểm cực trị (local minima và local maxima) của hàm số bằng công cụ đạo hàm, một thành phần cốt lõi của thuật toán Hạ giang (Gradient Descent) trong học sâu. chúng ta phân tích khái niệm các "điểm tới hạn" (critical points) nơi đạo hàm bằng 0, đồng thời thiết lập các tiêu chí toán học để phân biệt giữa cực tiểu và cực đại dựa trên dấu của đạo hàm ở các vùng lân cận. Nghiên cứu cũng thảo luận về hiện tượng "biến mất gradient" (vanishing gradient) tại các vùng hàm số không đổi, một thách thức lớn trong việc huấn luyện các mạng nơ-ron đa tầng.

---

## 1. Điểm tới hạn: Nơi Đạo hàm bằng 0

Trong giải tích, các điểm mà tại đó hàm số ngừng tăng hoặc ngừng giảm và bắt đầu đổi hướng được gọi là điểm tới hạn:
- **Nguyên lý:** Tại các đỉnh (cực đại) hoặc đáy (cực tiểu) của một đường cong, tiếp tuyến của đồ thị nằm ngang, nghĩa là độ dốc hay đạo hàm tại đó bằng chính xác 0.
- **Quy trình tìm kiếm:** Để tìm các điểm này, chúng ta tính đạo hàm của hàm mất mát, cho đạo hàm bằng 0 và giải phương trình tìm biến số $x$. Kết quả trả về là tập hợp tất cả các vị trí có tiềm năng là cực trị.

---

## 2. Phân biệt Cực tiểu (Minima) và Cực đại (Maxima)

Mặc dù cả cực tiểu và cực đại đều có đạo hàm bằng 0, chúng có đặc điểm thay đổi độ dốc khác nhau ở hai phía:
- **Cực tiểu (Minima):** Là mục tiêu của học sâu (cực tiểu hóa sai số).
    - Bên trái điểm cực tiểu: Hàm số đang giảm (đạo hàm âm).
    - Bên phải điểm cực tiểu: Hàm số đang tăng (đạo hàm dương).
- **Cực đại (Maxima):**
    - Bên trái điểm cực đại: Hàm số đang tăng (đạo hàm dương).
    - Bên phải điểm cực đại: Hàm số đang giảm (đạo hàm âm).
Việc thấu hiểu sự khác biệt này giúp thuật toán Gradient Descent biết cách điều chỉnh trọng số để luôn hướng về phía "thung lũng" của hàm mất mát thay vì leo lên các "đỉnh núi".

---

## 3. Thách thức từ Vùng phẳng và Vanishing Gradient

Ngoài cực tiểu và cực đại, còn có trường hợp thứ ba nơi đạo hàm bằng 0: **Vùng phẳng (Plateaus)**.
- **Đặc điểm:** Hàm số không đổi hoặc thay đổi cực kỳ chậm trong một khoảng rộng. Tại đây, đạo hàm biến mất (về 0) nhưng chúng ta chưa đạt được điểm tối ưu.
- **Hệ quả trong Deep Learning:** Khi gradient biến mất, mô hình ngừng học vì đạo hàm không còn cung cấp thông tin về hướng cần di chuyển. Đây là vấn đề phổ biến khi sử dụng các hàm kích hoạt như Sigmoid trong các mạng quá sâu.

---

## 4. Ứng dụng trong Thuật toán Hạ giang (Gradient Descent)

Thuật toán Gradient Descent tận dụng thông tin từ đạo hàm để thực hiện các bước di chuyển:
1. Nếu đạo hàm âm: Nghĩa là chúng ta đang ở sườn dốc bên trái cực tiểu, cần tăng $x$ để tiến về đáy.
2. Nếu đạo hàm dương: Nghĩa là chúng ta đang ở sườn dốc bên phải cực tiểu, cần giảm $x$ để lùi về đáy.
Sự tương tác liên tục giữa giá trị đạo hàm và vị trí giúp mô hình dần hội tụ về điểm có sai số thấp nhất có thể.

---

## 5. Kết luận
Tìm kiếm cực đại và cực tiểu không chỉ là bài toán tìm ẩn số, mà là hành trình tìm kiếm sự tối ưu cho trí tuệ nhân tạo. Khả năng phân tích các điểm tới hạn bằng đạo hàm giúp chúng ta định vị được các cấu hình trọng số tốt nhất cho mô hình. Việc nhận diện được các bẫy vùng phẳng và hiểu rõ cơ chế chuyển đổi dấu của đạo hàm là nền tảng để nắm vững các kỹ thuật tối ưu hóa tiên tiến, đảm bảo mô hình LLM có thể học tập hiệu quả từ những dữ liệu phức tạp nhất.

---

## Tài liệu tham khảo (Citations)
1. Phương pháp xác định cực trị và phân tích điểm tới hạn dựa trên `aero_LL_14_Derivatives find minima.md`. Phân tích dấu đạo hàm lân cận, phân biệt cực tiểu/cực đại và thảo luận về hiện tượng vanishing gradient.
