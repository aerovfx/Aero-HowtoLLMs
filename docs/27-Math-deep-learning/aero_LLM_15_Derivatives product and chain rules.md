# Toán học trong Học sâu: Quy tắc Nhân và Quy tắc Chuỗi (Product & Chain Rules)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về các kỹ thuật nâng cao trong tính toán đạo hàm cho các hàm số phức hợp, những thành phần không thể tách rời của thuật toán lan truyền ngược (backpropagation) trong học sâu. chúng ta phân tích cơ chế vận hành của quy tắc nhân (Product Rule) đối với các hàm số tương tác và quy tắc chuỗi (Chain Rule) đối với các hàm số lồng nhau. Nghiên cứu thực hiện thực nghiệm so sánh giữa phương pháp tính toán thủ công và sử dụng thư viện SymPy, qua đó khẳng định tầm quan trọng của việc tự động hóa tính đạo hàm trong các framework như PyTorch nhằm xử lý các kiến trúc nơ-ron đa tầng với hiệu năng và độ chính xác cao.

---

## 1. Quy tắc Nhân (Product Rule): Đạo hàm của sự Tương tác

Khi hai hàm số $f(x)$ và $g(x)$ nhân với nhau, đạo hàm của chúng không đơn giản là tích của các đạo hàm riêng lẻ:
- **Công thức:** $\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$.
- **Cơ chế:** Đạo hàm được tính bằng tổng của (đạo hàm hàm thứ nhất nhân với hàm thứ hai giữ nguyên) và (hàm thứ nhất giữ nguyên nhân với đạo hàm hàm thứ hai). Đây là nguyên lý cơ bản để tính toán sự thay đổi đồng thời của nhiều thành phần trong một nơ-ron.

---

## 2. Quy tắc Chuỗi (Chain Rule): Đòn bẩy của Backpropagation

Quy tắc chuỗi xử lý các trường hợp hàm lồng hàm $f(g(x))$, đây là cấu trúc phổ biến nhất trong mạng nơ-ron (trong đó đầu ra của lớp này là đầu vào của lớp kế tiếp):
- **Công thức:** $\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$.
- **Trực quan:** Đạo hàm cuối cùng bằng sản phẩm của sự thay đổi hàm bên ngoài nhân với sự thay đổi của hàm bên trong.
- **Ứng dụng:** Quy tắc này cho phép mô hình AI "phân phối" sai số từ lớp đầu ra ngược trở lại từng trọng số ở các lớp ẩn sâu bên trong, giúp mô hình biết chính xác cần điều chỉnh bao nhiêu ở từng vị trí để giảm thiểu lỗi tổng thể.

---

## 3. Thực thi Kỹ thuật và Tự động hóa

Việc tính toán các đạo hàm phức tạp bằng tay dễ dẫn đến sai sót và không khả thi đối với các mô hình lớn.
- **SymPy và Math ký hiệu:** SymPy cho phép kiểm chứng các quy tắc này một cách trực quan thông qua định dạng LaTeX đẹp mắt, giúp nhà nghiên cứu nắm vững bản chất lý thuyết.
- **Vai trò của Framework (PyTorch):** Một thông điệp then chốt là các kỹ sư AI không cần phải tự giải các phương trình đạo hàm phức tạp. PyTorch cung cấp hệ thống Autograd để thực hiện quy tắc chuỗi tự động một cách cực kỳ nhanh chóng và chính xác, cho phép chúng ta tập trung vào việc thiết kế kiến trúc thay vì tính toán đại số.

---

## 4. Tại sao cần thấu hiểu các Quy tắc này?

Mặc dù máy tính làm thay phần tính toán, việc hiểu rõ Quy tắc Chuỗi giúp nhà nghiên cứu:
1. **Chẩn đoán mô hình:** Hiểu tại sao gradient bị triệt tiêu (vanishing) trong các mạng quá sâu.
2. **Tối ưu hóa thiết kế:** Lựa chọn các hàm kích hoạt có đạo hàm "khỏe" để duy trì tín hiệu học tập.
3. **Nắm vững bản chất:** Thấu hiểu cách thức từng tham số nhỏ đóng góp vào thành bại của một dự đoán lớn.

---

## 5. Kết luận
Quy tắc nhân và quy tắc chuỗi là những "người hùng thầm lặng" đứng sau sự phát triển bùng nổ của trí tuệ nhân tạo hiện đại. Chúng là các mắt xích logic cho phép tri thức được truyền dẫn qua các tầng kiến trúc phức tạp. Với sự hỗ trợ của các công cụ lập trình mạnh mẽ, việc nắm vững các khái niệm này không còn là rào cản tính toán mà trở thành lợi thế tư duy, giúp chúng ta xây dựng và tinh chỉnh những mô hình LLM thông minh và bền bỉ hơn.

---

## Tài liệu tham khảo (Citations)
1. Cơ chế tính đạo hàm phức hợp và ứng dụng quy tắc chuỗi trong học sâu dựa trên `aero_LL_15_Derivatives product and chain rules.md`. Phân tích quy tắc nhân, hàm lồng nhau và vai trò của tự động hóa đạo hàm trong PyTorch.
