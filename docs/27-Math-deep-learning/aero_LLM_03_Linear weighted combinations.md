# Toán học trong Học sâu: Tổ hợp Tuyến tính có Trọng số (Linear Weighted Combinations)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về tổ hợp tuyến tính có trọng số, phép toán điện toán nền tảng cấu thành nên hoạt động của mọi nơ-ron nhân tạo. Chúng ta phân tích cơ chế xử lý thông tin đầu vào (activations) thông qua các các hệ số điều chỉnh (weights), vai trò của số hạng chệch (bias) trong việc dịch chuyển phân phối đầu ra, và sự khác biệt giữa tổ hợp trọng số với giá trị trung bình cộng. Nghiên cứu thực hiện thực nghiệm mô phỏng trên 10.000 mẫu để thẩm định tính chính xác của phương thức tích hợp bias, qua đó khẳng định rằng việc cộng bias sau bước tổng kết là phương pháp duy nhất đảm bảo sự kiểm soát hệ thống đối với trạng thái kích hoạt của nơ-ron.

---

## 1. Cơ chế Hoạt động của Nơ-ron Nhân tạo

Trong mạng nơ-ron, mỗi nút (node) được coi là một đơn vị xử lý thực hiện phép cộng có trọng số:
- **Đầu vào (Inputs/Activations):** Đại diện cho dữ liệu thô hoặc tín hiệu từ các lớp trước đó.
- **Trọng số (Weights):** Đại diện cho mức độ quan trọng hoặc cường độ kết nối giữa các nơ-ron. Một trọng số bằng 0 sẽ triệt tiêu hoàn toàn tầm ảnh hưởng của đầu vào tương ứng, trong khi trọng số có giá trị tuyệt đối lớn sẽ khuếch đại tín hiệu đó.
- **Tổ hợp Tuyến tính:** Kết quả của phép toán là tổng các tích giữa từng đầu vào và trọng số tương ứng. Nếu mọi trọng số đều bằng $1/n$ (với $n$ là số đầu vào), phép toán này trở thành tính trung bình cộng đơn thuần.

---

## 2. Vai trò của Số hạng Chệch (Bias)

Số hạng chệch ($b$) là một đầu vào đặc biệt không đến từ dữ liệu thực tế mà được sinh ra và học tập nội bộ bên trong mô hình.
- **Mục tiêu:** Cho phép nơ-ron dịch chuyển giá trị kích hoạt sang trái hoặc phải trên trục số, giúp mô hình linh hoạt hơn trong việc ra quyết định (ví dụ: xác định ngưỡng kích hoạt tối thiểu).
- **Thực nghiệm về tính dịch chuyển:** Nghiên cứu chỉ ra rằng việc thay đổi giá trị trung bình của trọng số không tạo ra sự dịch chuyển hệ thống đồng nhất trong kết quả đầu ra. Ngược lại, việc cộng trực tiếp một hằng số $b$ vào tổng cuối cùng là cách thực thi chính xác và ổn định nhất.

---

## 3. Thực thi Kỹ thuật và Phân tích Lỗi

### 3.1. Quy trình Tính toán
Phép toán được thực hiện qua hai giai đoạn:
1. **Phép nhân từng phần tử (Element-wise multiplication):** Nhân cặp tương ứng giữa vectơ trọng số và vectơ kích hoạt.
2. **Phép tổng (Summation):** Cộng dồn tất cả các tích thu được cộng với số hạng chệch.

### 3.2. Phân tích Sai sót trong Hiện thực hóa
Thực nghiệm so sánh hai phương thức tích hợp bias:
- **Phương thức sai:** Cộng bias vào trọng số trước khi nhân. Kết quả cho thấy phân phối đầu ra vẫn tập trung quanh điểm 0, không tạo ra sự dịch chuyển mong muốn.
- **Phương thức đúng:** Thực hiện tổ hợp tuyến tính trước, sau đó mới cộng bias. Kết quả histogram cho thấy toàn bộ phân phối dữ liệu dịch chuyển chính xác theo giá trị của $b$.

---

## 4. Tầm quan trọng trong Mô hình Ngôn ngữ Lớn (LLM)
Mọi tầng Transformer đều dựa trên hàng tỷ phép toán tổ hợp tuyến tính này. Việc hiểu rõ cách trọng số và bias tương tác giúp nhà nghiên cứu giải thích được tại sao mô hình lại ưu tiên các token nhất định trong một ngữ cảnh và cách mà các tham số được tinh chỉnh để đạt được độ chính xác cao trong bài toán dự đoán từ kế tiếp.

---

## 5. Kết luận
Tổ hợp tuyến tính có trọng số dù đơn giản về mặt số học nhưng lại là "nguyên tử" của trí tuệ nhân tạo. Sự kết hợp tinh tế giữa việc gán trọng số cho thông tin và điều chỉnh độ chệch thông qua bias cho phép các mạng nơ-ron học được những quy luật phức tạp từ dữ liệu. Việc làm chủ phép toán này là điều kiện tiên quyết để hiểu sâu hơn về tích vô hướng (dot product) và nhân ma trận – những chủ đề nòng cốt sẽ được trình bày trong các phần tiếp theo.

---

## Tài liệu tham khảo (Citations)
1. Cơ chế tổ hợp tuyến tính có trọng số và ứng dụng số hạng chệch dựa trên `aero_LL_03_Linear weighted combinations.md`. Phân tích cấu trúc nơ-ron, vai trò của bias và thực nghiệm về sự dịch chuyển phân phối đầu ra.
