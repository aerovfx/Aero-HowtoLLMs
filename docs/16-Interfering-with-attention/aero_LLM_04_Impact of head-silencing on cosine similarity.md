# Tác động của việc "Tắt tiếng" Head lên Độ tương đồng Cosine (Impact of Head-Silencing on Cosine Similarity)

## Tóm tắt (Abstract)
Báo cáo này khám phá tác động của việc cắt bỏ Attention Head lên mối quan hệ không gian giữa các token embeddings, sử dụng chỉ số Độ tương đồng Cosine (Cosine Similarity). Thay vì đo lường kết quả dự đoán cuối cùng, nghiên cứu tập trung vào việc quan sát cách các Representation của token "co lại" hoặc "giãn ra" trong residual stream sau khi một Head bị vô hiệu hóa. Sử dụng mô hình GPT-2 Medium và tác vụ với câu mẫu hài hước về Pizza, thực nghiệm chỉ ra rằng các can thiệp nhỏ ở tầng sớm có thể tạo ra "hiệu ứng gợn sóng" (ripple effect) kéo dài đến tận các tầng cuối của mô hình, minh chứng cho tính chất của một hệ thống hỗn loạn tất định.

---

## 1. Mở Đầu (Introduction)
Mục tiêu cốt lõi của tiểu khối Attention là tích hợp thông tin ngữ cảnh giữa các token. Nếu giả thuyết này đúng, việc vô hiệu hóa (silencing) một Attention Head sẽ trực tiếp làm thay đổi độ tương đồng giữa các vector đại diện của các token. Nghiên cứu này sử dụng Độ tương đồng Cosine để định lượng sự thay đổi này, cung cấp một góc nhìn nội tại hơn về cách các mạch thần kinh tương tác với nhau trước khi đưa ra dự đoán.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Cấu trúc Thực nghiệm và Chỉ số Đo lường
- **Kỹ thuật:** Sử dụng Forward Pre-hook để triệt tiêu một Head bất kỳ trong tầng `c_proj`.
- **Chỉ số:** Độ tương đồng Cosine giữa tất cả các cặp token trong câu: $CS(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$.
- **Công thức phần tử duy nhất:** Đối với ma trận đối xứng $M \times M$, số cặp token duy nhất (loại trừ đường chéo) là $\frac{M(M-1)}{2}$.

### 2.2. Phân tích T-test và Hiệu chỉnh Đa so sánh
Để xác định xem sự thay đổi độ tương đồng là có ý nghĩa thống kê hay chỉ là nhiễu, chúng ta thực hiện phép thử T-test trên tập hợp các cặp token.
- **Hiệu chỉnh Bonferroni:** Ngưỡng ý nghĩa được điều chỉnh thành $p < 0.05 / 24$ (số tầng) để tránh sai số loại I khi thực hiện nhiều phép thử đồng thời.

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Quỹ đạo của Độ tương đồng Cosine (Clean Model)
- **Xu hướng:** Độ tương đồng thường giảm dần ở các tầng đầu (decoupling) khi mô hình đang cố gắng phân hóa ý nghĩa các token dựa trên ngữ cảnh riêng biệt.
- **Hội tụ:** Ở các tầng cuối, độ tương đồng tăng mạnh hướng tới giá trị 1, cho thấy các embeddings đang hội tụ về một không gian biểu diễn chung để chuẩn bị dự đoán token tiếp theo.

### 3.2. Hiệu ứng Gợn sóng (Ripple Effect)
Thí nghiệm "Tắt tiếng" Head tại Layer 3 cho thấy:
- **Tác động tức thời:** Thay đổi nhỏ và đôi khi không có ý nghĩa thống kê ngay tại tầng bị can thiệp.
- **Lan truyền:** Tuy nhiên, sai lệch này không biến mất mà lan truyền qua các tầng tiếp theo. Đến các tầng cuối (Layer 20-24), hiệu ứng trở nên nhất quán và có ý nghĩa thống kê rõ rệt.
- **Ý nghĩa:** Điều này chứng minh rằng trong một hệ thống phức tạp, các sai số nhỏ ở đầu chuỗi có thể tích tụ và định hình lại toàn bộ trạng thái cuối của hệ thống.

---

## 4. Thảo Luận: Decoupling vs. Coupling
- **Decoupling:** Việc tắt Head đôi khi làm giảm độ tương đồng, cho thấy Head đó đóng vai trò "kết nối" các ý niệm.
- **Coupling:** Ngược lại, ở một số tầng khác, việc tắt Head lại làm tăng độ tương đồng, gợi ý rằng Head đó vốn dĩ có chức năng "phân biệt" và giữ các token xa nhau trong không gian vector.

---

## 5. Kết Luận
Việc phân tích độ tương đồng Cosine cung cấp cái nhìn chi tiết hơn về động lực học bên trong của Transformer so với việc chỉ nhìn vào xác suất đầu ra. Kết quả củng cố quan niệm về LLM như một hệ thống hỗn loạn tất định, nơi mọi thành phần dù nhỏ nhất đều đóng góp vào cấu trúc vĩ mô của sự hiểu biết ngôn ngữ.

---

## Tài liệu tham khảo (Citations)
1. Thí nghiệm Cosine Similarity trên GPT-2 Medium dựa trên `aero_LLM_04_Impact of head-silencing on cosine similarity.md`. Phân tích hiệu ứng Ripple và quỹ đạo hội tụ embeddings.
