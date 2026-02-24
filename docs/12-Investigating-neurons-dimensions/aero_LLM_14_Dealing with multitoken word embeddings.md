# Xử lý Biểu diễn Nơ-ron cho các Từ đa Token (Multi-token Words)

## Tóm tắt (Abstract)
Báo cáo này giải quyết một thách thức thực tiễn trong Diễn giải học (Mechanistic Interpretability): Cách trích xuất hoạt hóa cho các từ bị chia thành nhiều tokens bởi tokenizer (ví dụ: "toothpaste" $\rightarrow$ ["tooth", "paste"]). Qua phân tích lý thuyết và thực nghiệm trên GPT-2, nghiên cứu khẳng định rằng việc tập trung vào **token cuối cùng** là phương pháp tối ưu. Lý do cốt lõi nằm ở bản chất nhân quả (causal) của mô hình: tại token cuối, biểu diễn đã tích hợp toàn bộ thông tin từ các token thành phần phía trước, tạo thành một khái niệm ngữ nghĩa hoàn chỉnh. Báo cáo cũng cung cấp khung mã nguồn Python để xác định vị trí và phân tích sự biến thiên của các multi-token embeddings xuyên suốt các tầng.

---

## 1. Mờ Đầu (Introduction)
Trong tiếng Anh và nhiều ngôn ngữ khác, không phải từ nào cũng tương ứng với một token duy nhất. Các từ ghép ("toothpaste"), từ phức hoặc từ hiếm thường bị bẻ gãy. Khi nghiên cứu tính chọn lọc của nơ-ron đối với một "từ", câu hỏi đặt ra là chúng ta nên lấy dữ liệu từ token nào? Báo cáo này thiết lập một quy trình chuẩn hóa để xử lý các "đơn vị ngữ nghĩa đa thành phần" này.

---

## 2. Giả thuyết "Token cuối cùng là chìa khóa"

### 2.1. Cơ chế Tích hợp Ngữ cảnh
Xét từ "toothpaste":
1. Khi mô hình xử lý token "tooth", nó chưa biết từ tiếp theo là gì (có thể là "ache", "brush", hoặc "paste"). Biểu diễn tại đây chỉ mang tính dự đoán (predictive).
2. Khi mô hình xử lý token "paste" (đặc biệt là khi không có khoảng trắng phía trước và đi sau "tooth"), lớp Attention sẽ điều chế vector này dựa trên thông tin "tooth" đã có trong residual stream.
3. **Kết luận:** Tại vị trí "paste", mô hình mới thực sự sở hữu biểu diễn của khái niệm "toothpaste" hoàn chỉnh. "Tooth" không biết gì về "paste", nhưng "paste" biết rất nhiều về "tooth".

---

## 3. Quy trình Thực nghiệm và Triển khai Mã nguồn

### 3.1. Xác định vị trí Thuật toán (Algorithmic Indexing)
Trong một batch văn bản phức tạp, việc tìm vị trí của một cụm token đích (target sequence) đòi hỏi một quy trình kiểm duyệt nghiêm ngặt:
- Duyệt qua từng câu trong batch.
- Kiểm tra sự trùng khớp của token hiện tại và $k$ tokens phía trước với chuỗi đích.
- Lưu trữ index của token cuối cùng để phục vụ trích xuất `hidden_states`.

### 3.2. Quản lý Batch và Padding
Để xử lý các câu có độ dài khác nhau, nghiên cứu sử dụng kỹ thuật padding và `attention_mask`. Việc unpack dictionary thông qua toán tử `**` trong PyTorch giúp đẩy dữ liệu qua mô hình một cách hiệu quả, đảm bảo các token padding không làm nhiễu kết quả phân tích.

---

## 4. Phân tích Sự biến thiên Vector (Vector Displacement)
Nghiên cứu giới thiệu một phép đo thực nghiệm: Độ dài quỹ đạo của vector nhúng khi đi qua mô hình.
- **Công thức:** $\|v_l - v_{l-1}\|$, trong đó $v_l$ là biểu diễn tại tầng $l$.
- **Quan sát:** Sự thay đổi này phản ánh khối lượng công việc tính toán mà các lớp Attention và MLP đã thực hiện để tinh chỉnh ý nghĩa của token. Đối với các từ đa token, token cuối cùng thường bộc lộ sự biến thiên lớn ở các tầng giữa, nơi "phép cộng ngữ nghĩa" thực sự diễn ra.

---

## 5. Kết Luận
Việc hiểu rõ cách tokenizer phân rã ngôn ngữ là điều kiện tiên quyết cho mọi nghiên cứu nội soi mô hình. Báo cáo xác lập quy tắc: Để phân tích một khái niệm, hãy luôn nhìn vào token kết thúc chuỗi biểu diễn khái niệm đó. Phương pháp này không chỉ đảm bảo tính chính xác về mặt ngữ nghĩa mà còn nhất quán với cơ cấu vận hành của kiến trúc Transformer.

---

## Tài liệu tham khảo (Citations)
1. Kỹ thuật xử lý multi-token word embeddings trên GPT-2 dựa trên `aero_LLM_14_Dealing with multitoken word embeddings.md`. Lý thuyết về tích hợp thông tin tại token cuối và quy trình trích xuất vector.
