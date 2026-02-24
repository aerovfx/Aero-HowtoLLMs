# Cắt bỏ Attention Head và Dự đoán Token (Head Ablation and Token Prediction)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu tác động của việc can thiệp nhân quả lên các Attention Head riêng lẻ đối với khả năng tích hợp ngữ cảnh của mô hình GPT-2 Small. Bằng cách sử dụng cơ chế `Forward Pre-hook` để can thiệp vào tầng `c_proj` trước khi các đầu (heads) bị trộn lẫn tuyến tính, nghiên cứu thực hiện phương pháp cắt bỏ (ablation) bằng cách gán giá trị không (zeroing out) cho từng đầu. Thí nghiệm sử dụng tác vụ dự đoán thủ đô ("Berlin is the capital of...") để đo lường sự thay đổi của chỉ số Log Softmax. Kết quả cho thấy việc cắt bỏ một đầu gây ra sự suy giảm nhẹ nhưng nhất quán đối với xác suất của token đúng ("Germany"), trong khi vô tình làm tăng xác suất cho các token liên quan về mặt ngữ nghĩa nhưng sai về mặt thực tế ("France").

---

## 1. Mở Đầu (Introduction)
Các cơ chế Attention chịu trách nhiệm kéo các vector embeddings theo những hướng giúp dự báo chính xác token kế tiếp dựa trên ngữ cảnh và tri thức thế giới. Một trong những câu hỏi cơ bản của Diễn giải cơ học (Mechanistic Interpretability) là: làm thế nào để cô lập và thao tác một Attention Head duy nhất trong khi chúng thường bị trộn lẫn ngay lập tức sau khi tính toán? Báo cáo này trình bày kỹ thuật can thiệp vào tầng đầu ra của attention trước khi thông tin bị xáo trộn bởi ma trận trộn tuyến tính $W_O$.

---

## 2. Thiết Lập Thí Nghiệm (Methodology)

### 2.1. Cơ chế Forward Pre-hook và Tầng c_proj
Trong kiến trúc GPT-2 của OpenAI/HuggingFace, 12 attention heads sau khi được tính toán xong sẽ được nối tiếp (concatenated) và đưa vào tầng `c_proj`. Tầng này thực hiện phép nhân với ma trận trộn $W_O$ để tích hợp thông tin từ 12 heads vào residual stream.
- **Kỹ thuật:** Sử dụng `register_forward_pre_hook` vào tầng `c_proj`. 
- **Lý do:** Ở giai đoạn "Pre-input" này, dữ liệu vẫn ở dạng 12 khối 64 chiều nối tiếp nhau. Chúng ta có thể reshape tensor để tách riêng chiều `heads` và dễ dàng thực hiện can thiệp lên một đầu cụ thể (`head_to_ablate`).

### 2.2. Nhiệm vụ Kiểm thử Ngữ cảnh
Câu mẫu: "Berlin is the capital of..."
- **Token đúng (Target):** " Germany"
- **Token đối chứng (Contrast):** " France"
- **Mục tiêu:** Đo lường sự thay đổi xác suất (delta log softmax) khi một head bị "tắt tiếng".

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Sự Suy Giảm Tính Chính Xác (Probability Suppression)
Thực hiện cắt bỏ lần lượt 12 heads tại Transformer Block thứ 5:
- **Xác suất đúng:** Log Softmax cho "Germany" giảm xuống ở hầu hết các head. Điều này chứng tỏ mỗi head đều đóng góp một phần nhỏ nhưng quan trọng vào việc duy trì tri thức thực thể.
- **Sự trỗi dậy của nhiễu ngữ nghĩa:** Thú vị là xác suất cho "France" lại tăng lên. Điều này gợi ý rằng việc phá hủy một head logic làm suy yếu khả năng lọc context, khiến mô hình dễ bị nhầm lẫn giữa các khái niệm cùng nhóm (quốc gia châu Âu) nhưng không chính xác hoàn toàn.

### 3.2. Tính Bền Vững Của Mô Hình (Model Robustness)
Mặc dù Logits thay đổi, mô hình vẫn dự đoán đúng từ "Germany" là lựa chọn hàng đầu (top-1) trong cả 12 lần thử nghiệm. Điều này cho thấy kiến trúc LLM có tính dư thừa (redundancy) cao; việc mất một thành phần đơn lẻ hiếm khi làm sụp đổ hoàn toàn hành vi của mô hình trong các tác vụ đơn giản.

### 3.3. Tầm Quan Trọng Của Đo Lường Liên Tục
- **AI Safety:** Trong an toàn AI, chúng ta quan tâm nhất đến token thực tế mà mô hình sinh ra (categorical output). Tuy nhiên, chỉ nhìn vào kết quả đúng/sai là quá thô.
- **Insight:** Các phép đo số thực (continuous measurements) như logit difference tiết lộ những chuyển dịch nhỏ bên dưới bề mặt. Những chuyển dịch này có thể tích tụ và gây ra lỗi nghiêm trọng trong các hệ thống quy mô lớn hoặc các ngữ cảnh phức tạp hơn.

---

## 4. Kết Luận
Cắt bỏ Attention Head thông qua Pre-hook là một kỹ thuật phẫu thuật chính xác hơn so với việc tác động vào toàn bộ Hidden State. Thí nghiệm xác nhận rằng các Attention Heads hoạt động như các mạch logic phân tán. Câu hỏi quan trọng tiếp theo là: chúng ta nên thay thế giá trị bị cắt bỏ bằng số không, giá trị trung bình, hay một hằng số khác? Đây sẽ là trọng tâm của các thử thách lập trình kế tiếp.

---

## Tài liệu tham khảo (Citations)
1. Thí nghiệm Head Ablation trên GPT-2 Small dựa trên `aero_LLM_01_Head ablation and token prediction.md`. Phân tích sự cân bằng giữa tri thức thực tế và nhiễu ngữ nghĩa.
