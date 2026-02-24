# Cực đại hóa Hoạt hóa qua Lấy mẫu Dữ liệu (Activation Maximization via Data Sampling)

## Tóm tắt (Abstract)
Báo cáo này giới thiệu một phương pháp thay thế đơn giản và hiệu quả hơn để diễn giải các nơ-ron: Cực đại hóa Hoạt hóa thông qua Lấy mẫu Dữ liệu (Data Sampling). Thay vì sử dụng tối ưu hóa gradient trên nhiễu ngẫu nhiên, phương pháp này truyền trực tiếp hàng chục ngàn token từ văn bản thực tế qua mô hình và thống kê các token kích hoạt mạnh nhất một chiều (dimension) cụ thể. Thực nghiệm trên GPT-Neo 125M với văn bản "Through the Looking Glass" cho thấy khả năng xác định các nơ-ron có tính chọn lọc cao đối với các khái niệm ngôn ngữ như "tiền đề thời gian" (temporal precedence). Tuy nhiên, báo cáo cũng nhấn mạnh các thách thức về khả năng mở rộng (scalability) và tính phân tán của các biểu diễn trong các mô hình lớn hơn.

---

## 1. Mở Đầu (Introduction)
Trong các bài báo trước, chúng ta đã thấy những rào cản của việc tối ưu hóa gradient đối với dữ liệu rời rạc như ngôn ngữ. Phương pháp lấy mẫu dữ liệu giải quyết vấn đề này bằng cách sử dụng chính "ngôn ngữ tự nhiên" làm môi trường thử nghiệm. Bằng cách quan sát cách một nơ-ron phản ứng với hàng ngàn mẫu dữ liệu thực, chúng ta có thể xây dựng một bức tranh trực quan và dễ hiểu hơn về "sở thích" của nó.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Quy trình Lấy mẫu
- **Mô hình:** GPT-Neo (125 triệu tham số).
- **Dữ liệu:** Toàn bộ văn bản cuốn sách "Through the Looking Glass" được chia thành các batch (32 sequences x 256 tokens).
- **Kỹ thuật:** Truyền dữ liệu qua mô hình, trích xuất `hidden_states` tại một tầng ($L$) và chiều ($D$) cụ thể.
- **Thống kê:** Sử dụng `numpy.argmax` trên ma trận hoạt hóa (`32 x 256`) để tìm token gây ra phản hồi mạnh nhất trong mỗi batch. Lặp lại quy trình này 1000 lần trên các đoạn văn bản khác nhau.

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Sự hội tụ về Ngữ nghĩa (Semantic Convergence)
Kết quả phân tích tại Tầng 2, Chiều 345:
- **Token đứng đầu:** Từ " before" (có khoảng trắng phía trước) xuất hiện trong gần 50% số lần lấy mẫu.
- **Các token liên quan:** " first", " faster", " quicker", " head".
- **Nhận xét:** Nơ-ron này thể hiện sự điều chỉnh (tuning) rõ rệt đối với khái niệm "ưu tiên thời gian" hoặc "trình tự". Việc kết quả hội tụ về một nhóm từ có liên quan chặt chẽ chứng minh tính hiệu quả của phương pháp lấy mẫu.

### 3.2. Tính Bất định (The Randomness Factor)
Khi thử nghiệm trên các chiều khác (ví dụ: Chiều 5, Tầng 6), kết quả có thể phân tán hơn (ví dụ: "gun", "family", "states"). Điều này cho thấy không phải mọi chiều trong residual stream đều mã hóa một khái niệm đơn ngữ (monosemantic) có thể hiểu được bằng ngôn ngữ tự nhiên.

---

## 4. Ưu điểm và Hạn chế (Pros & Cons)

### 4.1. Ưu điểm
- **Tính Diễn giải cao:** Sử dụng từ ngữ thực tế giúp kết quả gần gũi với logic của con người.
- **Triển khai đơn giản:** Không yêu cầu tính toán gradient phức tạp hay hàm Loss.
- **Tính Linh hoạt:** Dễ dàng áp dụng cho bất kỳ thành phần nào (MLP, Attention heads, Hidden states).

### 4.2. Hạn chế
- **Khả năng Mở rộng:** Với hàng chục ngàn nơ-ron trong các mô hình lớn, việc kiểm tra thủ công từng đơn vị là bất khả thi.
- **Bỏ qua Ngữ cảnh:** Phương pháp này chỉ tập trung vào các token riêng lẻ, trong khi nhiều nơ-ron có thể mã hoá các cấu trúc ngữ pháp dài hoặc ý nghĩa phụ thuộc vào ngữ cảnh.
- **Mã hóa Phân tán:** Một khái niệm có thể được đại diện bởi sự phối hợp của nhiều nơ-ron thay vì chỉ một.

---

## 5. Kết Luận
Cực đại hóa Hoạt hóa qua lấy mẫu dữ liệu là một "bộ lọc" hữu ích để nhanh chóng phát hiện các nơ-ron có chức năng rõ ràng. Dù gặp khó khăn trong việc mở rộng quy mô, đây vẫn là một công cụ pháp chứng quan trọng trong bộ kỹ năng của nhà nghiên cứu Diễn giải học, giúp thu hẹp khoảng cách giữa các con số trừu tượng và ý nghĩa ngôn ngữ học.

---

## Tài liệu tham khảo (Citations)
1. Thực nghiệm lấy mẫu dữ liệu trên GPT-Neo dựa trên `aero_LLM_03_Activation maximization via data sampling.md`. Phân tích Tuning của nơ-ron đối với các khái niệm thời gian.
