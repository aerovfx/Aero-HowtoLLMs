# Triển khai Cực đại hóa Hoạt hóa: Từ Gradient Ascent đến Giải mã Token (Activation Maximization Implementation)

## Tóm tắt (Abstract)
Báo cáo này hướng dẫn chi tiết quy trình thực nghiệm để triển khai kỹ thuật Cực đại hóa Hoạt hóa trên mô hình GPT-2 Small bằng PyTorch. Thí nghiệm tập trung vào việc tối ưu hóa một ma trận nhúng ngẫu nhiên (random embeddings) để kích thích tối đa một chiều hoạt hóa cụ thể trong residual stream. Mặc dù quá trình tối ưu hóa toán học đạt được thành công rực rỡ (tăng cường độ hoạt hóa lên 3 bậc độ lớn), kết quả giải mã (decoding) sang văn bản cho thấy các chuỗi token thu được thiếu tính liên kết ngữ nghĩa đối với con người. Kết quả này củng cố giả thuyết về "tính phân tán" và "không gian biểu diễn phi ngôn ngữ" của các nơ-ron bên trong LLM.

---

## 1. Mở Đầu (Introduction)
Trong thực hành, Cực đại hóa Hoạt hóa biến quá trình suy diễn của mô hình thành một bài toán tối ưu hóa ngược. Thay vì truyền văn bản qua tokenizer, chúng ta tác động trực tiếp vào không gian nhúng (embedding space). Mục tiêu là tìm ra "chuỗi token lý tưởng" – dù có thể không tồn tại trong từ điển thực tế – mà mô hình coi là tín hiệu mạnh nhất cho một thành phần nội tại.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Khởi tạo Ma trận Nhúng giả lập
Chúng ta tạo một ma trận nhúng ngẫu nhiên cho một chuỗi gồm 5 tokens. Để đảm bảo tính tương đồng về mặt toán học với mô hình gốc, ma trận này được chuẩn hóa để có cùng độ lệch chuẩn (Standard Deviation) với ma trận nhúng đã huấn luyện của GPT-2.
- **Tham số tối ưu:** `requires_grad = True` được thiết lập cho ma trận nhúng để cho phép PyTorch tính toán gradient.

### 2.2. Cơ chế Pushing Embeddings trực tiếp
Một kỹ thuật quan trọng được sử dụng là tham số `inputs_embeds` trong hàm forward của Hugging Face. Điều này cho phép bỏ qua lớp Tokenizer và Position Embeddings, đẩy trực tiếp giá trị vector vào các Transformer Blocks.

### 2.3. Thiết lập Hàm Loss và Gradient Ascent
- **Mục tiêu:** Cực đại hóa hoạt hóa $a$ tại tầng 4, chiều 90.
- **Hàm tổn thất (Loss):** $L = -a + \lambda \|\theta\|_2^2$. Việc lấy dấu trừ biến bài toán thành cực tiểu hóa, phù hợp với hầu hết các bộ tối ưu (optimizers). Thành phần L2 được thêm vào để ngăn chặn hiện tượng bùng nổ trọng số.
- **Bộ tối ưu:** Adam Optimizer với tốc độ học (learning rate) 0.001 qua 500 bước lặp.

---

## 3. Kết Quả Thực Nghiệm (Results & Analysis)

### 3.1. Hiệu quả của Tối ưu hóa
Đồ thị giám sát cho thấy cường độ hoạt hóa của chiều mục tiêu tăng vọt từ mức gần 0 lên các giá trị dương rất lớn. Đồng thời, các chiều lân cận (neighboring dimensions) bị ức chế, chứng tỏ quá trình tối ưu hóa đã cô lập thành công tính chất đặc trưng của nơ-ron đích.

### 3.2. Nghịch lý Giải mã (The Decoding Paradox)
Bước cuối cùng là chuyển vector nhúng đã tối ưu về token thực thông qua độ tương quan Cosine (Cosine Similarity) với toàn bộ 50.257 tokens trong vocab. 
- **Kết quả văn bản:** "ad pc brisk brisk breast" hoặc các chuỗi vô nghĩa tương tự.
- **Phân tích:** Độ tương quan Cosine cao nhất thường chỉ dừng lại ở mức 0.17. Điều này chỉ ra rằng "vector lý tưởng" mà nơ-ron tìm kiếm nằm ở một vùng không gian không có word nhúng nào thực sự đại diện cho nó.

---

## 4. Thảo Luận: Tại sao phương pháp này "thất bại" trong việc diễn giải?
Dù toán học vận hành chính xác, Activation Maximization trong LLM thường không mang lại tri thức con người có thể hiểu ngay lập tức (human-interpretable). Điều này phản ánh:
- **Sự khác biệt với Vision Models:** Trong hình ảnh, các pixel có tính liên tục. Trong ngôn ngữ, các điểm nhúng nằm rải rác và không có "vùng chuyển tiếp" giữa các khái niệm.
- **Tính đa ngữ (Polysemanticity):** Nơ-ron mục tiêu có thể đang phản ứng với một mô thức cấu trúc phức tạp (như "từ 2 âm tiết bắt đầu bằng phụ âm") hơn là một khái niệm ngữ nghĩa đơn giản.

---

## 5. Kết Luận
Việc thực hiện Activation Maximization không chỉ là bài tập lập trình về Hooks và Gradients, mà còn là một quy trình pháp chứng (forensic process) để hiểu về giới hạn của mô hình. Thất bại trong việc tạo ra văn bản có nghĩa của phương pháp này chính là bằng chứng quan trọng nhất về tính phức tạp của không gian biểu diễn trong LLM, đặt nền móng cho việc sử dụng các kỹ thuật cao cấp hơn như Sparse Autoencoders.

---

## Tài liệu tham khảo (Citations)
1. Quy trình triển khai Code cho Activation Maximization dựa trên `aero_LLM_02_Activation maximization (code).md`. Phân tích việc sử dụng `inputs_embeds` và nghịch lý trong Decoding.
