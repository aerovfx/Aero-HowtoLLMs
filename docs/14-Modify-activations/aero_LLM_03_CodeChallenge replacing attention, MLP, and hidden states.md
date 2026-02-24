# Thử thách Lập trình: Thay thế Hoạt hóa Attention, MLP và Hidden States

## Tóm tắt (Abstract)
Thử thách lập trình này mở rộng khả năng thao túng mô hình thông qua việc can thiệp sâu vào các thành phần nội tại: Attention Heads, MLP Neurons và Hidden States. Nghiên cứu thực hiện bốn bài tập thực hành từ việc triệt tiêu (Zero-out) một Attention Head cụ thể, bơm nhiễu Gaussian vào các Neuron MLP có chỉ số chẵn, đến việc thay đổi quy mô (Scaling) toàn bộ Hidden States của một Transformer Block. Kết quả cho thấy sự biến thiên của phổ hoạt hóa tăng dần theo độ sâu của mô hình và sự xuất hiện của các cơ chế bù trừ (Compensation) tại lớp cuối cùng khi đối mặt với các can thiệp cực đoan. Báo cáo cũng làm rõ cấu trúc dữ liệu Tuple của Hidden States trong hệ sinh thái HuggingFace.

---

## 1. Mở Đầu (Introduction)
Việc hiểu rõ cách can thiệp vào từng thành phần của mô hình không chỉ giúp kiểm chứng các giả thuyết nhân quả mà còn hé lộ cách thức thông tin được truyền dẫn và biến đổi qua các tầng. Thử thách này tập trung vào kỹ năng lập trình Hook nâng cao và khả năng phân tích tác động hạ nguồn (Downstream impact). Chúng ta sẽ quan sát cách việc "bóp nghẹt" hoặc "phóng đại" tín hiệu tại một tầng ảnh hưởng đến các tầng kế tiếp trong dòng chảy Transformer.

---

## 2. Tiết Thiết Lập Thử Thách (Methodology & Exercises)

### 2.1. Bài tập 1: Triệt tiêu Attention Head cụ thể (Selective Head Ablation)
Mục tiêu là xác định và đưa về giá trị 0 toàn bộ ma trận $K$ (Key) của Attention Head thứ 3 (Index 2) tại một tầng Transformer.
- **Tính toán Index:** Với $d\_model=768$ và 12 heads, mỗi head có $d\_head=64$. Index bắt đầu của Head 3 trong ma trận $K$ rời rạc là $2 \times 64 = 128$. Tuy nhiên, nếu thao tác trên ma trận $QKV$ gộp, ta phải cộng thêm $n\_embed$ (để bỏ qua toàn bộ $Q$).
- **Thực thi:** Tách ma trận $QKV \to$ Clone ma trận $K \to$ Gán 0 cho vùng Index tương ứng $\to$ Concatenate lại.

### 2.2. Bài tập 2: Bơm Nhiễu Gaussian vào MLP (MLP Noise Injection)
Thay thế các Neuron MLP có chỉ số chẵn (0, 2, 4...) bằng nhiễu Gaussian có trung bình $\mu=10$.
- **Phân tích:** Đây là một can thiệp "phi tự nhiên" vì phân phối hoạt hóa thực tế của MLP (sau khi qua GELU) thường dịch chuyển âm hoặc bị triệt tiêu về 0 để tạo tính thưa (Sparsity). Việc bơm nhiễu với $\mu=10$ tạo ra một sự lệch pha phân phối (Out-of-distribution) cực lớn, giúp quan sát rõ rệt sự hỗn loạn của tín hiệu truyền đi.

### 2.3. Bài tập 3 & 4: Thay đổi quy mô Hidden States (Hidden State Scaling)
Can thiệp vào đầu ra cuối cùng của một Transformer Block (trước khi vào Block kế tiếp).
- **Cấu trúc Tuple:** Khác với Attention hay MLP, biến `output` tại Block Transformer là một **Tuple**. Phần tử đầu tiên `output[0]` là Tensor hoạt hóa chính, các phần tử sau chứa thông tin bổ trợ như hidden_states hoặc attentions (nếu được kích hoạt).
- **Thực thi:** `modified_hidden = output[0].mul(scaling_factor)`. Sau đó phải đóng gói lại vào Tuple trước khi `return`: `(modified_hidden,) + output[1:]`.
- **Kết quả:** Khi giảm quy mô (Scale 0.1) tại Block 9, ta thấy độ lệch chuẩn (variability) sụt giảm đột ngột tại tầng đó nhưng bắt đầu "hồi sinh" ở các tầng sau. Khi phóng đại (Scale 10), tín hiệu bùng nổ nhưng lớp cuối cùng của mô hình có xu hướng nén lại để bù trừ (Compensating) trước khi xuất ra Embeddings Layer.

---

## 3. Khám Phá Cấu Trúc Tuple Của HuggingFace
Tại sao đầu ra Transformer lại là Tuple? 
Thực nghiệm cho thấy khi thiết lập `output_attentions=True`, Tuple này sẽ nở rộng từ 1 phần tử lên 2 phần tử. Phần tử thứ hai chứa ma trận Attention $[Batch, Heads, Seq, Seq]$. Việc dùng Tuple cho phép mô hình linh hoạt trả về nhiều loại dữ liệu khác nhau mà không làm gãy cấu trúc Hook đồng nhất. Tuy nhiên, khuyến nghị nghiên cứu vẫn là sử dụng Hook trực tiếp vào các Sub-layers để kiểm soát dữ liệu tinh vi hơn.

---

## 4. Kết Luận
Việc nắm vững kỹ thuật thao túng Tuple và Indexing chính xác trong ma trận QKV là chìa khóa để thực hiện các thí nghiệm nhân quả phức tạp. Thử thách này khẳng định rằng mô hình học sâu không phải là một khối tĩnh, mà là một thực thể động có khả năng phản ứng và bù trừ trước các tác động từ bên ngoài, đặc biệt là ở những tầng tiệm cận đầu ra.

---

## Tài liệu tham khảo (Citations)
1. Thử nghiệm can thiệp đa thành phần (Multi-component interference) dựa trên `aero_LLM_03_CodeChallenge replacing attention, MLP, and hidden states.md`. Giải thích cơ chế đóng gói Tuple trong cấu trúc `huggingface-transformer`.
