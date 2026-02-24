# Mục 18: Khám phá Nơ-ron và Số chiều trong LLM

Mục này chứa các báo cáo nghiên cứu và thử thách lập trình về Diễn giải học cơ học (Mechanistic Interpretability), tập trung vào việc phân tích chức năng của các nơ-ron đơn lẻ và các sublayers trong kiến trúc Transformer.

## Danh sách các Báo cáo

### 1. Cực đại hóa Hoạt hóa (Activation Maximization)
- **[Aero LLM 01]**: Cực đại hóa Hoạt hóa qua Gradient Ascent (Lý thuyết).
- **[Aero LLM 02]**: Triển khai Cực đại hóa Hoạt hóa (Code).
- **[Aero LLM 03]**: Cực đại hóa Hoạt hóa qua Lấy mẫu Dữ liệu.
- **[Aero LLM 04]**: Thử thách: Tính tái lập của Cực đại hóa Hoạt hóa.

### 2. Kỹ thuật Nội soi Mô hình (Model Introspection)
- **[Aero LLM 05]**: Trích xuất hoạt hóa sử dụng Hooks trong PyTorch.
- **[Aero LLM 06]**: Mối quan hệ giữa Hooks và `output.hidden_states`.
- **[Aero LLM 07]**: Làm rõ về Hidden States tầng cuối và vai trò của LayerNorm.

### 3. Phân tích Tính chọn lọc (Selectivity Analysis)
- **[Aero LLM 08]**: Thử thách: Điều chỉnh Ngữ pháp trong nơ-ron MLP (Phần 1).
- **[Aero LLM 09]**: Thử thách: Điều chỉnh Ngữ pháp trong nơ-ron MLP (Phần 2).
- **[Aero LLM 10]**: Thử thách: Hoạt hóa được điều chế bởi ngữ cảnh trong MLP.

### 4. Thống kê Hoạt hóa và Token (Token & Activation Statistics)
- **[Aero LLM 11]**: Biểu đồ hoạt hóa theo độ dài token (Phần 1).
- **[Aero LLM 12]**: Biểu đồ hoạt hóa theo độ dài token (Phần 2).
- **[Aero LLM 13]**: Biểu đồ hoạt hóa theo độ dài token (Phần 3).
- **[Aero LLM 14]**: Xử lý biểu diễn cho các từ đa token (Multi-token words).

### 5. Phân tích Chuyên sâu và Hồi quy Logistic
- **[Aero LLM 15]**: Thử thách: Các phép chiếu MLP được điều chỉnh theo danh mục (Phần 1).
- **[Aero LLM 16]**: Thử thách: Các phép chiếu MLP được điều chỉnh theo danh mục (Phần 2).
- **[Aero LLM 17]**: Phân loại qua Hồi quy Logistic: Lý thuyết và Mã nguồn.
- **[Aero LLM 18]**: So sánh Hồi quy Logistic và Kiểm định T-test.

### 6. Nghiên cứu thực thể và Phủ định (Entity & Negation Studies)
- **[Aero LLM 19]**: Điều chỉnh Danh từ riêng trong GPT-2 Medium.
- **[Aero LLM 20]**: Thử thách: Điều chỉnh Phủ định trong nơ-ron MLP (Phần 1).
- **[Aero LLM 21]**: Thử thách: Điều chỉnh Phủ định trong nơ-ron MLP (Phần 2).
- **[Aero LLM 22]**: Thử thách: Điều chỉnh Phủ định trong nơ-ron MLP (Phần 3).
- **[Aero LLM 23]**: Thử thách: Điều chỉnh Phủ định trong nơ-ron QVK (Attention).

---
*Ghi chú: Các báo cáo này được trình bày theo định dạng khoa học, bao gồm Tóm tắt (Abstract), Phương pháp nghiên cứu (Methodology), Kết quả và Thảo luận.*
