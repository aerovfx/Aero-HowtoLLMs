
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [13 Investigating layers](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Thử Thách Lập Trình (Code Challenge): Ứng Dụng Logit Lens Trong Mạng BERT (Phần 2)

## Tóm tắt (Abstract)
Tiếp nối hành trình giải mã quá trình tiến hóa quyết định nội bộ (Internal Logits Evolutions), báo cáo này mô tả phương pháp thiến lập đúng đắn thuật toán Logit Lens đối với nền tảng kiến trúc BERT. Bắt nguồn từ tính phức hợp trong Mô-đun Giải Mã (Decoder Module/Predictions) của BERT – đòi hỏi nhiều kích hoạt Gelu và LayerNorm chứ không đơn thuần là ma trận trọng số như GPT, nghiên cứu tái thiết kế luồng quy chiếu mảng (Tensor Projections) đúng cách. Bằng việc phân tách Z-score khuyết từ trên 24 hệ biến thiên, ta quan sát được "khoảnh khắc Eureka", nơi mạng học sâu bẻ gãy quỹ đạo rối rắm sang định hình chắc chắn về từ ngữ, chứng minh cách mạng Neural lắp ráp ngữ cảnh thành khối.

---

## 1. Mở Đầu (Introduction)
Khi thực nghiệm Logit Lens trên kiến trúc GPT, phép toán thực sự giản đơn: Áp dụng nhân Tích vô hướng (Dot product) giữa Trạng thái ẩn lớp trung gian $L_i$ và mảng Trọng số tuyến tính giải mã $LM\_Head^T$. Thế nhưng cơ chế BERT (Bidirectional Encoder Representations from Transformers) cất giấu điều bất ngờ: Khối `model.predictions` chịu trách nhiệm quy đổi (Unembeddings) được nhồi thêm hàm kích hoạt phi tuyến (Gelu), tinh chỉnh LayerNorm và hàng loạt biến đổi phức tạp.
Nếu tiếp tục chép nguyên công thức Logit Lens cổ điển, hệ thống đổ nát và đáp án giải mã chỉ là nhiễu vô nghĩa. 

Nghiên cứu này thiết lập **Bản nguyên tắc chuẩn** để soi Logit Lens trên BERT, đồng thời quét toàn bộ mô hình (Sweep layers) để phơi bày tấm bản đồ Heatmap quá trình đưa ra quyết định Masked Token.

---

## 2. Phương Pháp Chỉnh Vị (Methodology Corrections)

### 2.1. Phép Sai Lầm Phân Chiếu Vô Hướng (The Incorrect Dot-Product Approach)
Sử dụng tầng ẩn áp chót (Layer 22/24), nếu ta trích bóc thô lỗ đoạn mã trọng số `model.predictions.decoder.weight` ra để nhân ma trận, token được dự giải mã trả về kết quả ảo giác (Hallucination) dưới dạng "hash", "wives" và Z-score chìm thảm bại xuống đáy rãnh ($Z \approx 3$). Nguyên nhân đến từ việc dòng vector kia đã lọt sổ khâu bù trừ phi tuyến Gelu và LayerNorm, làm méo mó các vector khoảng cách (Distance alignment).

### 2.2. Phép Đúng Đắn: Đẩy Qua Tổng Mô-Đun Đầu Bảng (Full-Module Forwarding)
Công thức Logits Lens chân thực của mô hình BERT bắt buộc phải đẩy vector ẩn $L_i$ đi qua toàn bộ khối module kiến trúc cuối cùng thay vì tự thực hiện phép nhân nháp:
$$ \text{Logits}_{L_i} = \text{model.predictions}(\text{Hidden\_States}_{L_i}) $$
Khi thao tác đúng, kết quả lập tức đồng bộ. Ở Layer 22, phương pháp đẩy Module cho ra Z-score khổng lồ lên tới $\approx 25$ đến $30\ \sigma$, và từ đoán giải mã xuất hiện chính xác đáng kinh ngạc. 

---

## 3. Phân Phối Trực Quan Qua Tầng (Layer-wise Analysis & Heatmaps)

### 3.1. Phân Lũy Tiến Biểu Đồ Tuyến (Z-Score Trajectories)
Đánh giá độ chắc chắn Z-Score (khoảng cách điểm kỳ vọng của Predicted Token với bầy đàn Vocab phân bổ xung quanh) tại tất cả 24 Blocks Transformer, ta thấy:
- **Tầng 0 - 10:** Z-score dao động ở mức nhiễu ngẫu nhiên. Mạng liên tục đoán những mảnh từ ghép điên rồ như "accreditation", "fellowship" dù đầu vào không hề mang tính học thuật.
- **Tầng 11 - 15:** Z-score nhúc nhích tịnh tiến lên.
- **Tầng 15 - 24:** Một khuỷu bộc phát (Knee curve) bóp ngoặt góc độ và chĩa thẳng lên giời thẳng tắp. Tín hiệu này phác thảo rõ "Khoảnh khắc Eureka", nơi mạng Neural tìm đủ manh mối, thu thập đủ lực đong bù context 2 bên trái phải, đồng hóa để chốt hạ một quyết định sắc nhọn (Token: "way").

### 3.2. Hiệu Ứng Thuật Cảnh (Visualizing The Snap)
Trải phẳng Heatmap với trục Y (Layers) và trục X (Tất cả Masked tokens lặp vòng), được Normalization Min-Max Scaling (từ $0 \to 1$). 
- Bản đồ màu minh hoạ sự vô thức (đen mù tịt) kéo dài qua nửa đầu lộ trình. 
- Đến 1/3 chặng cuối cùng, ánh sáng Softmax loé lên với màu Vàng rực. Một số Token ngữ pháp dễ nối (như "do", "you") được mô hình bóc tách và ngộ ra quyết định sớm hơn (Tầm layer 10-12) so với các Token mang hàm nghĩa rộng hơn. 

---

## 4. Kết Luận
Logit Lens không phải thuật toán công thức vạn năng. Nó đòi hỏi nhà nghiên cứu phải đọc hiểu cấu trúc "Unembeddings" của từng nền tảng LLM riêng biệt. Bằng việc định tuyến đúng qua khối lưới `predictions`, BERT phơi bày vẻ đẹp tư duy kết nối đặc trưng: Sự im lìm ở tầng dưới và cú giật bừng sáng mạnh mẽ ở tầng trên. Phương pháp vẽ Heatmap Z-Score cung ứng cỗ máy quét sinh học MRI sắc nét đo đạc hành trình nhận thức (Cognitive-like mapping) của AI qua chiều độ Trí tuệ.

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm đo lường hiệu lệnh phân đoạn qua `aero_LLM_17_CodeChallenge Logit Lens in BERT (part 2).md` (Kết nối thực nghiệm Logit Lens Full-Module qua 24 layers mạng BERT Large, so đối chứng với kiến trúc thô Dot-Product).
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
