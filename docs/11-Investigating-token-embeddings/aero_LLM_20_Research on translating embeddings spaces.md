
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [11 Investigating token embeddings](../index.md)

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
# Sự Dịch Chuyển Và Đồng Tồn Biểu Diễn Giữa Các Không Gian Nhúng

## Tóm tắt

Bài báo khoa học này nêu bật một trong những luồng suy nghĩ tham vọng nhất của Giới trí tuệ nhân tạo học thuật: Liệu sự khác biệt của hàng loạt các bộ não LLMs (như Word2Vec, GloVe, BERT hay GPT) chỉ là kết quả của sự xô lệch trục tọa độ? Liệu có tồn tại một Không gian biểu diễn phổ quát (Universal Platonic Space) và các ma trận phân lớp từ nhúng của mỗi mạng lưới nơ-ron thực chất hoàn toàn có thể được "biên dịch chéo" lẫn nhau? 

---

## 1. Giả Thuyết Không Gian Ngôn Ngữ Phổ Quát (Platonic Embedding Space)

Hiện tại, việc khai thác cấu trúc ma trận nhúng của hai mô hình $M_1$ (ví dụ: Word2Vec) và $M_2$ (ví dụ: GPT-2) luôn cho thấy các phương sai chiều không hề tuyến tính đè lên nhau. Không có hai ma trận embeddings nào hoàn toàn khít lại do sự chênh lệch hàm mục tiêu tối ưu lúc đào tạo (Objective function optimization).

Dù vậy, một luồng triết học và kiến trúc học thuyết (Alignment Hypothesis) đưa ra ý tưởng rằng có một chiều không gian siêu việt và vô hướng (Platonic space) $\mathbb{U}$ quy tụ toàn bộ đặc tính và khối tương quan ngôn ngữ loài người. Các ma trận $E_{\text{w2v}}$ và $E_{\text{gpt}}$ hiện chỉ coi là các chùm tia sáng (Projections layer) mang bản chụp tĩnh của khối lượng tư duy ấy.

### 1.1 Tìm Phép Biến Đổi Vô Hướng Biên Dịch Chéo (Cross-lingual / Cross-model Mapping)
Nếu hệ học của hai mô hình là chung quy luật, thì về mặt lý thuyết thuần túy Toán Hình Học, có thể ánh xạ (map) từ vựng không gian này sang không gian kia (Translation Mapping) bằng bộ khung quy tắc bao gồm ma trận xoay (Rotation $W$) và co dãn chiều (Scaling matrix $S$):
$$
E_2 \approx E_1 \cdot W + b 
$$
Việc dịch chuyển này thường được nỗ lực đạt thông qua Căn chỉnh Procrustes Trực giao (Orthogonal Procrustes problem), một bài toán tìm ma trận trực giao tối ưu để chồng khít hai khối vector mà không sử dụng sự uốn nắn phi tuyến. Trọng điểm chi phí mất mát:
$$ 
\text{Loss} = \| E_1 W - E_2 \|_F^2 \quad \text{với điều kiện } W^\top W = I
$$

---

## 2. Thách Thức Sự Chuyển Hóa Của Đồ Thị Ngôn Ngữ

Việc thiết lập những hàm biên dịch đồng quy mô cho mô hình Embeddings gặp phải rào cản chí mạng là "Sự Di Động" (Dynamism) của mô hình hóa. 

### Rào cản Kiến trúc Attention so với Từ vựng tĩnh
- **Mô Hình Tĩnh (Word2Vec / GloVe):** Sở hữu kết cấu lưới một-đối-một cứng rắn, "Trái táo" mãi mãi là 1 điểm ảnh Euclidean không đổi ở tọa độ tuyệt đối.
- **Mô Hình Động Theo Ngữ Cảnh (Transformer / GPT / BERT):** "Trái táo" khi kết hợp cùng chuỗi hội thoại về "Apple M2" và "Apple Pie" sẽ bị bẻ cong thành các ma trận nhúng biến dị dựa trên ma trận tỷ trọng lưới lưu ý (Attention weights remapping). 

Do đó, vector nhúng trong Transformer không bao giờ là bất di bất dịch, chúng sẽ trượt đi, uốn lượn tại dòng Residual Stream để lấp đầy sự nhiễu loạn ngẫu nhiên của các nút Sampling có nhiệt độ (Softmax Sampling with Temperature T).

---

## 3. Khởi Điểm Hệ Nghiên Cứu Mới

Sự nỗ lực của toán học để biến biên dịch Vector Matrix Translation tuy chứa đựng sự bấp bênh đối với độ sâu phức tạp, nhưng đóng vai trò cực kỳ quan trọng đối với khả năng diễn giải cơ chế (Mech Interp). Sự đào sâu về tính bất toàn của các phép trực giao Procrustes giúp củng cố bản chất thực sự của phương trình Transformer: Sự khôn ngoan của máy móc không tới từ tọa độ lưu từ điển, mà từ vòng lặp cộng nhồi vector của các Layer phi tuyến với sự nhiễu tín học (Randomness Token distribution).

---

## Tài liệu tham khảo

1. **Smith, S., et al. (2017).** *Offline bilingual word vectors, orthogonal transformations and the inverted softmax.* ICLR. (Chỉ ra sự ánh xạ 2 không gian embeddings dịch thuật Procrustes).
2. **Conneau, A., et al. (2018).** *Word Translation Without Parallel Data*. ICLR.
3. Tài liệu định hướng bài giảng *Investigating token embeddings - Translating Embeddings Spaces*.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
