
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [16 Interfering with attention](../index.md)

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
# Thử thách Lập trình: Vá lỗi Head và Token trong tác vụ IOI (Head and Token Patching in IOI)

## Tóm tắt (Abstract)
Báo cáo này trình bày kết quả của một thử thách lập trình nâng cao nhằm cô lập các thành phần tính toán chịu trách nhiệm cho tác vụ Nhận dạng Tân ngữ Gián tiếp (IOI). Bằng cách thực hiện vá lỗi (patching) nhắm mục tiêu vào duy nhất một Attention Head và một Token cuối cùng trong chuỗi, nghiên cứu phân tích sự thay đổi của logit difference (IOI score). Kết quả cho thấy các can thiệp ở mức độ này tạo ra hiệu ứng cực kỳ tinh vi, củng cố lý thuyết về cấu trúc cộng dồn của residual stream và vai trò "tinh chỉnh" của tiểu khối Attention. Báo cáo cũng thảo luận về mô tả toán học của Transformer Block để giải thích sự khác biệt giữa các cấp độ can thiệp.

---

## 1. Mở Đầu (Introduction)
Tiếp nối các thực nghiệm vá lỗi Hidden States, thử thách này yêu cầu mức độ chính xác cao hơn: thay vì ghi đè toàn bộ thông tin tại một tầng, chúng ta chỉ thay đổi một "mảnh" thông tin nhỏ nhất có thể – một Head cụ thể tại Token dự đoán. Mục tiêu là xác định xem liệu có những "Name Mover Heads" (các đầu dịch chuyển tên) cụ thể nào đóng vai trò then chốt trong việc giải quyết mâu thuẫn giữa Subject và Indirect Object hay không.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Can thiệp Vi phẫu (Precision Patching)
Sử dụng mô hình GPT-2 Large với vòng lặp kép qua 36 tầng và 20 heads:
- **Đối tượng can thiệp:** Chỉ duy nhất token cuối cùng (token dự đoán) của chuỗi recipient được thay thế bằng hoạt hóa từ chuỗi donor.
- **Vị trí:** Lớp `c_proj` (pre-hook) để đảm bảo can thiệp xảy ra trước khi các heads bị trộn lẫn.
- **Chỉ số đo lường:** IOI Score được tính bằng sai lệch logit giữa đáp án đúng (Sally) và đáp án sai (Sam).

### 2.2. Kỹ thuật Reshaping
Dữ liệu tại `c_proj` có kích thước `[batch, tokens, heads * head_dim]`. Việc `reshape` thành `[batch, tokens, heads, head_dim]` là bắt buộc để có thể chỉ định chính xác head cần vá mà không làm ảnh hưởng đến các heads lân cận.

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Phân tích Bản đồ Tác động (Impact Mapping)
- **Quan sát:** Khác với việc vá Hidden States (tạo ra sự sụt giảm IOI score khổng lồ), việc vá từng head đơn lẻ chỉ tạo ra những thay đổi rất nhỏ, thường không thể phát hiện được bằng mắt thường trên biểu đồ quét (scatter plot).
- **Xu hướng:** Hiệu ứng thường rõ ràng hơn ở các tầng cuối của mô hình, nơi các thông tin ngữ cảnh đã được tinh luyện để chuẩn bị cho việc giải mã token. Một số head làm giảm nhẹ IOI score, trong khi số khác lại làm tăng – cho thấy sự phân hóa chức năng của các Attention Heads.

### 3.2. Giải thích Toán học về sự khác biệt
Đầu ra của một Transformer Block ($x_{out}$) được mô tả bởi phương trình:
$$x_{out} = x_{in} + \Delta Attention(LN(x_{in})) + \Delta MLP(LN(x_{in} + \Delta Attention))$$
Trong đó:
- $\Delta Attention$ là tổng đóng góp của tất cả các heads.
- Can thiệp của chúng ta chỉ nhắm vào $1/N_{heads}$ của thành phần $\Delta Attention$ tại duy nhất một vị trí token.
- Do đó, phần lớn thông tin trong $x_{out}$ vẫn đến từ $x_{in}$ (residual stream) và đầu ra của MLP, giải thích tại sao hiệu ứng lại vô cùng tinh vi.

---

## 4. Thảo Luận: Sự Phát triển của Diễn giải học Cơ học
Kết quả này minh chứng cho một quy luật trong khoa học: khi kiến thức tăng lên, các thực nghiệm sẽ chuyển từ "búa tạ" (sledgehammer) sang "vi phẫu" (微手术). 
- Việc một can thiệp nhỏ đến mức gần như vô hình vẫn tạo ra sự thay đổi về số liệu logit là một minh chứng cho tính nhạy cảm và độ chính xác của mô hình Transformer.
- Để đạt được hiệu ứng lớn như vá Hidden States, chúng ta cần xác định và vá đồng thời một "nhóm" các heads có chức năng tương đồng (circuit analysis).

---

## 5. Kết Luận
Thử thách này khẳng định rằng Attention Heads không hoạt động độc lập để tạo ra ý nghĩa, mà chúng đóng góp những "tinh chỉnh" (tweaks) nhỏ vào một dòng chảy thông tin khổng lồ. Việc tìm kiếm "Name Mover Heads" đòi hỏi sự kết hợp giữa phân tích định lượng chính xác và một khung lý thuyết vững chắc về dòng chảy thông tin trong residual stream.

---

## Tài liệu tham khảo (Citations)
1. Thử thách Precision Head Patching trên tác vụ IOI dựa trên `aero_LLM_07_CodeChallenge Head and token patching in IOI.md`. Phân tích định lượng và mô tả toán học của Transformer Block.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Cắt bỏ Attention Head và Dự đoán Token (Head Ablation and Token Prediction)](aero_LLM_01_Head ablation and token prediction.md) | [Xem bài viết →](aero_LLM_01_Head ablation and token prediction.md) |
| [Thử thách Lập trình: Dự đoán Token sau khi Cắt bỏ Head (Phần 1)](aero_LLM_02_CodeChallenge Token prediction after head ablations (part 1).md) | [Xem bài viết →](aero_LLM_02_CodeChallenge Token prediction after head ablations (part 1).md) |
| [Thử thách Lập trình: Dự đoán Token sau khi Cắt bỏ Head (Phần 2)](aero_LLM_03_CodeChallenge Token prediction after head ablations (part 2).md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Token prediction after head ablations (part 2).md) |
| [Tác động của việc "Tắt tiếng" Head lên Độ tương đồng Cosine (Impact of Head-Silencing on Cosine Similarity)](aero_LLM_04_Impact of head-silencing on cosine similarity.md) | [Xem bài viết →](aero_LLM_04_Impact of head-silencing on cosine similarity.md) |
| [Thử thách Lập trình: GPT-2 có thực sự thích Pizza Dứa? (Một nghiên cứu về can thiệp Attention chính xác)](aero_LLM_05_CodeChallenge Does GPT2 like pineapple pizza.md) | [Xem bài viết →](aero_LLM_05_CodeChallenge Does GPT2 like pineapple pizza.md) |
| [Vá lỗi Attention Head trong tác vụ Nhận dạng Tân ngữ Gián tiếp (Attention Head Patching in IOI)](aero_LLM_06_Attention head patching in IOI.md) | [Xem bài viết →](aero_LLM_06_Attention head patching in IOI.md) |
| 📌 **[Thử thách Lập trình: Vá lỗi Head và Token trong tác vụ IOI (Head and Token Patching in IOI)](aero_LLM_07_CodeChallenge Head and token patching in IOI.md)** | [Xem bài viết →](aero_LLM_07_CodeChallenge Head and token patching in IOI.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
