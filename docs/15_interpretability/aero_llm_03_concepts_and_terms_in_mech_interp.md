
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [15 interpretability](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Các Khái Niệm, Thuật Ngữ và Phương Pháp Trong Diễn Giải Cơ Chế (Mech Interp)

## Tóm tắt
Để bước vào lĩnh vực Khả năng diễn giải cơ chế (Mechanistic Interpretability - Mech Interp), việc làm quen với hệ thống thuật ngữ chuyên môn là bắt buộc. Bài viết này hệ thống hóa các khái niệm nền tảng trong Mech Interp, bao gồm phân tích tương quan (Observation-based) và can thiệp nhân quả (Intervention-based), so sánh cách tiếp cận Từ dưới lên (Bottom-up) và Từ trên xuống (Top-down), đồng thời giải thích nguyên lý thiết kế mô hình có thể diễn giải (Training interpretable models) và giả định về tính phổ quát (Universality) của đồ thị tính toán học sâu.

---

## 1. Phương Pháp Quan Sát (Non-causal) vs. Can Thiệp Nhân Quả (Causal Interventions)

Trong nghiên cứu mạch Transformer, phương pháp thực nghiệm chia thành hai trường phái tuyến tính và phi tuyến chính:

### 1.1 Quan sát phi nhân quả (Observation-based / Correlational)
Nghiên cứu quan sát là việc định tuyến (push) tokens qua mô hình và đọc các ma trận trọng số (weights) hoặc ma trận trạng thái (activations) sinh ra trong quá trình truyền tiến (forward pass) mà *không thay đổi* bất cứ tính toán nào.
- Kỹ thuật thường được dùng là **sử dụng Hook**. Cụm từ "Hook" là một thuật ngữ lập trình (như trong thư viện PyTorch) nhằm đính kèm một hàm (callback) vào một tensor ở một lớp (layer) nhất định để trích xuất hoặc ghi lại giá trị activation nội bộ (ví dụ vector $h_l$) đang được tính toán.
- Phương pháp này xây dựng bằng chứng tương quan (Correlational evidence). Nó giúp phát hiện các mẫu (patterns), chẳng hạn như liên kết một bộ từ vựng nhất định với điểm kích hoạt cao ở một neuron cụ thể.

### 1.2 Can thiệp Nhân quả (Intervention-based / Causal)
Nghiên cứu can thiệp tiến hành sửa đổi, thao túng cấu trúc giá trị trực tiếp. Thay vì chỉ đọc dữ liệu từ Hook, nhà nghiên cứu sẽ sử dụng Hook để ghi đè (overwrite), chặn (ablate), hoặc kết hợp giá trị tính toán ngay trên luồng chạy.
Ví dụ: Thay thế activation $h_l$ bằng một giá trị cố định hoặc một vector nhiễu $\epsilon$:

h'_l = h_l + \epsilon

Việc theo dõi xem kết quả thay đổi này ảnh hưởng đến vector đầu ra (output behavior) thế nào tạo ra **bằng chứng nhân quả (Causal evidence)** nhằm khẳng định sự tham gia của node đó vào vi mạch tổng thể.

Dù can thiệp mang lại bằng chứng đáng tin cậy hơn, hệ thống LLM sở hữu số lượng biến số khổng lồ, khiến không gian tìm kiếm trở nên bất khả thi nếu không có các mô hình quan sát hướng dẫn trước. 

---

## 2. Diễn Giải Mô Hình Tính Toán (Trained vs. Interpretable Models)

Có sự khác biệt triết học giữa hai hướng đi trong An toàn AI:
- **Interpreting Trained Models:** Tập trung vào các LLM khổng lồ, tối ưu hóa mạnh mẽ được tạo ra bởi AI thương mại (ví dụ GPT-4). Các mô hình này vốn được phát triển không nhằm giúp định danh diễn giải, nên chúng mang bản chất là "hộp đen".
- **Training Interpretable Models:** Thiết kế các cấu trúc mô hình có thuật toán minh bạch ngay từ vòng huấn luyện đầu tiên. Sự phát triển bị giới hạn bởi khả năng của con người trong việc tính toán và giám sát rủi ro (Risk-assessments).

Trên thực tế, áp lực thương mại khiến các "hộp đen" trở nên phổ biến, do đó Mech Interp chủ yếu tập trung vào loại hình thứ nhất.

---

## 3. Khung Tiếp Cận Không Gian: Từ Dưới Lên vs. Từ Trên Xuống

### 3.1 Theo Cấu Trúc Đáy (Bottom-Up)
Phương pháp này bắt đầu bằng đơn vị thông tin nhỏ nhất (như một ma trận trọng số neuron $W$ riêng lẻ hoặc từng vector activation), sau đó ghép nối chúng để phác họa hành vi tổng thể (emergent properties). Mặc dù rất chính xác, phương thức Bottom-Up thường gặp khó khăn vì sự phức tạp do sự phụ thuộc vào ngữ cảnh (context-dependency) và khó mở rộng cho các hệ thống hàng tỷ tham số.

### 3.2 Theo Khái Niệm Đỉnh (Top-Down)
Được gọi là hướng tiếp cận tâm lý học (psychological approach). Phương pháp này quan sát một biểu hiện hành vi bên ngoài của mô hình (VD: Thiên kiến giới tính), và sau đó truy xuất ngược (trace back) nhằm tìm các không gian mẫu kích hoạt (activation patterns). Dẫu vậy, phương pháp Top-Down có khả năng báo động giả (False alarms / Type 1 errors) và dính phải các lối phân tích tư duy võ đoán (post-hoc non-mechanistic interpretations).

*Thực tiễn yêu cầu một sự giao thoa (hybrid) chặt chẽ giữa hai lăng kính này trong bất kỳ quy trình nghiên cứu Mech Interp nào.*

---

## 4. Giả Định Phổ Quát (Universality) 

Nguyên lý phổ quát (Universality) trong Mech Interp là giả định cho rằng các cấu trúc kiến tạo vi mạch nơ-ron—dù là sinh hình ảnh hay ngôn ngữ, thông số bé hay siêu to, kiến trúc khác nhau (như BERT, GPT-2 hay Claude)—đều chia sẻ **những motif tính toán chung**.

Nếu tính phổ quát có thật, các phát hiện trên một mạng nơ-ron đồ chơi (Toy Models) chỉ có 1 lớp (layer) Attention hoàn toàn có thể được ngoại suy và bảo toàn thuộc tính toán học ở các mô hình Deep Learning bậc cao. Nó là xương sống định hướng cho hầu hết các nỗ lực tối ưu mô hình, cho phép thu gọn độ khó của toán học ngược (Reverse engineering). Tuy nhiên, đây vẫn đang là một giả thuyết (assumption) hy vọng chứ chưa thu được cơ sở chứng minh toán học chắc chắn trong không gian học sâu hiện đại.

---

## 5. Kết Luận

Một vốn từ vựng hệ thống và các khái niệm thiết kế nền tảng trong Mech Interp là vũ khí để các nhà nghiên cứu xuyên thủng "hộp đen". Nắm bắt được phương thức thiết kế Hook phi nhân quả lẫn nhân quả, làm rõ sự tương quan đối lập của Bottom-up và Top-down, cùng niềm tin vào sự Phổ quát của các hàm toán học sẽ chuẩn bị vững chắc cho bất kỳ ai muốn lấn sâu vào kiến trúc siêu mạng nơ-ron của hiện tại và tương lai.

---

## Tài liệu tham khảo

1. **Elhage, N., et al. (2021).** *A Mathematical Framework for Transformer Circuits.* Anthropic.
2. **Olah, C., et al. (2020).** *Zoom In: An Introduction to Circuits.* Distill.
3. **Casper, S., et al. (2023).** *Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback.*
4. **Alain, G., & Bengio, Y. (2016).** *Understanding intermediate layers using linear classifier probes.* ICLR Workshop.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Khả Năng Diễn Giải Cơ Chế (Mechanistic Interpretability) Là Gì?](aero_llm_01_what_is_mech_interp_mechanistic_interpretability_.md) | [Xem bài viết →](aero_llm_01_what_is_mech_interp_mechanistic_interpretability_.md) |
| [Mối Liên Hệ Giữa Diễn Giải Cơ Chế (Mechanistic Interpretability) và An Toàn AI](aero_llm_02_how_does_mech_interp_relate_to_ai_safety.md) | [Xem bài viết →](aero_llm_02_how_does_mech_interp_relate_to_ai_safety.md) |
| 📌 **[Các Khái Niệm, Thuật Ngữ và Phương Pháp Trong Diễn Giải Cơ Chế (Mech Interp)](aero_llm_03_concepts_and_terms_in_mech_interp.md)** | [Xem bài viết →](aero_llm_03_concepts_and_terms_in_mech_interp.md) |
| [Lý Thuyết và Thực Nghiệm (Theoretical & Empirical Approaches) Trong Nghiên Cứu và Giảng Dạy Khả Năng Diễn Giải Cơ Chế](aero_llm_04_theoretical_and_empirical_approaches_in_research_and_teaching.md) | [Xem bài viết →](aero_llm_04_theoretical_and_empirical_approaches_in_research_and_teaching.md) |
| [Những Lời Chỉ Trích Tổng Quát Về Diễn Giải Cơ Chế (Mechanistic Interpretability)](aero_llm_05_general_criticisms_of_mechanistic_interpretability.md) | [Xem bài viết →](aero_llm_05_general_criticisms_of_mechanistic_interpretability.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
