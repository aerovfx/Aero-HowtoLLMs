
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [12 Investigating neurons dimensions](../index.md)

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
# Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 2)

## Tóm tắt (Abstract)
Báo cáo này tiếp tục thử thách nghiên cứu về mối quan hệ giữa độ dài token và hoạt hóa nơ-ron MLP, tập trung vào việc so sánh các phương pháp phân tích thống kê: Phân nhóm dựa trên trung vị (Median Split) và Phân tích tương quan tuyến tính (Pearson Correlation). Nghiên cứu thực nghiệm trên GPT-Neo 125M và 1.3B bộc lộ những động lực học phức tạp xuyên suốt các tầng. Kết quả cho thấy xu hướng "thiên kiến token ngắn" ở các tầng giữa, sự khác biệt định tính ở tầng đầu tiên, và sự thu hẹp phân phối ở các tầng cuối khi mô hình chuyển dịch tiêu điểm từ token hiện tại sang dự báo token tiếp theo. Báo cáo cũng thảo luận về giới hạn của tính phổ quát (universality) giữa các quy mô mô hình khác nhau.

---

## 1. Mở Đầu (Introduction)
Trong trí tuệ nhân tạo, việc biến đổi các biến liên tục thành các phạm trù (discretization) – như việc chia token thành "ngắn" và "dài" – là một công cụ trực quan hóa hữu ích nhưng tiềm ẩn rủi ro mất mát thông tin. Báo cáo này đối chiếu phương pháp phân nhóm truyền thống với các phép đo tương quan bảo toàn độ phong phú của dữ liệu, nhằm mục đích giải mã cách LLM điều chế năng lượng nơ-ron dựa trên cấu trúc hình thái của token.

---

## 2. Phân tích Phân nhóm (Median Split Analysis)

### 2.1. Phân phối Hoạt hóa tại Tầng 5
Thực nghiệm chia 8192 tokens thành ba nhóm dựa trên độ dài trung vị (4 ký tự). Biểu đồ histogram cho thấy các dải hoạt hóa của nhóm "Ngắn", "Trung bình" và "Dài" có sự trùng lắp đáng kể. Tuy nhiên, quan sát kỹ cho thấy nhóm token dài có xu hướng dịch chuyển nhẹ về phía các giá trị hoạt hóa âm hơn.
- **Nhận định:** Phương pháp phân nhóm chỉ ra xu hướng mờ nhạt, xác nhận rằng độ dài token không phải là biến số duy nhất quyết định cường độ phản ứng của MLP.

---

## 3. Phân tích Tương quan Tuyến tính (Correlation Analysis)

### 3.1. Phép đo Pearson trên Tập Standardized
Để khai thác toàn bộ 16 mức độ dài token, nghiên cứu áp dụng hệ số tương quan $r$ giữa giá trị hoạt hóa đã được chuẩn hóa ($z$-score) và độ dài token.
- **Công thức:** $r = \frac{\text{cov}(x,y)}{\sigma_x \sigma_y}$. Khi dữ liệu đã được chuẩn hóa ($\mu=0, \sigma=1$), hệ số tương quan chính bằng tích vô hướng của hai vector chia cho $n-1$.
- **Kết quả:** Phần lớn nơ-ron bộc lộ tương quan âm yếu. Điều này củng cố giả thuyết rằng nơ-ron MLP kích hoạt mạnh hơn đối với các token ngắn – vốn là những token xuất hiện với tần suất cao (high frequency) trong tập huấn luyện.

---

## 4. Động lực học Xuyên tầng và Quy mô Mô hình

### 4.1. Đặc thù Tầng đầu và Tầng cuối
1. **Tầng 1 (First Block):** Bộc lộ hành vi trái ngược hoàn toàn với phần còn lại của mô hình – tương quan dương mạnh mẽ (token dài kích hoạt mạnh hơn). Đây là tầng gần với dữ liệu thô nhất, nơi mô hình đang thực hiện các phân tích hình thái sơ cấp.
2. **Các tầng cuối:** Phân phối tương quan co hẹp lại và tiến dần về mức 0. Giải thích cơ học: Tại điểm gần đầu ra, mô hình không còn xử lý thuộc tính của token hiện tại mà đã chuyển sang trạng thái trừu tượng để dự báo token tương lai.

### 4.2. So sánh GPT-Neo 125M vs 1.3B
Khi nâng quy mô mô hình lên 10 lần, chúng ta quan sát thấy sự đứt gãy của tính phổ quát:
- **Tính đa hình (Multimodality):** Ở mô hình 1.3B, phân phối hoạt hóa không còn là Gaussian đơn thuần mà bộc lộ các cụm (clusters) rõ rệt, gợi ý rằng các nơ-ron ở quy mô lớn đã phân hóa thành các nhóm chức năng chuyên biệt hóa sâu sắc hơn.

---

## 5. Thảo Luận: Biến nhiễu Tần suất (The Frequency Confound)
Báo cáo lưu ý rằng "độ dài token" có mối tương quan nghịch đảo chặt chẽ với "tần suất xuất hiện". Các token ngắn thường là các từ chức năng hoặc gốc từ phổ biến. Do đó, sự kích hoạt mạnh hơn của MLP đối với token ngắn có thể thực chất là phản ứng với sự quen thuộc (familiarity) thay vì độ dài vật lý.

---

## 6. Kết Luận
Nghiên cứu khẳng định rằng việc quan sát hành vi nơ-ron MLP cần phải được thực hiện trên toàn bộ lộ trình của residual stream. Sự khác biệt định tính giữa các tầng và sự thay đổi đặc tính khi tăng quy mô mô hình nhắc nhở chúng ta rằng các quy luật tìm thấy ở mô hình nhỏ (toy models) có thể không luôn đúng đối với các hệ thống AI cấp độ sản xuất (production-grade).

---

## Tài liệu tham khảo (Citations)
1. Phân tích tương quan nơ-ron và tác động tầng trên GPT-Neo dựa trên `aero_LLM_12_CodeChallenge Activation histograms by token length (part 2).md`. So sánh quy mô 125M và 1.3B qua các biểu đồ histograms và Heatmaps.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 12-Investigating-neurons-dimensions](README.md) | [Xem bài viết →](README.md) |
| [Cực đại hóa Hoạt hóa (Activation Maximization): Cơ sở Lý thuyết và Những thách thức trong LLM](aero_LLM_01_Activation maximization via gradient ascent (theory).md) | [Xem bài viết →](aero_LLM_01_Activation maximization via gradient ascent (theory).md) |
| [Triển khai Cực đại hóa Hoạt hóa: Từ Gradient Ascent đến Giải mã Token (Activation Maximization Implementation)](aero_LLM_02_Activation maximization (code).md) | [Xem bài viết →](aero_LLM_02_Activation maximization (code).md) |
| [Cực đại hóa Hoạt hóa qua Lấy mẫu Dữ liệu (Activation Maximization via Data Sampling)](aero_LLM_03_Activation maximization via data sampling.md) | [Xem bài viết →](aero_LLM_03_Activation maximization via data sampling.md) |
| [Thử thách Lập trình: Kiểm chứng Tính lặp lại của Cực đại hóa Hoạt hóa (Reproducibility of Activation Maximization)](aero_LLM_04_CodeChallenge Reproducibility of activation maximization.md) | [Xem bài viết →](aero_LLM_04_CodeChallenge Reproducibility of activation maximization.md) |
| [Giải phẫu Nội tại Mô hình bằng Hooks: Kỹ thuật Trích xuất Hoạt hóa (Extracting Activations via Hooks)](aero_LLM_05_Extracting activations using hooks.md) | [Xem bài viết →](aero_LLM_05_Extracting activations using hooks.md) |
| [Mối tương quan giữa Hooks và Hidden States: Giải cấu trúc Khối Transformer (Reconstructing Transformer Blocks)](aero_LLM_06_Relation between hooks and output.hidden_states.md) | [Xem bài viết →](aero_LLM_06_Relation between hooks and output.hidden_states.md) |
| [Làm rõ về Hidden States Tầng cuối: Vai trò của LayerNorm (Clarification of Final Hidden States)](aero_LLM_07_Clarification of final hidden_states output.md) | [Xem bài viết →](aero_LLM_07_Clarification of final hidden_states output.md) |
| [Thử thách Lập trình: Tính Chọn lọc Ngữ pháp của Nơ-ron MLP (Phần 1)](aero_LLM_08_CodeChallenge Grammar tuning in MLP neurons (part 1).md) | [Xem bài viết →](aero_LLM_08_CodeChallenge Grammar tuning in MLP neurons (part 1).md) |
| [Thử thách Lập trình: Tính Chọn lọc Ngữ pháp của Nơ-ron MLP (Phần 2)](aero_LLM_09_CodeChallenge Grammar tuning in MLP neurons (part 2).md) | [Xem bài viết →](aero_LLM_09_CodeChallenge Grammar tuning in MLP neurons (part 2).md) |
| [Thử thách Lập trình: Sự Điều chế Ngữ cảnh trong Hoạt hóa MLP (Context-modulated Activation)](aero_LLM_10_CodeChallenge Context-modulated activation in MLP.md) | [Xem bài viết →](aero_LLM_10_CodeChallenge Context-modulated activation in MLP.md) |
| [Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 1)](aero_LLM_11_CodeChallenge Activation histograms by token length (part 1).md) | [Xem bài viết →](aero_LLM_11_CodeChallenge Activation histograms by token length (part 1).md) |
| 📌 **[Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 2)](aero_LLM_12_CodeChallenge Activation histograms by token length (part 2).md)** | [Xem bài viết →](aero_LLM_12_CodeChallenge Activation histograms by token length (part 2).md) |
| [Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 3)](aero_LLM_13_CodeChallenge Activation histograms by token length (part 3).md) | [Xem bài viết →](aero_LLM_13_CodeChallenge Activation histograms by token length (part 3).md) |
| [Xử lý Biểu diễn Nơ-ron cho các Từ đa Token (Multi-token Words)](aero_LLM_14_Dealing with multitoken word embeddings.md) | [Xem bài viết →](aero_LLM_14_Dealing with multitoken word embeddings.md) |
| [Thử thách Lập trình: Hình chiếu MLP Điều chỉnh theo Danh mục (Phần 1)](aero_LLM_15_CodeChallenge Category-tuned MLP projections (part 1).md) | [Xem bài viết →](aero_LLM_15_CodeChallenge Category-tuned MLP projections (part 1).md) |
| [Thử thách Lập trình: Hình chiếu MLP Điều chỉnh theo Danh mục (Phần 2)](aero_LLM_16_CodeChallenge Category-tuned MLP projections (part 2).md) | [Xem bài viết →](aero_LLM_16_CodeChallenge Category-tuned MLP projections (part 2).md) |
| [Hồi quy Logistic: Lý thuyết và Triển khai Phân loại Nơ-ron](aero_LLM_17_Classification via logistic regression theory and code.md) | [Xem bài viết →](aero_LLM_17_Classification via logistic regression theory and code.md) |
| [Đối chiếu Hồi quy Logistic và Kiểm định T-test: Giả định và Ứng dụng](aero_LLM_18_Logistic regression vs. t-test assumptions and applications.md) | [Xem bài viết →](aero_LLM_18_Logistic regression vs. t-test assumptions and applications.md) |
| [Điều chỉnh Danh từ riêng trong GPT-2 Medium](aero_LLM_19_Proper noun tuning in GPT2-medium.md) | [Xem bài viết →](aero_LLM_19_Proper noun tuning in GPT2-medium.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 1)](aero_LLM_20_CodeChallenge Negation tuning in MLP neurons (part 1).md) | [Xem bài viết →](aero_LLM_20_CodeChallenge Negation tuning in MLP neurons (part 1).md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 2)](aero_LLM_21_CodeChallenge Negation tuning in MLP neurons (part 2).md) | [Xem bài viết →](aero_LLM_21_CodeChallenge Negation tuning in MLP neurons (part 2).md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 3)](aero_LLM_22_CodeChallenge Negation tuning in MLP neurons (part 3).md) | [Xem bài viết →](aero_LLM_22_CodeChallenge Negation tuning in MLP neurons (part 3).md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron QVK (Attention)](aero_LLM_23_CodeChallenge Negation tuning in QVK neurons.md) | [Xem bài viết →](aero_LLM_23_CodeChallenge Negation tuning in QVK neurons.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
