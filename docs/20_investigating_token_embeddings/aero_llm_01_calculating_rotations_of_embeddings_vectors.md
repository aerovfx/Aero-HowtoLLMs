
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [20 investigating token embeddings](index.md)

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
# Phân Khảo Token Embeddings: Đo Lường Góc Quay Của Vector Biểu Diễn

## Tóm tắt (Abstract)
Báo cáo này đánh dấu sự dịch chuyển trọng tâm nghiên cứu từ cơ chế phân bổ tín hiệu toàn tầng (layer-wise) sang việc theo dõi hành trình tiến hoá của từng cá thể vector nhúng (embeddings vectors). Bằng hệ thống toán học biến đổi từ *Độ tương đồng Cosine* (Cosine Similarity) sang *Góc quay* (Angle of rotation) đo bằng hàm lượng giác chuẩn nghịch đảo $\arccos$, nghiên cứu định lượng mức độ điều chỉnh ngữ cảnh (context modulation) mà các khối Transformer áp đặt lên các Token qua mỗi chặng. Thực nghiệm trên GPT-2 XL vạch ra một đồ thị biên độ góc sắc nét: Khởi động với các vòng xoay khổng lồ (lên tới $90^\circ$ tại Transformer đầu tiên), rồi nguội dần ổn định với những bước xoay nhỏ ở các tầng trung gian, và bùng nổ trở lại ở khâu giải mã sát vách lối ra.

---

## 1. Mở Đầu (Introduction)
Trong một không gian ẩn đa chiều (High-dimensional space), mỗi từ vựng khởi thủy (ví dụ: "cat" hay "her") được cấp phát một vector tĩnh từ Ma trận Token Embeddings. Theo lý thuyết nền tảng của Transformer, lúc chưa được nạp bối cảnh, các vector này hoàn toàn cô lập. 
Tuy nhiên, khi mũi tên vector rẽ nước đi sâu vào các lớp mô hình (Transformer Blocks), cơ chế bộ chú ý (Attention) và mạng lan truyền tiến (MLP) "lôi kéo" vector này lệch khỏi trục hoành thẳng đứng ban đầu. Quá trình bẻ góc thay đổi phương hướng được điều phối hoàn toàn dựa trên sự hiện diện của các token đứng cạnh nhau (Surrounding Context). Bài báo cáo này tập trung lượng hóa những "cú quay xe" này từ khối kiến trúc hiện tại so với khối kiến trúc ở tầng ngay trước nó.

---

## 2. Nền Tảng Toán Học & Tiền Xử Lý (Methodology)

### 2.1. Phương Trình Chuyển Trục (Rotation Formula)
Hệ số độ đo khoảng cách quen thuộc là Độ tương đồng Cosine (Cosine Similarity). Tuy nhiên, để tính được cường độ chệch phương bằng hình học rõ ràng, khái niệm *Góc (Angle)* được giới thiệu.
Sử dụng công thức tích vô hướng và độ dài Norm vector:

$$

$$

\cos(\theta) = \frac{\langle x, y \rangle}{\|x\| \cdot \|y\|}

$$

$$

Ta cô lập hệ số góc $\theta$ bằng hàm lượng giác ngược (Arc Cosine):

$$

$$

\theta = \arccos\left(\frac{\langle x, y \rangle}{\|x\| \cdot \|y\|}\right)

$$

$$

Kết quả ra đơn vị Radians, được nhân với tỷ số $180 / \pi$ để trả về đơn vị độ (Degrees). Điểm ưu việt của góc quay là loại bỏ nhiễu đến từ độ giãn nở (Vector Lengths) của các Activation Norms.

### 2.2. Dữ Liệu Input (Targeted Setup)
Trọng số học tập: `gpt2-xl` (Mô hình tỷ đô với 48 Transformer blocks).
Nguồn mồi (Prompt corpus): Sử dụng bộ sưu tập 54 câu (cung cấp bởi Claude AI), tất cả đều chứa chung một từ khóa đại từ nhân xưng là `"her"`.
Bởi vì chiều dài các câu văn (lengths) lệch nhau, Padding ID (chính là EOS token) được áp dụng tại khâu Batch Tokenization để điền vào những khoảng trống dư thừa, tạo thành hệ tensor mượt mà có sức chứa dài bằng câu dài nhất. 

Các phép đo chéo được tiến hành dựa trên:
1. **Target Word:** Vector đại diện cho chữ `"her"`.
2. **Non-Target Word:** Vector đại diện cho từ đứng ngay đằng trước chữ `"her"` (ví dụ: "introduced", "married").
3. **Random Control:** Vector xáo trộn bắt cặp ngẫu nhiên giữa bất cứ vị trí và bất cứ Layer tùy ý nào.

---

## 3. Khám Phá Trực Quan (Analysis & Visualizations)

### 3.1. Phân Tích Độ Lệch Ngẫu Nhiên (Randomly Shuffled Angles)
Với các vector không có định hướng liên kết, mức chênh lệch góc dao động xoay quanh $70^\circ \to 80^\circ$. Đáng lẽ nó phải sát mốc cự tuyệt $90^\circ$ (Orthogonal - vuông góc hoàn toàn). Mức hụt này chứng minh phát hiện ở phần đầu khoá học: Tokenizer GPT luôn bị kẹp một biên độ thiên kiến độ chệch (Bias) cho phép chừng mực các Cực lân cận từ vựng mang chỉ số dương với nhau.

### 3.2. Cú "Quay Xe" Lịch Sử Tại Tầng Nông Thiết Lập (Early Blocks)
Vào thời điểm nạp vector gốc thoát khỏi Embedding Matrix tiến vào Khối Transformer Số 1, vector của "her" và mọi từ khác nhận một cú rẽ cực mạnh, gần như quay ngắt góc $90^\circ$ hoàn chỉnh. 
**Giải mã cơ học:** Khối Transformer vòng 1 phải gồng gánh luồng dung nạp văn cảnh nguyên sinh khổng lồ, trộn lẫn Positional Encoding vào ý nghĩa văn phạm thô (Zero context modulation).

### 3.3. Dòng Chảy Thay Đổi Tiệm Cận Khối Đại Từ (Pronoun Binding Constraints)
Tiến hành nội suy t-Test cho mức xoay của đại từ `"her"` so với các Động từ Non-Target liền kề, ta thấy từ Khối giữa đến Khối cuối: 
- Từ bình thường chững lại, chỉ quay lệch tầm $10^\circ$. 
- Đi sâu xuống Layer 30+, mức xoay của "her" (Target - $\approx 15^\circ$) trội hơn hẳn so với khối Non-Target (Động từ). Lý giải: Các chữ biểu thị ngôi xưng ("her") bị lệ thuộc nặng nề vào các cấu trúc liên đới vĩ mô (Co-reference bindings) ở mệnh đề phía sau. LLM bắt buộc phải sửa vector của đại từ liên tục để nắn khớp ngữ cảnh phức tạp của các nhân vật thay vì giữ nó cố định.

---

## 4. Kết Luận
Việc ứng dụng toán học hình chiếu (Arc Cosine) biến việc theo dõi tính trừu tượng của Transformers thành việc cân chỉnh vòng xoay kim la bàn rất dễ diễn giải. Thông qua độ xoay, ta bắt quả tang LLM chỉ tốn duy nhất 1 Layer đầu để nhồi nhét khái niệm cú pháp cục bộ, và dành đến hàng chục Layers cuối để điều hướng và tinh chỉnh các khái niệm đa tham chiếu như danh đại từ. 

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm đo đạc từ thư viện lệnh ngầm tại `aero_LLM_01_Calculating rotations of embeddings vectors.md` (Triển khai công thức toán Arc Cosine cho Embeddings và thống kê Independent T-test).
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[Phân Khảo Token Embeddings: Đo Lường Góc Quay Của Vector Biểu Diễn](aero_llm_01_calculating_rotations_of_embeddings_vectors.md)** | [Xem bài viết →](aero_llm_01_calculating_rotations_of_embeddings_vectors.md) |
| [Thử Thách Lập Trình (Code Challenge): Tiến Hóa Đa Tầng Của Các Điều Chỉnh Góc Quay Tuần Tự](aero_llm_02_codechallenge_laminar_evolution_of_sequential_angular_adjustments.md) | [Xem bài viết →](aero_llm_02_codechallenge_laminar_evolution_of_sequential_angular_adjustments.md) |
| [Đo Lường Độ Dài Đường Dẫn (Path Length) Sự Tương Quan Với Dự Đoán Token](aero_llm_03_path_length_and_logit_token_prediction.md) | [Xem bài viết →](aero_llm_03_path_length_and_logit_token_prediction.md) |
| [Thử Thách Lập Trình: Phân Rã Độ Dài Đường Dẫn Luồng Số Dư (Phần 1)](aero_llm_04_codechallenge_residual_stream_path_length_decomposition_part_1_.md) | [Xem bài viết →](aero_llm_04_codechallenge_residual_stream_path_length_decomposition_part_1_.md) |
| [Thử Thách Lập Trình: Phân Rã Độ Dài Đường Dẫn Luồng Số Dư (Phần 2)](aero_llm_05_codechallenge_residual_stream_path_length_decomposition_part_2_.md) | [Xem bài viết →](aero_llm_05_codechallenge_residual_stream_path_length_decomposition_part_2_.md) |
| [Quỹ Đạo Không Gian Trạng Thái (State-Space Trajectories) Của Hệ Vector Ngôn Ngữ](aero_llm_06_state_space_trajectories_through_embedding_space.md) | [Xem bài viết →](aero_llm_06_state_space_trajectories_through_embedding_space.md) |
| [Phân Loại Từ Loại Bằng Thư Viện SpaCy Trong Phân Tích Mechanistic Interpretability](aero_llm_07_parts_of_speech_with_spacy_library.md) | [Xem bài viết →](aero_llm_07_parts_of_speech_with_spacy_library.md) |
| [Thử Thách Lập Trình: So Sánh Độ Dài Quỹ Đạo Của Danh Từ Và Tính Từ (Phần 1)](aero_llm_08_codechallenge_do_nouns_or_adjectives_have_longer_trajectories_part_1_.md) | [Xem bài viết →](aero_llm_08_codechallenge_do_nouns_or_adjectives_have_longer_trajectories_part_1_.md) |
| [Thử Thách Lập Trình: So Sánh Độ Dài Quỹ Đạo Của Danh Từ Và Tính Từ (Phần 2)](aero_llm_09_codechallenge_do_nouns_or_adjectives_have_longer_trajectories_part_2_.md) | [Xem bài viết →](aero_llm_09_codechallenge_do_nouns_or_adjectives_have_longer_trajectories_part_2_.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
