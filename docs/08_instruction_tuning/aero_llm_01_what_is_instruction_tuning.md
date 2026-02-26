
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [08 instruction tuning](index.md)

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
# Instruction Tuning (Tinh Chỉnh Bằng Chỉ Thị) Trong Các Mô Hình Ngôn Ngữ Lớn (LLMs)

## Tóm tắt

Bài viết này trình bày cơ sở lý thuyết và bản chất toán học của **Instruction Tuning** - một kỹ thuật tinh chỉnh (fine-tuning) đóng vai trò nòng cốt trong việc chuyển đổi các Mô hình Ngôn ngữ Lớn (LLMs) từ trạng thái dự đoán văn bản thuần túy sang trạng thái các trợ lý AI có khả năng tuân thủ mệnh lệnh của con người. Dựa trên các tài liệu nghiên cứu hiện đại, chúng tôi phác họa kiến trúc toán học liên quan đến mô hình xác suất tự hồi quy (autoregressive probability), kỹ thuật tối ưu hóa qua hàm giảm thiểu Cross-Entropy, cùng với các rào cản hiện tại như hiện tượng ảo giác (hallucination) và thiên kiến học máy. Bài viết đồng thời phân định mối tương quan giữa thuật toán này với RLHF trong kiến trúc An toàn AI.

---

## 1. Giới thiệu

Sự tiến hóa của các LLMs như GPT-3, PaLM, và LLaMA đã minh chứng cho khả năng học tập không giám sát (unsupervised learning) vô tiền khoáng hậu dựa trên các tập dữ liệu quy mô toàn bộ internet. Tuy nhiên, một mô hình nền tảng (foundation model) thông thường vấp phải khó khăn nghiêm trọng khi phải tuân thủ các chỉ báo mệnh lệnh trực tiếp từ người dùng. Chẳng hạn, khi nhận lệnh "Hãy dịch câu này sang tiếng Việt", chúng có thể tiếp tục bổ sung thêm các câu vào đoạn văn tiếng Anh ban đầu thay vì dịch nó.

*Instruction Tuning* ra đời định hình lại không gian huấn luyện. Bằng cách tái cấu trúc khối tập dữ liệu dưới định dạng chuỗi "Câu lệnh - Phản hồi" (Instruction - Response), phương pháp này khơi dậy năng lực suy luận Zero-shot của mô hình, chuyển ngữ cảnh của một bộ sinh chữ (text-completer) thành một hệ thống đối thoại có mục đích xác định.

---

## 2. Bản Chất Sự Khác Biệt: Fine-Tuning Truyền Thống và Instruction Tuning

Một mô hình tự hồi quy tiêu chuẩn vận hành theo xu hướng đưa ra dự đoán mã thông báo (token) tiếp theo tùy theo một cửa sổ chuỗi lịch sử. Fine-tuning truyền thống thường thu hẹp trọng số của mô hình vào một tác vụ cụ thể và giới hạn duy nhất (VD: phân loại cảm xúc tích cực/tiêu cực).

Trái lại, **Instruction Tuning** bao trùm hàng ngàn tác vụ khác nhau, trong đó tất cả đều được biểu diễn thông qua cấu trúc ngôn ngữ tự nhiên (Natural Language Instructions). Mục tiêu không chỉ là "học thực thi duy nhất một tác vụ", mà là "học **cách làm theo** đa dạng chỉ dẫn".

### 2.1 Cấu Trúc Đặc Trưng Dữ Liệu
Mỗi một mẫu dữ liệu trong bộ dữ liệu huấn luyện thường là một tổ hợp các yếu tố:
- **Câu lệnh (Instruction):** Định hướng hành vi ứng xử, bối cảnh cho mô hình (VD: "Phân tích tâm lý nhân vật chính trong văn bản dưới đây").
- **Ngữ cảnh bổ trợ (Input):** Đoạn văn bản cụ thể cần xử lý (nếu có).
- **Phản hồi mong đợi (Target Output):** Chuỗi token đáp án hoàn hảo mà mô hình cần phải tối đa hóa xác suất tái tạo (maximize likelihood).

---

## 3. Khung Nền Toán Học Của Instruction Tuning

Quá trình Instruction Tuning về bản chất vẫn tuân thủ các định lý thống kê về xác suất của học máy hiện đại (Next-token prediction). Tuy nhiên, hàm phân phối được gò ép mạnh lại vào định dạng đặc thù của lời nhắc (prompt bias).

### 3.1 Mô hình Xác Suất Tự Hồi Quy (Autoregressive Probability Model)

Khi đầu vào là một chuỗi token $X = (x_1, x_2, ..., x_t)$, mạng mô hình sẽ được huấn luyện để học cách cực đại hóa xác suất có điều kiện của toàn bộ chuỗi:

$$

P_\theta(X) = \prod_{t=1}^{T} P_\theta(x_t \mid x_{<t})

$$


Trong phương trình thống kê trên:
- $\theta$ đại diện cho cấu trúc ma trận tham số (weights) khổng lồ nội bộ của mạng nơ-ron Transformer.
- $x_{<t}$ là phần bối cảnh lưu giữ tất cả các vector token đứng trước vị trí $t$.

### 3.2 Huấn luyện với Hàm Mất Mát Negative Log-Likelihood (NLL)

Trọng tâm của pha Instruction Tuning (SFT - Supervised Fine Tuning), chúng ta chỉ muốn tính toán lỗi trên phạm vi mô hình sinh ra phần phản hồi $Y = (y_1, y_2, ..., y_N)$ khi cho trước biểu thức Chỉ thị $I$. Hàm mục tiêu (Objective function) dựa trên Cross-Entropy Loss được thiết lập lại dưới biểu diễn NLL (Negative Log-Likelihood) để che vạch (masking) phần lệnh gốc:

$$

\mathcal{L}_{SFT}(\theta) = - \frac{1}{N} \sum_{i=1}^{N} \log P_\theta(y_i \mid I, y_{<i})

$$


Điểm khác biệt ở đây là thuật toán lan truyền ngược (back-propagation) chỉ gửi tín hiệu lỗi (gradient) tính trên mạng của tập token thuộc về Output $Y$ (phần phản hồi ảo). Còn đối với các token mang vai trò Prompt $I$ (mệnh lệnh), Loss được nhân với không để chúng bị che khuất, tránh việc mô hình học ngược lại phong cách "ra lệnh" cho con người.

### 3.3 Tối Ưu Hóa Bằng Thuật Toán Gradient Descent

Nhằm di chuyển hội tụ hệ thống ma trận $\theta$, thuật toán tăng cường tối ưu động lượng như Adam hoặc AdamW được triển khai thông qua công thức Gradient Descent:

$$

\theta_{k+1} = \theta_k - \eta \cdot \nabla_\theta \mathcal{L}_{SFT}

$$


Trong đó, $\eta$ là hệ số tốc độ học (learning rate), và $\nabla_\theta \mathcal{L}_{SFT}$ biểu trưng cho đạo hàm riêng vi phân của hàm mất mát. 

---

## 4. Mối Liên Kết Tương Giao Với Căn Chỉnh An Toàn (Alignment & RLHF)

Instruction Tuning (hay còn được định danh là Supervised Fine-Tuning - SFT trong kiến trúc Ouyang 2022) đóng vai trò chất xúc tác trung tâm, tạo nền tảng thiết yếu để tiến tới một hệ thống tinh chỉnh An toàn (AI Safety) nghiêm ngặt hơn là **RLHF** (Reinforcement Learning from Human Feedback) - Cơ chế phạt và thưởng dựa trên hàm phần thưởng từ đánh giá con người. 

Khi kết hợp với quy trình RLHF (điển hình bằng thuật toán PPO), hàm tối ưu của mô hình sẽ trải qua quá trình Regularization với $Kullback–Leibler (KL) divergence$ nhằm tránh việc mô hình suy sụp hoàn toàn hình dáng vốn có (mode collapse) so với bản chuẩn Instruction Tuning ban đầu:

$$

\mathcal{L}_{RL} = \mathbb{E}_{x \sim \pi_\theta}[R(x, y)] - \beta D_{KL}(\pi_\theta \mid \mid \pi_{ref})

$$


Tham số ràng buộc $\pi_{ref}$ ở đây chính là bộ khung mô hình được giải xuất ra từ việc chắt lọc qua Instruction Tuning. KL Divergence ép mô hình duy trì sự linh hoạt tri thức nền của SFT trong lúc dần hội tụ lại với hàng rào an toàn cực hình do môi trường con người định ra.

---

## 5. Hạn Chế Hiện Tại

Bất chấp sự đột phá về trải nghiệm giao tiếp, quy trình Instruction Tuning đang phải vượt qua nhiều rào cản nhận thức:
- **Hiện tượng Ảo Giác (Hallucination):** Nếu một cấu trúc lệnh mang sắc thái xa lạ hoặc chưa từng tiếp xúc (Out-of-distribution) rơi vào ma trận phân phối, mô hình ngôn ngữ vẫn sinh ra vector cao điểm nhất dựa trên Softmax. Hậu quả là AI tự tin dàn dựng kiến thức vật lý thay vì nhận thức được độ mù vô hướng của bản thân (lack of epistemic awareness).
- **Hấp Thu Thiên Kiến (Bias Integration):** Các chuẩn mực thiên kiến xã hội sẽ vô tình được khuếch đại vào trọng số nếu các annotator có góc nhìn chủ quan ngầm trong lúc xây dựng bảng mẫu instruction.
- **Tiêu tốn nguồn lực tạo dữ liệu có giám sát:** Tối ưu hóa Instruction đòi hỏi bộ mẫu phản hồi (gold-standard responses) quy mô phải do chính con người xử lý. Nó tạo ra rào cản chi phí đè nặng lên các viện nghiên cứu.

---

## 6. Kết luận

Instruction Tuning không thực thi tái viết kiến trúc mạng phi tuyến học sâu cốt lõi như Transformer, nhưng nó mang lại định nghĩa biến thiên toàn cục về mô hình hóa ứng dụng con người. Việc ép buộc không gian ma trận tự hồi quy phải thu hẹp trong phạm vi hàm tỷ suất NLL trên bộ lệnh đích giúp một siêu dữ liệu (foundation weights) hóa phép thành những cố vấn phân tích đa phong cách. Phương pháp này chính là điểm khởi đầu thiết yếu để Trí tuệ Nhân tạo hiện hữu tiến về trạng thái Căn chỉnh Đạo đức chặt chẽ hơn.

---

## Tài liệu tham khảo

1. **Wei, J. et al. (2022).** *Finetuned Language Models Are Zero-Shot Learners.* (Nghiên cứu về mô hình FLAN).
2. **Ouyang, L. et al. (2022).** *Training Language Models to Follow Instructions with Human Feedback.* (Cơ sở kiến trúc InstructGPT).
3. **Brown, T. et al. (2020).** *Language Models are Few-Shot Learners.* (Đánh giá giới hạn tư duy Zero-shot).
4. **Sanh, V. et al. (2022).** *Multitask Prompted Training Enables Zero-Shot Task Generalization.* (Nghiên cứu mô hình T0).
5. **Vaswani, A. et al. (2017).** *Attention Is All You Need.* (Nền tảng kiến trúc mạng Transformer).
6. **Schulman, J. et al. (2017).** *Proximal Policy Optimization Algorithms.* (Ứng dụng Loss RL PPO).
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[Instruction Tuning (Tinh Chỉnh Bằng Chỉ Thị) Trong Các Mô Hình Ngôn Ngữ Lớn (LLMs)](aero_llm_01_what_is_instruction_tuning.md)** | [Xem bài viết →](aero_llm_01_what_is_instruction_tuning.md) |
| [Instruction Tuning trong Mô hình Ngôn ngữ Lớn](aero_llm_02_some_datasets_for_instruction_tuning.md) | [Xem bài viết →](aero_llm_02_some_datasets_for_instruction_tuning.md) |
| [Huấn luyện Chatbot theo Instruction Tuning và Mô hình System–User–Assistant](aero_llm_03_training_a_chatbot_with_system_user_assistant.md) | [Xem bài viết →](aero_llm_03_training_a_chatbot_with_system_user_assistant.md) |
| [Instruction Tuning với GPT-2 trong Huấn luyện Mô hình Ngôn ngữ](aero_llm_04_instruction_tuning_with_gpt2.md) | [Xem bài viết →](aero_llm_04_instruction_tuning_with_gpt2.md) |
| [aero llm 05 codechallenge instruction tuning gpt2 large part 1](aero_llm_05_codechallenge_instruction_tuning_gpt2_large_part_1_.md) | [Xem bài viết →](aero_llm_05_codechallenge_instruction_tuning_gpt2_large_part_1_.md) |
| [Phân tích nâng cao quá trình Instruction Tuning cho GPT-2 Large: Ổn định huấn luyện, động học gradient và tối ưu hoá tính toán](aero_llm_06_codechallenge_instruction_tuning_gpt2_large_part_2_.md) | [Xem bài viết →](aero_llm_06_codechallenge_instruction_tuning_gpt2_large_part_2_.md) |
| [Reinforcement Learning from Human Feedback (RLHF): Cơ sở lý thuyết, mô hình toán học và ứng dụng trong huấn luyện mô hình ngôn ngữ lớn](aero_llm_07_reinforcement_learning_from_human_feedback_rlhf_.md) | [Xem bài viết →](aero_llm_07_reinforcement_learning_from_human_feedback_rlhf_.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
