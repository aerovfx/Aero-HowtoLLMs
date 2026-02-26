
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
# Thử Thách Lập Trình (Code Challenge): Tiến Hóa Đa Tầng Của Các Điều Chỉnh Góc Quay Tuần Tự

## Tóm tắt (Abstract)
Thực nghiệm này đánh giá mức độ biến đổi góc đo (angular adjustments) giữa các Embeddings Vectors liên tiếp bên trong cùng môt tầng mạng Transformer (intra-layer analysis). Trái ngược với phân tích xuyên tầng (cross-layer) ở phần trước, bài khảo sát tập trung rọi hệ chiếu vào sự lệch phương hướng giữa [Token Mục Tiêu] và [Token Kế Trước Nó], đối chứng với các cặp Token ghép ngẫu nhiên. Kết quả thu được hé lộ một bài toán thú vị (Mystery of Bimodal Angles): Chỉ số chênh lệch góc bị gãy đôi thành hai thái cực. Sau khi bóc tách, nghiên cứu chỉ ra rằng biên độ góc vọt lên dữ dội ở những cặp chứa Token đầu tiên của câu, minh chứng cho một đặc thù thiết yếu: Mô hình Ngôn ngữ có "độ trễ khởi động" (warm-up lag) và xử lý Token từ đầu tiên rất khác so với phần thân câu gốc.

---

## 1. Mở Đầu (Introduction)
Từ bài học đo lường mức độ tự chuyển hóa của một từ vựng khi đi xuyên qua mạng LLM bằng phương thức Góc lượng giác (Rotation Angle) thay vì Cosine Similarity, tiếp nối dòng chảy, ta đặt ra giả thuyết: Liệu sự khác biệt góc giữa các Vector của mạng từ vựng ngay trong cùng một ngữ cảnh *thuộc một Layer* sẽ biểu diễn như thế nào? Cặp Token kế sát nhau – có chia sẻ góc quay ổn định hơn so với hai Token không liên quan bất kỳ?
Hạt giống của thực nghiệm nằm ở việc theo dõi biến thiên góc giữa đại từ mục tiêu (`"her"`) và động từ đi kèm sát nút (ví dụ: `"promoted"`, `"admired"`). 

---

## 2. Tiết Thiết Lập Thực Nghiệm (Methodology)

### 2.1. Cấu Trúc Khảo Sát
- Kế thừa bộ dữ liệu 54 câu mồi xoay quanh đại từ nhân xưng `"her"`.
- Token mục tiêu đại biểu là $Target_{index}$ vị trí phát hiện qua Regex.
- Tái sử dụng công thức Độ dịch góc tiêu chuẩn (Arc Cosine). 

### 2.2. Phân Cụm Biến Số (Variables Clusters)
Ta thu thập hai phổ tín hiệu góc trên toàn bộ 48 Hidden Layers:
1. **Target sequence:** Góc sai phân giữa cặp liền kề: \{Token ở vị trí $(i)$ và Target Token ở vị trí $(i+1)$\}.
2. **Non-target sequence:** Cặp góc chênh hai Token ngẫu nhiên tùy ý sinh ra từ mọi nơi cấu trúc thuộc văn bản, với điều kiện chúng phải tuân thủ thứ tự trước và sau và không dính dáng đến Target Token. Phép đo này coi như mức Cơ sở đối chiếu kiểm chứng (Baseline).

---

## 3. Kháo Sát Đánh Giá Dữ Liệu: Bí Ẩn Đám Mây Đôi Mũ (The Bimodal Mystery)

### 3.1. Sự Tăng Trưởng Biên Độ Lệch (Amplitude Enlargement)
Kết quả ghi nhận cường độ bẻ góc (Rotational Shift) vượt xa bậc cỡ $0.1 \to 0.2$ Radians của bài thử trước. Thay vào đó, góc mở rất rộng. Vì so sánh 2 từ khóa khác mâm nhau trên không gian mạng, sự cách trở ngữ nghĩa là điều tất yếu. Biểu đồ quạt tán sắc từ khối Layers trung tâm tới điểm kích xuất từ ra, phản ánh biến số giãn xoắn phức hệ theo Context.

### 3.2. Hiện Tượng Chuẩn Phân Cực: Đỉnh Núi Hai Vòi (Bimodal Distribution)
Kết quả làm đồ thị bối rối khi toàn bộ đường line Cơ sở đỏ Non-target phân hóa thành 2 thái cực đục khoét hoàn toàn khác biệt:
- Nhóm 1: Vector chênh lệch lơ lửng sải mình quanh đường bình quân xấp xỉ mức $1.0\to 1.5$ rads. 
- Nhóm 2: Mép biểu đồ vần vũ những đường vọt dựng đứng kịch kim từ rất sớm (đạt $2.5\to 3.0$ rads).
Việc tính toán giá trị Trung vị trên hệ hình này vô hình chung tạo ra một đường giữa lơ lửng, không phản ánh bất kỳ khuynh hướng tập trung chuẩn (Central Tendency) hợp lý nào của hệ phân phối. Sự trung bình biến thành ảo giác.

---

## 4. Chìa Khóa Lời Giải (Mystery Reveal)
Để vạch trần phân mảnh, một bài test thứ 3 được cấu thành:
Dò quét góc độ tuần tự cặp Index tĩnh theo vị trí xuất hiện: $\{}$Vị trí $0\leftrightarrow 1\}$, }$\{$Vị trí $1\leftrightarrow 2\}$, }$\{$Vị trí $2\leftrightarrow 3\}$, v.v.

**Kết Quả Chốt:**
Tất cả các "tuyến dị biệt" quăng mình lên độ cao bứt tốc ngất ngưởng sinh ra thuần túy là cặp chứa *Vị trí số 0* và *1* (Tức là Token mở màn của Văn bản văn cảnh). 

Mảng Token hóa ban đầu không hề nắm giữ tín hiệu ngữ cảnh tham chiếu nào. Chỉ khi Token thứ 2 thứ 3 chen vào, mô hình mới tiếp nhập (nhồi context) đẩy văng vector xoắn xoáy đi theo một góc độ cự tuyệt khổng lồ. Kể từ Token thứ 2 trở đi, những Token kế tiếp sở hữu lực quay dịu dần và bắt đầu ổn định sự định hướng quỹ đạo đồng dạng hơn. 

---

## 5. Kết Luận
Việc quan sát luồng chảy tầng (laminar evolution) của Transformer đưa ra quy tắc sinh tồn với diễn biến thí nghiệm Mechanistic Interpretability: Bất cứ bộ chỉ số hay đánh giá hệ số chú ý nào đặt lên những **Token mở câu đầu tiên** của LLMs đều là nhiễu do "độ trễ thu điện" (engines warming up). Góc thay đổi của hai Vector liên tiếp có thể không hữu ích bằng luồng truy vết 1 Vector duy nhất. Dù vậy, nó khảm vào bài trắc nghiệm một nguyên tắc: Bỏ qua các chuỗi Token đầu của khối văn bản để tránh làm xiêu vẹo đồ thị chung. 

---

## Tài Liệu Tham Khảo (Citations)
1. Lý luận và giải quyết thực tiễn dựa trên tập code trích ở `aero_LLM_02_CodeChallenge Laminar evolution of sequential angular adjustments.md` (Hướng tiếp cận bẫy lỗi về Bimodal Data và quy tắc tảng băng chìm "warm-up first token").
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Phân Khảo Token Embeddings: Đo Lường Góc Quay Của Vector Biểu Diễn](aero_llm_01_calculating_rotations_of_embeddings_vectors.md) | [Xem bài viết →](aero_llm_01_calculating_rotations_of_embeddings_vectors.md) |
| 📌 **[Thử Thách Lập Trình (Code Challenge): Tiến Hóa Đa Tầng Của Các Điều Chỉnh Góc Quay Tuần Tự](aero_llm_02_codechallenge_laminar_evolution_of_sequential_angular_adjustments.md)** | [Xem bài viết →](aero_llm_02_codechallenge_laminar_evolution_of_sequential_angular_adjustments.md) |
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
