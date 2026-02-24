# Số Học Tuyến Tính và Rút Trích Tương Đồng Giữa Các Từ Nhúng (Word Embeddings Analogies)

## Tóm tắt

Khả năng thực hiện các phép toán đại số phi tuyến dạng `Vector(King) - Vector(Man) + Vector(Woman) = Vector(Queen)` là một trong những minh họa trực quan kinh điển nhất khi nhắc tới sức mạnh của Không gian nhúng từ (Word Embeddings). Bài báo này phân tách bản chất hình học của số học vector trong nghiên cứu ngôn ngữ, làm rõ phép trừ chiều không gian (Dimension subtraction) tương quan tới logic diễn giải ngữ nghĩa (semantic linear axes) như thế nào. Trái nghịch với những lời tâng bốc từ truyền thông, bài viết cũng đưa ra những sự giới hạn và rủi ro từ góc nhìn Hình học đại số tuyến tính của Embeddings.

---

## 1. Giới thiệu: Chuyển Ngữ Nghĩa Thành Các Tổ Hợp Tuyến Tính

Kể từ sự ra đời phôi thai của Word2Vec (Mikolov 2013) và sau đó là GloVe hay Transformers, hệ thống AI chuyển đổi cấu trúc ngữ nghĩa tiếng anh khô khan sang một không gian điểm Euclidean ($N$-dimensions). Nếu sự học hóa này thật sự bắt nguồn từ một lý thuyết tập hợp, các hướng và độ dọc vector bắt buộc phải mang tính cấu hình tuyến tính để trả lời cho những tỷ lệ tương đồng (Analogies). 

Trong không gian này, khoảng cách từ "Man" (Nam giới) đến "King" (Vua) được định hình bởi vector bản chất (Ví dụ: Sự quyền lực). Do đó việc sao chép vector bản chất này lên tọa độ của "Woman" (Nữ giới) trên lý thuyết phải trả về một vị trí rất gần với đại lượng mang ý nghĩa "Sự quyền lực của nữ giới" - tức là "Queen" (Nữ hoàng).

---

## 2. Kỹ Thuật Tính Toán Số Học Embeddings (Vector Arithmetic)

Khởi tạo hệ thống không gian vector liên tục (Continuous Vector Space Model), phép suy luận tương đồng bao hàm theo các bước:

### 2.1 Phương trình Tuyến tính Tương quan cơ sở
Gọi $v_w \in \mathbb{R}^D$ là biểu diễn vector $D$ chiều của từ $w$. Phương trình số học cốt lõi lấy ý tưởng từ quy luật hình bình hành (Parallelogram law):
$$
v_{analogy} = v_{king} - v_{man} + v_{woman}
$$
Đây là quá trình triệt tiêu (subtract) một vector thuộc tính trừu tượng (như *giới tính*) và tiêm (inject) vào một thành phần thuộc tính khác. 

### 2.2 Thuật toán Argmax với Cosine Similarity
Vì $v_{analogy}$ không chắc chắn đáp thẳng vào tâm của một từ vựng xác thực có sẵn (do độ trôi dạt - concept drift trong không gian nhiễu), bài toán hiện ra dưới dạng một hàm tìm điểm lân cận gần nhất (Nearest Neighbors Search):

$$
\text{target\_word} = \text{argmax}_{w \in V \setminus \{king, man, woman\}} \cos(v_{analogy}, v_w)
$$
Trong đó:
- $\cos(A, B) = \frac{A \cdot B}{\|A\|\|B\|}$ tính bằng ma trận khoảng cách Gram.
- Tập tìm kiếm $V$ phải loại bỏ các từ nằm ở phần nón gốc nhằm ngăn cản mô hình tái sinh ra đáp án tầm thường do sự bùng nổ của quy chuẩn chuẩn hóa L2 (L2 constraints).

---

## 3. Ranh Giới Ảo Ảnh: Sự Thiết Thiếu Tuyến Tính Ở Đồ Thị Ngôn Ngữ Phức Tạp

Tuy đạt độ kinh ngạc cao trên những từ có độ phổ biến cụ thể, khi làm toán ở những khái niệm mơ hồ như định nghĩa "Trục thời gian" (Time axis), việc tìm điểm lân cận với ma trận `v_{tomorrow} - v_{yesterday}` đa phần thất bại và trả về điểm dự đoán là những nhiễu ngẫu nhiên.

**Do Đâu Nảy Sinh Hiện Tượng Phân Đứt Sự Kiện?**
- **Hình Học Cung Phi Tuyến:** Số học cộng trừ tự suy diễn rằng không gian thông tin (Latent space distribution) tuân theo mô hình đại số tuyến tính Euclid. Nhưng thực chất các đa tạp vi phân phân bổ (embeddings manifolds) bị bóp méo qua cấu trúc phi tuyến hàm Softmax hoặc Relu của hàm mạng Neural Networks.
- **Tiếng Vọng Truyền Thông (Cherry Picking Artifact):** Khi truyền thông công nghệ liên tục lan truyền phép toán "King - Man", nó che giấu đi sự thật rằng phương trình này là bản thiết kế do con người lồng ghép (hand-picked), không mang tính tổng quát cho các cấu trúc cú pháp văn phạm vĩ mô và ngôn ngữ ẩn dụ (metaphors). Đánh giá sự tinh vi của AI trên sự phác họa giản đơn hoàn nguyên chủ nghĩa (reductionist view) tiềm tàng các lỗ hổng phòng thủ AI Safety, lãng quên rủi ro từ logic lập trình đứt gãy.

---

## 4. Kết luận

Mô hình vector tuyến tính số học trên các mô hình ngôn ngữ lớn xác nhận định dạng thông tin trong AI chứa một trật tự không gian liên tiếp. Thay vì dựa dẫm một cách mê tín rằng phép cộng trừ đơn giản có thể giải phẫu toàn bộ đặc tính tính toán (Computational linearity) của LLM, chúng ta cần hướng đến một phổ vi phân sâu rộng hơn để gỡ gạc mạng lưới thông điệp chéo của não bộ AI.

---

## Tài liệu tham khảo

1. **Mikolov, T., et al. (2013).** *Linguistic Regularities in Continuous Space Word Representations.* NAACL-HLT. (Sự khởi đầu của phương trình Analogy).
2. **Levy, O., & Goldberg, Y. (2014).** *Linguistic Regularities in Sparse and Explicit Word Representations.* CoNLL.
3. **Ethayarajh, K., et al. (2019).** *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings.* EMNLP.
4. Tài liệu bài giảng *Tokens investigating - Embeddings arithmetic and analogies*.
