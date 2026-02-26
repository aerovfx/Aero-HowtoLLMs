
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [05 embeddings spaces](index.md)

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
# So sánh Biểu Diễn Từ Vựng giữa Wikipedia và Twitter bằng Phân Tích Tương Đồng Biểu Diễn (RSA)

## Tóm tắt

Trong nghiên cứu xử lý ngôn ngữ tự nhiên (NLP), các mô hình embedding học được biểu diễn vector của từ dựa trên ngữ cảnh. Tuy nhiên, khi hai mô hình được huấn luyện trên các miền dữ liệu khác nhau — ví dụ như từ điển bách khoa toàn thư của [Wikipedia](chatgpt://generic-entity?number=0) và dữ liệu mạng xã hội từ [Twitter](chatgpt://generic-entity?number=1) — thì không gian vector thu được có thể khác biệt đáng kể. Bài viết này trình bày phương pháp so sánh hai không gian embedding thông qua **Cosine Similarity** và **Representational Similarity Analysis (RSA)**, minh họa bằng câu mẫu “The quick brown fox jumps over the lazy dog”. Các công thức toán học được bổ sung nhằm làm rõ nền tảng lý thuyết.

---

## 1. Giới thiệu

$$
Word embedding ánh xạ mỗi từ w vào một vector \mathbf{v}_w \in \mathbb{R}^d, trong đó:
$$

$$
f: w \rightarrow \mathbf{v}_w
$$

với $d$ là số chiều của không gian nhúng.

Khi hai mô hình embedding được huấn luyện trên hai tập dữ liệu khác nhau (ví dụ: văn bản bách khoa và tweet ngắn), ta có:

$$

$$

f_{wiki}(w) = \mathbf{v}_w^{(wiki)}

$$

$$

$$
f_{twitter}(w) = \mathbf{v}_w^{(twitter)}
$$

$$
Do khác biệt về miền dữ liệu và phân bố ngôn ngữ, các vector thu được không thể so sánh trực tiếp từng chiều. --- ## 2. Độ tương đồng Cosine Để đo mức độ tương đồng giữa hai từ w_i và w_j trong cùng một mô hình, ta sử dụng **cosine similarity**:
$$

$$
\text{cosine}(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j} {\|\mathbf{v}_i\| \|\mathbf{v}_j\|}
$$

$$
Trong đó: - \mathbf{v}_i \cdot \mathbf{v}_j là tích vô hướng. - \\mid\mathbf{v}_i\\mid là chuẩn Euclid:
$$

$$
\|\mathbf{v}_i\| = \sqrt{\sum_{k=1}^{d} v_{ik}^2}
$$

$$
Cosine similarity nằm trong khoảng:
$$

-1 $\le$q \text{cosine} $\le$q 1

$$
Quan sát thực nghiệm cho thấy trong một số cặp từ, embedding từ Twitter cho giá trị cosine cao hơn so với embedding từ Wikipedia, phản ánh tính ngữ cảnh gần gũi hơn trong văn bản mạng xã hội. --- ## 3. Vấn đề: Không gian embedding khác nhau Mặc dù có thể so sánh cosine similarity trong *cùng một mô hình*, ta không thể so sánh trực tiếp:
$$

\mathbf{v}_w^{(wiki)} \neq \mathbf{v}_w^{(twitter)}

$$
Lý do: 1. Các không gian được học độc lập. 2. Trục tọa độ không đồng nhất. 3. Phép quay (rotation) của không gian không làm thay đổi khoảng cách nội tại nhưng làm thay đổi tọa độ tuyệt đối. Giả sử tồn tại một ma trận quay trực giao \mathbf{R}:
$$

$$
\mathbf{v}_w^{(twitter)} \approx \mathbf{R} \mathbf{v}_w^{(wiki)}
$$

$$
Khi đó, tọa độ khác nhau nhưng cấu trúc tương đối có thể vẫn được bảo toàn. --- ## 4. Representational Similarity Analysis (RSA) ### 4.1 Ý tưởng RSA không so sánh vector trực tiếp, mà so sánh **ma trận tương đồng nội bộ** giữa các từ trong từng mô hình. Giả sử ta có tập n từ trong câu: > “The quick brown fox jumps over the lazy dog”
$$

Ta xây dựng ma trận tương đồng S \in \mathbb{R}^{n \times n}:

$$

$$

S_{ij} = \text{cosine}(\mathbf{v}_i, \mathbf{v}_j)

$$

$$

Ta có:

$$
S^{(wiki)} \quad \text{và} \quad S^{(twitter)}
$$

---

### 4.2 So sánh hai ma trận

Ta vector hóa phần tam giác trên (không tính đường chéo):

$$
\mathbf{s}^{(wiki)}, \quad \mathbf{s}^{(twitter)}
$$

Sau đó tính hệ số tương quan Pearson:

$$

$$

r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})} {\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}}

$$

$$

Nếu:

- $r $\approx$ 1$: Hai không gian có cấu trúc quan hệ tương đồng cao.
- $r $\approx$ 0$: Cấu trúc khác biệt.
- $r < 0$: Quan hệ nghịch đảo.

---

## 5. Minh họa quy trình thực nghiệm

### Bước 1: Lấy chỉ số từ (word indices)

Với mỗi từ $w$ trong câu:

$$
\text{index}_{wiki}(w)
$$

$$
\text{index}_{twitter}(w)
$$

Lưu ý: Một số từ có thể không xuất hiện (ví dụ: chữ hoa “The”).

---

### Bước 2: Trích xuất embedding

$$

$$

\mathbf{v}_w^{(wiki)} = E^{(wiki)}[\text{index}(w)]

$$

$$

$$
\mathbf{v}_w^{(twitter)} = E^{(twitter)}[\text{index}(w)]
$$

$$
--- ### Bước 3: Tính ma trận tương đồng
$$

$$
S^{(model)}_{ij} = \frac{\mathbf{v}_i \cdot \mathbf{v}_j} {\|\mathbf{v}_i\| \|\mathbf{v}_j\|}
$$

$$
--- ### Bước 4: Tính tương quan giữa hai ma trận
$$

$$
\text{RSA score} = \text{corr}(\mathbf{s}^{(wiki)}, \mathbf{s}^{(twitter)})
$$

$$
--- ## 6. Phân tích kết quả Nếu embedding từ Twitter cho giá trị cosine cao hơn trong nhiều cặp từ, điều này có thể phản ánh: - Ngôn ngữ trên mạng xã hội mang tính ngữ cảnh chặt chẽ. - Các từ xuất hiện trong cấu trúc hội thoại ngắn, làm tăng mật độ đồng xuất hiện. Trong khi đó, Wikipedia có phong cách học thuật, phân bố từ rộng hơn, dẫn đến cấu trúc embedding phân tán hơn. --- ## 7. Thảo luận RSA cho phép ta: - So sánh hai không gian embedding không cùng hệ trục. - Đánh giá tính tương đồng cấu trúc. - Tránh phụ thuộc vào tọa độ tuyệt đối. Phương pháp này thường được sử dụng trong: - Khoa học thần kinh tính toán. - So sánh mô hình ngôn ngữ lớn. - Phân tích đa miền dữ liệu. --- ## 8. Kết luận So sánh embedding giữa Wikipedia và Twitter không thể thực hiện bằng cách đối chiếu trực tiếp vector. Tuy nhiên, thông qua cosine similarity và đặc biệt là Representational Similarity Analysis (RSA), ta có thể đánh giá mức độ tương đồng cấu trúc giữa hai không gian biểu diễn. Về mặt toán học:
$$

\text{So sánh trực tiếp vector} \neq \text{So sánh cấu trúc quan hệ}