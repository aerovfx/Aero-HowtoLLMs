
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
# Cạnh Tranh Tìm Từ Đồng Nghĩa BERT vs GPT: Cơ Chế Tokenization Đa Ký Tự

## Tóm tắt

Bài báo cáo thực nghiệm so sánh phương pháp trích lọc các mảng "hàng xóm" gần nhất dựa trên kỹ luật k-Nearest Neighbors (k-NN) bằng thuật chuẩn khoảng cách Euclidean giữa hai siêu kiến trúc mô hình ngôn ngữ lớn (LLMs): Hệ tự mã hóa sinh đôi BERT và Hệ tự hồi quy một chiều GPT-2. Điểm đặc sắc tập trung giải quyết bài toán sụp đổ của một token đa phần tử trước dấu cách không gian (Space tokens), cấu trúc mà GPT-2 chia rẽ các chuỗi đồng nhất. Hiện tượng trung bình hóa đa véc-tơ (vector mean-pooling) để ép hợp một Vector sẽ cho ra các từ vực xa rời với đồng cấu sinh học logic.

---

## 1. Bản Đồ Mật Độ Và Khoảng Cách Xa Tương Đối

Thực nghiệm bắt đầu với một từ hạt giống (Seed Token) không có chuỗi không gian xen vào, ví dụ: cụm `"ring"`.
Thuật toán lấy vetor mã của `ring` càn quét đo khoảng cách Euclidean Distance ($\| \vec{a} - \vec{b} \|$) so với hằng số $50.000$ (tập Vocab) các véctơ mã trong cả BERT và GPT-2. Các hệ quả trực quan:
- **Biểu Đồ Lệch Histograms:** Đường hình chuông (Gaussian curves) của GPT-2 và BERT có phân bố bình thường mượt mà và tập trung xa dần về khu vực trung bình. Cả hai đều chừa lại một dải siêu hẹp (Long-tail) từ khoảng cách cực tiểu cho vài Token siêu liên đới, trong khi lượng lớn hàng chục ngàn từ ở phương trời xa thẩm.
- Tuy nhiên, quần thể không trung chuẩn (Non-normalized points) của BERT được giữ co lại dầy đặc, trong khi GPT-2 tạo độ dãn mật độ vector cao hơn hàng chuỗi chỉ mục.

Khi Normalized Matrix ($\|\vec{v}\| = 1$), lực kéo của độ dài vector bị triệt biến, biến đồ thị Histogram Euclidean của hai gã khổng lồ này đè lên nhau trùng khớp thành một biểu đồ hợp vĩ duy nhất, hé lộ sức mạnh thực sự của hướng góc Vector (Direction Angles).

---

## 2. Hệ Mã Hóa Lệch (The Space Sensitivity)

Việc khai thác Synonym qua k-NN trên BERT dễ dàng cung cấp chuỗi Top 15 khá chặt chẽ: `rings`, `ringing`, `fifth`, `sixth`.. (các từ ngữ đồng lõa ngữ pháp). 
Khi ta thả hạt giống mới là `" ring"` (khoảng trắng nằm trước kí tự), BERT Tokenizer lập tức ném chuỗi khoảng trắng đi vì cơ chế Phân lớp Mức độ Chú ý (Classification token) của BERT không quan tâm yếu tố hình thức ngữ pháp hiển thị. 

**GPT-2 là một vũ trụ khác biệt:**
Bộ mã hóa Byte-Pair Encoding BPE của GPT-2 xem khoảng trắng cũng là xương sống cấu thành từ vựng nội hàm.
- Với hạt giống `"ring"`, GPT-2 tìm ra những token ngẫu nhiên dựa vào cấu trúc đồ họa hình học Orthographically (ví dụ: `ringa`, `ringred`, `drying`, `ping`) thay vì bất kì ý nghĩa ngữ nghĩa nào.
- Chỉ khi áp dụng Normalization và chèn dấu khoảng trống đầu hạt `" ring"`, GPT-2 mới khải huyền ra các mảng từ khóa Synonym đáng sợ như: `amulet` (bùa ngải chuỗi), `circle` (vòng xoay), `necklace` (chuỗi hạt), `bracelet` (vòng tay đeo). Tức là GPT chỉ hoạt động não bộ kết tủa Synonym khi từ vựng bị ngắt đứt với tiếp vị ngữ dư thừa.

---

## 3. Khủng Hoảng Phân Rã Tokenize Và Biện Pháp Mean Pooling
Thử thách bùng nổ khi sử dụng tìm kiếm đồng nghĩa cho hạt giống `"beauty"`. 
- Bật Tokenize của BERT: Nhận rễ `"beauty"` làm 1 Single Token $\to$ Euclidean Scan mượt mà.
- Bật Tokenize của GPT-2: Chữ `"beauty"` bị cưa xẻ nát bung thành **2 Tokens độc lập**.

Không thể dùng thước dây k-NN cho 2 ngọn véc-tơ độc lập, kiến trúc sư chỉ được phép chọn 1 trong 2 giải pháp:
1. Tính khoảng cách 50.000 điểm từ véc-tơ $\vec{v}_1$, làm tương tự cho $\vec{v}_2$. Sau đó cộng Ave18-RAGe 50.000 cặp khoảng cách (Khoảng cách kéo trung bình).
2. Ép trung bình 2 Véc-tơ bằng hàm nhúng Vector (Mean Pooling) $\vec{E}_{\text{seed}} = \frac{\vec{v}_1 + \vec{v}_2}{2}$. Sau đó dùng một Vector duy nhất này phóng chổi quét mạng lưới Không gian (Option 2).

Nếu dùng Mean-Pooling phương thức 2, không gian phân hóa trả bề một hệ tương đệ từ đồng nghĩa ấn tượng đỉnh điểm: Dải GPT-2 bắn ra `beautiful, gorgeous, pretty, wonderful, lovely`.
Việc sáp nhập mã độc lập không giết chết nội hàm, nó tạo ra Tình trạng Chuyển giao Đa hướng (Multi-direction Translation), một tính chất sống còn để kết tinh các kiến thức phức tạp của Human Language vào AI.

---

## Tài liệu tham khảo

1. **Bojanowski, P., et al. (2017).** *Enriching Word Vectors with Subword Information.* TACL (Cùng kiến trúc token hoá subword ảnh hưởng k-NN).
2. **Sennrich, H., et al. (2016).** *Neural Machine Translation of Rare Words with Subword Units.* ACL.
3. Tài liệu thực hành lập trình *BERT v GPT kNN kompetition.*
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
