# Phân Tích Sự Phân Cụm và Tương Đồng Biểu Diễn (RSA) Trong Ma Trận Q và K

## Tóm tắt (Abstract)
Nghiên cứu này chuyển hướng từ việc khảo sát sự nhất quán của một token đích mang các ngữ cảnh khác nhau sang việc đánh giá cách mô hình ngôn ngữ (như GPT-2 Medium) mã hóa các từ đích riêng biệt (thuộc 3 nhóm ngữ nghĩa) dưới cùng một ngữ cảnh chung. Thông qua hai chỉ số quan trọng là **Chỉ số Chọn lọc Phân nhóm (Selectivity Index)** và **Phân tích Tương đồng Biểu diễn (Representational Similarity Analysis - RSA)** đối với ma trận truy vấn (Q) và khóa (K), báo cáo chứng minh một điều mạnh mẽ: không gian học được của trí tuệ nhân tạo có xu hướng gom cụm định hướng từ vựng (grouping category) và các cách thức chia sẻ tính tương đồng biểu diễn tại mạng cấu trúc Q và K là đồng dạng đến kinh ngạc.

---

## 1. Mở Đầu (Introduction)
Các nơ-ron mạng trong `Self-Attention` không lưu trữ thông tin rời rạc mà tổ chức chúng thành các cấu trúc đại số trừu tượng. Để đo đạc cấu trúc hình học đại số này, chúng tôi ứng dụng `Representational Similarity Analysis` (RSA). Cụ thể, thay vì xem xét các giá trị tuyệt đối hay so sánh các lớp ẩn lệch pha (dimensionalities difference), ta chỉ tập trung tính ma trận tương đồng (similarity matrix) trong nội bộ một không gian mạng Q hoặc K. Đồng thời, đánh giá hiện tượng "tụ đàn" ngữ nghĩa thông qua `Selectivity Index`.

Mục tiêu chính: Trả lời câu hỏi *"Liệu kiến trúc của Q có hiểu và sắp xếp ý nghĩa các từ vựng giống như cách K tổ chức hay không?"*

---

## 2. Phương Pháp Chuyên Biệt (Methodology)

### 2.1. Đảo Ngược Kịch Bản Thực Nghiệm
Đảo ngược với tình huống trước, dữ liệu đầu vào hiện tại được kiểm soát bằng một chuỗi văn cảnh giống hệt nhau về lý lịch: *"The next word is [Target Word]"*.
- **Tập Token Đích (Target Words):** Bố trí 34 tokens độc lập thuộc 3 hạng mục ngữ nghĩa phân biệt gồm Vũ trụ (Space), Nội thất (Furniture) và Trái cây (Fruits). 
- Do tiền sử đoạn văn đứng trước hoàn toàn trùng khớp, hệ thống giải ép các tín hiệu $Attention$ trên token đích sau cùng sẽ chỉ tập trung vào khác biệt cốt lõi ở nhóm phân loại.

### 2.2. Đo Lường Sự Chọn Lọc Kéo Cụm (Selectivity Index)
Chỉ số (Index) này tính tỷ lệ giữa mức liên kết "cùng loại" và "kích thước chéo loại":
$$ \text{Selectivity Index} = \frac{\text{Ave18-RAGe}(\text{Cosine}_{cùng\ nhóm})}{\text{Ave18-RAGe}(\text{Cosine}_{khác\ nhóm})} $$
Thông qua thuật toán Matrix Mask, ta áp cho cụm nhóm Vũ trụ, Nội thất, Trái cây để tính tổng điểm. Kết quả cho điểm số $> 1.0$ là một bằng chứng rõ nét của hiệu ứng nơ-ron phân nhóm đặc trưng.

### 2.3. Phân Tích Tương Đồng Biểu Diễn (RSA - Representational Similarity Analysis)
Khai triển qua các bước:
1. So sánh từng chéo ma trận Cosine Similarity nội bộ Matrix $Q$ lên biểu đồ histogram.
2. Lập lại tương tự với Vector ma trận $K$.
3. Cắt lấy hai khu vực ma trận tương đồng (Upper Triangle/ Unique elements matrix) của $Q$ và $K$ rồi tính Hệ số tương quan (Correlation).

---

## 3. Khám Phá Khối Dữ Liệu Nội Tại (Analysis & Results)

Tiến hành đánh giá cụ thể tại một tầng ngẫu nhiên (Ví dụ Layer 5 - GPT-2 Medium):
1. **Sức mạnh phân cụm Của Hệ Ma Trận ($K$)**: 
   Thực tiễn chỉ ra mạng truy hồi "khóa" ($K$) mang hệ số Cosine Similarities cục bộ cao hơn so với "truy vấn" ($Q$). Điều này hoàn toàn tuyến tính với lý thuyết: mạng $K$ mã hóa lịch sử ngữ cảnh (mà ở đây là hoàn toàn đồng nhất), do đó nó thể hiện sự nhạy bén và ôm đồm kết cấu chung chật chẽ hơn mạng $Q$ vốn dành để phóng tác điều mới lạ.
   
2. **Sự Đồng Điệu Cấu Trúc Biểu Diễn (RSA Correlation):**
   Mặc dù $Q$ và $K$ đảm nhận hai vai trò toán học khác biệt, tích hợp trọng số (weights matrix) độc lập, kết quả $RSA$ đạt mốc vô cùng ấn tượng ($r > 0.8 / 0.9$). Điều này vạch trần cơ chế chia sẻ: Dù thao tác rời rạc nhưng góc nhìn tổ chức hình học phân cụm giữa các từ đồng nghĩa bên trong hai mạng đều song trùng.

---

## 4. Kết Luận (Conclusion)
Nghiên cứu kết cẩn một công trình mang tính cơ học: Mô hình Transformer mã hóa khái niệm (Concept coding) thông qua các chùm lưới. Ở mỗi không gian, dẫu biên độ $Q$ và $K$ có độ co giãn biến đổi khác nhau, nhưng bức tranh trật tự thế giới quan về phân loại ngôn ngữ (loài quả, hệ sao, vật dụng) luôn gắn kết theo một hệ trục tương đồng. 

Chỉ số "Selectivity Index" và $RSA$ vạch ra một định lượng chính xác giúp ta đo đạc thành công sự kết tinh này ở tầng thứ 5 (layer 5). Mở đường cho câu hỏi ở các tầng sâu khác thì tình huống sẽ thay đổi ra sao.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh thí nghiệm: `aero_LLM_04_Grouping and RSA in Q and K matrices.md` (Hướng dẫn triển khai RSA, Selectivity Mask Category cho 34 Tokens GPT-2 Medium).
