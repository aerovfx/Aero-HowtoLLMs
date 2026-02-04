# Chapter 1: Understanding Large Language Models

Dưới đây là quy trình chi tiết để huấn luyện một **Mô hình Ngôn ngữ Lớn (LLM)** từ đầu, dựa trên 5 giai đoạn chính

---

### **1. Thu thập và xử lý dữ liệu**  
**Mục tiêu**: Tạo bộ dữ liệu đa dạng, chất lượng, sẵn sàng cho huấn luyện.  
- **Nguồn dữ liệu**:  
  - Văn bản đa dạng từ sách, bài báo, trang web (Common Crawl), code (GitHub), Wikipedia, diễn đàn...  
  - Ví dụ: GPT-3 dùng 570GB văn bản, BERT dùng BookCorpus + Wikipedia.  
- **Tiền xử lý**:  
  - Lọc nội dung độc hại, spam, trùng lặp (sử dụng **deduplication**).  
  - Chia văn bản thành các đoạn/phân đoạn (chunking).  
  - Chuẩn hóa văn bản: Xóa HTML, chuyển về lowercase (nếu cần), xử lý emoji/biểu tượng.  
- **Token hóa**:  
  - Sử dụng tokenizer như **BPE** (Byte-Pair Encoding) hoặc **SentencePiece** để chia văn bản thành các đơn vị con (subwords).  
  - Ví dụ: GPT dùng BPE, BERT dùng WordPiece.  
- **Xử lý thiên kiến (bias)**:  
  - Cân bằng dữ liệu theo giới tính, vùng miền, ngôn ngữ.  
  - Dùng công cụ như **DebiasWe** hoặc loại bỏ từ ngữ gây kỳ thị.  

---

### **2. Chuẩn bị kiến trúc mô hình**  
**Mục tiêu**: Thiết kế kiến trúc phù hợp với quy mô và tài nguyên tính toán.  
- **Chọn kiến trúc nền**:  
  - **Transformer** (phổ biến nhất) với các biến thể như GPT (autoregressive), BERT (bidirectional), T5 (encoder-decoder).  
- **Xác định quy mô**:  
  - Số lớp (layers), số head trong cơ chế **self-attention**, kích thước embedding.  
  - Ví dụ: GPT-3 có 96 lớp, 175 tỷ tham số.  
- **Khởi tạo tham số**:  
  - Dùng kỹ thuật khởi tạo trọng số như **He initialization** hoặc **Xavier initialization**.  
- **Chuẩn bị hệ thống phân tán**:  
  - Cấu hình **data parallelism** (chia dữ liệu) hoặc **model parallelism** (chia mô hình) trên nhiều GPU/TPU.  
  - Sử dụng framework như **PyTorch Distributed** hoặc **TensorFlow Mesh**.  

---

### **3. Huấn luyện mô hình (Pretraining)**  
**Mục tiêu**: Giúp mô hình học biểu diễn ngôn ngữ tổng quát từ dữ liệu thô.  
- **Tối ưu hóa**:  
  - Dùng optimizer **AdamW** với learning rate warmup (ví dụ: từ 1e-6 đến 3e-4).  
  - Gradient clipping để tránh exploding gradients.  
- **Quản lý bộ nhớ**:  
  - **Mixed Precision Training** (kết hợp FP16/FP32) để tiết kiệm bộ nhớ.  
  - **Gradient Checkpointing** (tính lại activation thay vì lưu trữ).  
- **Huấn luyện phân tán**:  
  - Batch size lớn (hàng triệu token), chia trên hàng trăm GPU.  
  - Ví dụ: GPT-3 huấn luyện trên 1.024 GPU V100 trong vài tuần.  
- **Theo dõi**:  
  - Giám sát loss, perplexity, gradient norms.  
  - Kiểm tra độ hội tụ qua các bài benchmark như **LAMBADA** (đoán từ tiếp theo trong văn cảnh dài).  

---

### **4. Tinh chỉnh mô hình (Fine-tuning & RLHF)**  
**Mục tiêu**: Điều chỉnh mô hình cho các tác vụ cụ thể hoặc cải thiện hành vi.  
- **Fine-tuning**:  
  - Huấn luyện lại trên tập dữ liệu nhỏ, chuyên biệt (ví dụ: hỏi đáp y tế, dịch máy).  
  - Dùng learning rate nhỏ hơn pretraining (ví dụ: 1e-5) để tránh overfitting.  
- **Reinforcement Learning from Human Feedback (RLHF)**:  
  - **Bước 1**: Thu thập phản hồi từ con người (ví dụ: xếp hạng các câu trả lời).  
  - **Bước 2**: Huấn luyện **reward model** để đánh giá câu trả lời.  
  - **Bước 3**: Tối ưu mô hình bằng PPO (Proximal Policy Optimization) để tối đa phần thưởng.  
  - Ví dụ: ChatGPT được tinh chỉnh bằng RLHF để tăng tính hữu ích và an toàn.  

---

### **5. Triển khai và đánh giá**  
**Mục tiêu**: Đưa mô hình vào thực tế và đo lường hiệu suất.  
- **Tối ưu hóa triển khai**:  
  - **Quantization**: Giảm độ chính xác trọng số (32-bit → 8-bit) để tăng tốc suy luận.  
  - **Pruning**: Loại bỏ các neuron không quan trọng.  
  - **Distillation**: Nén mô hình lớn thành mô hình nhỏ (ví dụ: DistilBERT).  
- **Đánh giá**:  
  - **Đo lường định lượng**: Perplexity, độ chính xác trên tập test (ví dụ: GLUE, SuperGLUE).  
  - **Đánh giá định tính**: Kiểm tra khả năng tạo văn bản mạch lạc, tránh toxic content.  
  - **Kiểm tra tính công bằng**: Phát hiện bias qua công cụ như **Fairness Indicators**.  
- **Giám sát sau triển khai**:  
  - Thu thập phản hồi người dùng, cập nhật mô hình định kỳ.  

---

### **Ví dụ về quy trình hoàn chỉnh**  
1. **GPT-4**:  
   - Pretraining trên hàng nghìn tỷ token từ internet → Fine-tuning cho các tác vụ cụ thể → RLHF để chỉnh hành vi.  
2. **BLOOM**:  
   - Huấn luyện trên 46 ngôn ngữ, xử lý bias qua cộng đồng mở → Triển khai dưới dạng mã nguồn mở.  

---

### **Thách thức chính**  
- **Chi phí**: Huấn luyện GPT-3 tốn ~$4.6 triệu.  
- **Đạo đức**: Nguy cơ lan truyền thông tin sai lệch, bias.  
- **Môi trường**: Tiêu thụ năng lượng lớn (ví dụ: huấn luyện BERT ~1,500 lbs CO₂).  

Nào cùng bắt đầu nhé 🚀
