
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [07 fine tune pretrained models](../index.md) > [fine tuning](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Nghiên Cứu Về Fine-Tuning Large Language Models: Từ Transformer Đến LoRA

## Tóm Tắt

Fine-tuning các mô hình ngôn ngữ lớn (Large Language Models - LLMs) đã trở thành một trong những kỹ thuật quan trọng nhất trong lĩnh vực xử lý ngôn ngữ tự nhiên hiện đại. Bài viết này trình bày tổng quan toàn diện về các phương pháp fine-tuning, từ kiến trúc Transformer cơ bản đến các kỹ thuật tối ưu như Parameter-Efficient Fine-Tuning (PEFT) và Low-Rank Adaptation (LoRA). Chúng tôi phân tích chi tiết kiến trúc của FLAN-T5, một trong những mô hình tiên phong trong việc áp dụng instruction tuning, đồng thời đề cập đến các công thức toán học minh họa và so sánh hiệu quả của các phương pháp khác nhau.

**Từ khóa:** Large Language Models, Fine-Tuning, Transformer, LoRA, PEFT, FLAN-T5, Transfer Learning

---

## 1. Giới Thiệu

### 1.1. Bối Cảnh và Tầm Quan Trọng

Trong những năm gần đây, các mô hình ngôn ngữ lớn (LLMs) đã cách mạng hóa lĩnh vực trí tuệ nhân tạo và có ảnh hưởng sâu rộng đến đời sống hàng ngày của con người [1]. Từ các trợ lý ảo như Siri và Alexa trong gia đình, đến các bot hỗ trợ khách hàng tự động, LLMs đang đứng sau việc nâng cao trải nghiệm người dùng và làm cho công nghệ trở nên dễ tiếp cận hơn [1].

Trong lĩnh vực y tế, LLMs hỗ trợ bác sĩ trong việc chẩn đoán nhanh hơn và xây dựng phác đồ điều trị cá nhân hóa. Giáo dục là một lĩnh vực khác đang được biến đổi bởi LLMs, với khả năng tạo ra một gia sư cá nhân hoạt động 24/7, có khả năng thích nghi với phong cách học của từng học sinh [1]. Trong lĩnh vực kinh doanh, từ soạn thảo email đến tạo báo cáo, LLMs giúp hợp lý hóa giao dịch, nâng cao hiệu quả vận hành.

### 1.2. Mục Tiêu của Bài Viết

Bài viết này nhằm cung cấp một cái nhìn toàn diện về:
- Kiến trúc Transformer - nền tảng của các LLMs hiện đại
- Mô hình FLAN-T5 và kỹ thuật instruction tuning
- Các phương pháp Transfer Learning và Fine-tuning truyền thống
- Kỹ thuật Parameter-Efficient Fine-Tuning (PEFT)
- Thuật toán Low-Rank Adaptation (LoRA) và các cải tiến

---

## 2. Kiến Trúc Transformer

### 2.1. Giới Thiệu về Transformer

Transformer, được giới thiệu bởi các nhà nghiên cứu tại Google vào năm 2017 trong bài báo "Attention Is All You Need" [2], đã trở thành xương sống của hầu hết các LLMs hiện đại. Kiến trúc này đã thay đổi căn bản cách máy tính xử lý ngôn ngữ.

Khác với các mô hình cũ như Recurrent Neural Networks (RNNs) hoặc Long Short-Term Memory (LSTMs) xử lý dữ liệu tuần tự, Transformer xử lý tất cả các phần của dữ liệu đồng thời [3]. Việc xử lý song song này tương tự như nhiều trạm làm việc trong bếp hoạt động cùng lúc, giúp tăng tốc đáng kể các tác vụ.

### 2.2. Cấu Trúc Cơ Bản của Transformer

Transformer bao gồm các lớp encoder và decoder. Encoder đọc và xử lý văn bản đầu vào, trong khi decoder tạo ra đầu ra dựa trên thông tin đó [3]. Có thể hình dung encoder như nhân viên bếp chuẩn bị nguyên liệu, còn decoder như đầu bếp kết hợp các nguyên liệu để tạo ra món ăn.

#### 2.2.1. Cơ Chế Self-Attention

Một trong những đổi mới quan trọng nhất của Transformer là cơ chế self-attention. Cơ chế này cho phép mô hình đánh giá tầm quan trọng của các từ khác nhau trong một câu so với các từ khác [3].

Công thức scaled dot-product attention được định nghĩa như sau [2]:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Trong đó:
- $Q$ (Query): Ma trận truy vấn
- $K$ (Key): Ma trận khóa  
- $V$ (Value): Ma trận giá trị
- $d_k$: Kích thước của vector khóa

Multi-head attention cho phép mô hình tập trung vào nhiều vị trí khác nhau trong câu cùng lúc:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

#### 2.2.2. Positional Encoding

Do Transformer xử lý dữ liệu song song nên cần thêm positional encoding để mô hình hiểu được thứ tự của các từ:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### 2.3. Ứng Dụng của Transformer trong LLMs

Các lớp của Transformer có thể được xem như các "bộ não thu nhỏ", mỗi lớp đưa ra quyết định riêng về phần nào của văn bản quan trọng [3]. Các lớp này xếp chồng lên nhau tạo thành mạng lưới mạnh mẽ giúp tinh chỉnh ngôn ngữ và khả năng sinh văn bản.

LLMs sử dụng kiến trúc Transformer để thực hiện xuất sắc các tác vụ như dịch thuật, tạo nội dung, và nhiều tác vụ khác bằng cách hiệu quả trong việc hiểu và tạo ra văn bản giống con người [3].

---

## 3. Mô Hình FLAN-T5

### 3.1. Giới Thiệu về FLAN-T5

FLAN-T5 (Fine-tuned Language Net - Text-to-Text Transfer Transformer) là một mô hình Transformer encoder-decoder được phát triển bởi Google, được xây dựng dựa trên kiến trúc T5 [4]. FLAN-T5 có thể được xem như một đầu bếp lành nghề không chỉ giỏi tạo ra nhiều loại món ăn mà còn dễ dàng thích nghi với các công thức mới [4].

### 3.2. Kiến Trúc T5 và Text-to-Text Framework

Mô hình T5 gốc chuyển đổi tất cả các tác vụ NLP thành định dạng text-to-text thống nhất, trong đó đầu vào và đầu ra được xử lý như các chuỗi văn bản [4]. Điều này bao gồm mọi thứ từ dịch thuật, tóm tắt đến trả lời câu hỏi.

### 3.3. Instruction Tuning

FLAN-T5 nâng cao T5 bằng kỹ thuật called instruction tuning. Thay vì huấn luyện trên các tập dữ liệu theo định dạng tác vụ cụ thể, FLAN-T5 sử dụng một tập hợp đa dạng các prompts hoặc hướng dẫn trong giai đoạn huấn luyện [4][5].

Phương pháp này huấn luyện mô hình hiểu và tạo phản hồi tốt hơn dựa trên các hướng dẫn ngôn ngữ tự nhiên, mở rộng khả năng xử lý các tác vụ mà mô hình không được huấn luyện rõ ràng [4].

#### 3.3.1. FLAN Collection

FLAN Collection ( Flan 2022) kết hợp các tập hợp hướng dẫn phổ biến trước đó bao gồm Flan 2021, T0++, Super-Natural Instructions, cùng với một số bổ sung mới về suy luận và đối thoại [5]. Kết quả cho thấy FLAN-T5 vượt trội so với T5 trong fine-tuning tác vụ đơn lẻ và yêu cầu ít fine-tuning hơn để hội tụ [5].

### 3.4. Sử Dụng FLAN-T5 trong Thực Tế

Để sử dụng FLAN-T5, người dùng chỉ cần đóng khung tác vụ như một hướng dẫn ngôn ngữ tự nhiên [4]. Ví dụ:
- Để tóm tắt văn bản: "Tóm tắt bài viết sau đây."
- Để dịch thuật: "Dịch văn bản sau từ tiếng Anh sang tiếng Pháp."

Sự linh hoạt này làm cho FLAN-T5 trở nên cực kỳ mạnh mẽ trong các ứng dụng thực tế nơi các tác vụ có thể khác nhau đáng kể [4].

---

## 4. Transfer Learning và Fine-Tuning

### 4.1. Khái Niệm Transfer Learning

Transfer learning trong AI liên quan đến việc lấy một mô hình đã được pre-train trên một tập dữ liệu lớn và thích nghi nó cho một tác vụ chuyên biệt với các sửa đổi nhỏ [6]. Điều này thường được thực hiện bằng cách thêm một thành phần hoặc head mới vào mô hình được huấn luyện cụ thể trên tác vụ mới, trong khi giữ nguyên phần lớn cấu trúc của mô hình gốc [6].

Ví dụ, một mô hình ngôn ngữ pre-trained có thể được thêm một lớp output mới để phân loại cảm xúc email, trong đó chỉ lớp mới này học từ các email, trong khi phần còn lại của mô hình giữ nguyên [6].

### 4.2. Fine-Tuning

Fine-tuning liên quan đến việc điều chỉnh toàn bộ mô hình và tập dữ liệu mới [6]. Ở đây, tất cả các weights và biases trong mô hình được cập nhật thông qua một giai đoạn huấn luyện tiếp theo. Cách tiếp cận này đòi hỏi nhiều tài nguyên tính toán hơn, nhưng là cần thiết khi một tác vụ mới khác biệt đáng kể so với các tác vụ mà mô hình được huấn luyện ban đầu [6].

So sánh:
- **Transfer Learning**: Giống như một khóa học cập nhật nhanh cho đầu bếp
- **Fine-tuning**: Giống như theo học toàn bộ chương trình ẩm thực

### 4.3. Khi Nào Sử Dụng Phương Pháp Nào

Việc lựa chọn giữa transfer learning và fine-tuning phụ thuộc vào nhu cầu cụ thể [6]:

| Tiêu chí | Transfer Learning | Fine-tuning |
|----------|-------------------|-------------|
| Độ tương đồng tác vụ | Cao | Thấp |
| Dữ liệu cần thiết | Ít | Nhiều |
| Tài nguyên tính toán | Thấp | Cao |
| Thời gian huấn luyện | Nhanh | Chậm |
| Độ chính xác | Tốt cho tác vụ tương tự | Tối ưu cho tác vụ khác biệt |

Transfer learning lý tưởng khi các tác vụ tương tự đủ và tài nguyên hạn chế, vì nó cho phép thích nghi nhanh hơn với ít dữ liệu hơn [6]. Fine-tuning tốt nhất khi các tác vụ khác biệt rất nhiều hoặc khi độ chính xác tối đa là quan trọng, mặc dù chi phí cao hơn và thời gian dài hơn [6].

---

## 5. Parameter-Efficient Fine-Tuning (PEFT)

### 5.1. Giới Thiệu về PEFT

Parameter-Efficient Fine-Tuning (PEFT) là một nhóm các kỹ thuật nhằm giảm thiểu số lượng tham số cần huấn luyện trong quá trình fine-tuning mô hình ngôn ngữ lớn [7]. PEFT tập trung vào việc điều chỉnh một tập hợp nhỏ các tham số của mô hình thay vì toàn bộ mô hình [7].

Hãy tưởng tượng bạn là một đầu bếp làm việc với nguyên liệu hạn chế. Bạn cần tạo ra một món ăn gourmet mà không có quyền truy cập vào đầy đủ các nguyên liệu. Đây là thách thức tương tự trong machine learning khi dữ liệu ít [7].

### 5.2. Sự Khác Biệt Giữa PEFT, Transfer Learning và Fine-Tuning

| Phương pháp | Mô tả | Tài nguyên cần thiết |
|-------------|-------|---------------------|
| **Traditional Fine-tuning** | Điều chỉnh tất cả các tham số của mô hình | Rất cao |
| **Transfer Learning** | Thêm các lớp mới vào mô hình pre-trained | Trung bình |
| **PEFT** | Thêm các adapters nhỏ, chỉ huấn luyện adapters | Thấp |

Fine-tuning truyền thống có thể đòi hỏi nhiều tài nguyên, giống như có một nhà bếp được trang bị đầy đủ [7]. PEFT, ngược lại, giống như nghệ thuật nấu nướng với những gì bạn có, tối ưu hóa việc sử dụng mỗi nguyên liệu [7].

### 5.3. Adapters trong PEFT

Adapters là các module nhẹ được chèn vào mô hình pre-trained [7]. Trong quá trình huấn luyện, chỉ các adapters này được cập nhật trong khi phần còn lại của mô hình giữ nguyên. Phương pháp này giảm đáng kể tài nguyên tính toán cần thiết và làm cho quá trình huấn luyện nhanh và hiệu quả hơn [7].

Ví dụ, nếu bạn huấn luyện mô hình ngôn ngữ để hiểu tài liệu pháp lý, bạn có thể chèn các adapters chuyên biệt về thuật ngữ pháp lý và ngữ cảnh [7]. Các adapters này được huấn luyện với tập dữ liệu hạn chế của bạn, thích nghi mô hình để thực hiện tốt trên tác vụ cụ thể này mà không cần huấn luyện lại toàn bộ mô hình [7].

### 5.4. Tại Sao PEFT Quan Trọng Khi Dữ Liệu Hạn Chế

PEFT quan trọng vì nó về hiệu suất. Với PEFT, bạn có thể đạt được hiệu suất cao với ít điểm dữ liệu hơn và ít sức mạnh tính toán hơn [7]. Điều này đặc biệt có lợi trong các kịch bản nơi việc thu thập lượng lớn dữ liệu gắn nhãn không thực tế hoặc quá tốn kém [7].

---

## 6. Low-Rank Adaptation (LoRA)

### 6.1. Giới Thiệu về LoRA

Low-Rank Adaptation (LoRA) là một phương pháp PEFT mạnh mẽ, được đề xuất bởi Hu et al. trong bài báo "LoRA: Low-Rank Adaptation of Large Language Models" [8]. LoRA đông cứng các trọng số pre-trained và tiêm các ma trận phân rã hạng thấp (rank decomposition matrices) có thể huấn luyện vào mỗi lớp của kiến trúc Transformer [8].

Hãy tưởng tượng bạn có một công thức tuyệt vời. Bạn muốn cải thiện món ăn mà không cần thay đổi toàn bộ quy trình nấu nướng. Bạn mang đến một công cụ chuyên biệt như một microplane để bào vỏ chanh. Công cụ này tạo ra tác động lớn với nỗ lực tối thiểu [9].

### 6.2. Cơ Sở Toán Học của LoRA

Trong một lớp neural network điển hình, trọng số được biểu diễn bởi một ma trận lớn W với kích thước $d \times d$ [9]. Trong fine-tuning truyền thống, ma trận này được điều chỉnh để cải thiện hiệu suất mô hình. Tuy nhiên, quá trình này có thể tốn kém về tính toán và đòi hỏi nhiều dữ liệu.

LoRA đề xuất sử dụng phân rã hạng thấp:

$$W' = W + \Delta W = W + BA$$

Trong đó:
- $W \in \mathbb{R}^{d \times d}$: Ma trận trọng số pre-trained (đông cứng)
- $B \in \mathbb{R}^{d \times r}$: Ma trận hạng thấp thứ nhất
- $A \in \mathbb{R}^{r \times d}$: Ma trận hạng thấp thứ hai
- $r \ll d$: Hạng (rank) của ma trận thích nghi

#### 6.2.1. Số Lượng Tham Số Cần Huấn Luyện

Với kích thước ma trận gốc $n = 512$ và rank $r = 1$:
- Số tham số cần fine-tune trong LoRA: $512 \times 1 \times 2 = 1,024$ tham số
- Số tham số trong ma trận gốc: $512^2 = 262,144$ tham số
- Giảm khoảng 256 lần [9]

Nếu sử dụng floating-point 32 precision: $1,024 \times 32 = 32,768$ bytes thay vì hơn 2 triệu [9].

### 6.3. Lợi Ích của LoRA

So với GPT-3 175B fine-tuned với Adam, LoRA có thể giảm số lượng tham số có thể huấn luyện xuống 10,000 lần và yêu cầu bộ nhớ GPU xuống 3 lần [8]. LoRA thực hiện tương đương hoặc tốt hơn so với fine-tuning về chất lượng mô hình trên RoBERTa, DeBERTa, GPT-2, và GPT-3, mặc dù có ít tham số hơn, throughput huấn luyện cao hơn, và không có overhead inference [8].

### 6.4. Các Thách Thức Khi Triển Khai LoRA

#### 6.4.1. Overfitting vs Generalizability

Overfitting xảy ra khi mô hình học quá tốt dữ liệu huấn luyện, nắm bắt nhiễu và chi tiết không khái quát hóa sang dữ liệu mới chưa thấy [10]. Nó giống như một món ăn được điều chỉnh theo khẩu vị của một số người cụ thể nhưng không hấp dẫn khán giả rộng hơn [10].

Generalizability là về việc đảm bảo mô hình hoạt động tốt trên dữ liệu mới, tương tự như tạo ra một món ăn làm hài lòng nhiều loại khẩu vị khác nhau [10].

#### 6.4.2. Lựa Chọn Rank

Việc chọn rank phù hợp cho LoRA adapters rất quan trọng [10]. Sử dụng Microplane để bào vỏ là hoàn hảo, nhưng dùng nó để rửa phô mai sẽ không hiệu quả [10]. Tương tự, rank xác định có bao nhiêu tham số được đưa vào và điều chỉnh.

- **Rank thấp hơn**: Ít tham số hơn, giúp ngăn overfitting, nhưng có thể giới hạn khả năng học các pattern phức tạp
- **Rank cao hơn**: Nhiều tham số hơn, tăng khả năng học, nhưng tăng nguy cơ overfitting [10]

Lời khuyên thực tế: Bắt đầu với rank thấp và tăng dần trong khi theo dõi hiệu suất mô hình và dữ liệu validation [10].

#### 6.4.3. Điều Chỉnh Tham Số

Điều chỉnh tham số trong LoRA giống như nêm gia vị món ăn [10]. Bạn cần tìm lượng phù hợp của mỗi nguyên liệu để làm cho món ăn hoàn hảo. Điều này liên quan đến việc điều chỉnh learning rate, batch size, và số epoch để tối ưu hóa việc huấn luyện mô hình [10].

- **Learning rate**: Kiểm soát mức độ điều chỉnh các tham số mô hình trong quá trình huấn luyện. Quá cao có thể khiến mô hình hội tụ quá nhanh đến giải pháp không tối ưu; quá thấp có thể làm quá trình huấn luyện rất chậm [10]
- **Batch size**: Batch lớn có thể ổn định huấn luyện nhưng đòi hỏi nhiều bộ nhớ hơn [10]
- **Số epoch**: Đủ để đảm bảo mô hình học nhưng không quá nhiều để overfitting [10]

### 6.5. Các Biến Thể của LoRA

Nhiều biến thể của LoRA đã được đề xuất để cải thiện hiệu suất:

- **LoRA+**: Cải thiện hiệu suất 1-2% bằng cách sử dụng learning rate khác nhau cho các ma trận A và B [11]
- **QLoRA**: Phiên bản lượng tử hóa của LoRA, giảm chi phí tính toán thêm bằng cách lượng tử hóa trọng số pre-trained xuống 4 bit [11]
- **AdaLoRA**: Cắt tỉa động các tham số không quan trọng
- **ScaLoRA**: Tích hợp progressive high-rank weight update từ các incremental low-rank [12]
- **NB-LoRA**: Parameterization mới cho phép explicit bounds trên mỗi singular value của ma trận adaptation [13]

---

## 7. Kết Luận

### 7.1. Tổng Kết

Trong bài viết này, chúng tôi đã trình bày tổng quan toàn diện về các phương pháp fine-tuning Large Language Models, từ kiến trúc Transformer nền tảng đến các kỹ thuật tối ưu như PEFT và LoRA. Các điểm chính bao gồm:

1. **Transformer Architecture**: Nền tảng của LLMs hiện đại, sử dụng cơ chế self-attention để xử lý ngôn ngữ hiệu quả [2][3]

2. **FLAN-T5**: Mô hình tiên phong sử dụng instruction tuning, cho phép hiểu và thực thi nhiều loại hướng dẫn ngôn ngữ tự nhiên [4][5]

3. **Transfer Learning vs Fine-tuning**: Transfer learning phù hợp khi tác vụ tương tự và tài nguyên hạn chế; fine-tuning tối ưu khi tác vụ khác biệt và cần độ chính xác cao [6]

4. **PEFT**: Giải pháp hiệu quả khi dữ liệu hạn chế, sử dụng adapters nhẹ để thích nghi mô hình [7]

5. **LoRA**: Phương pháp PEFT phổ biến nhất, sử dụng ma trận hạng thấp để giảm đáng kể số tham số cần huấn luyện (lên đến 10,000 lần) trong khi vẫn duy trì hiệu suất tương đương hoặc tốt hơn fine-tuning truyền thống [8][9][10]

### 7.2. Hướng Nghiên Cứu Tương Lai

Các hướng nghiên cứu tiếp theo bao gồm:
- Phát triển các phương pháp chọn rank tự động cho LoRA
- Nghiên cứu về initialization strategies tốt hơn cho ma trận adapters
- Kết hợp LoRA với các kỹ thuật quantization để giảm thêm tài nguyên
- Ứng dụng PEFT vào các mô hình đa phương thức

---

## Tài Liệu Tham Khảo

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** *Attention Is All You Need.* Advances in Neural Information Processing Systems, 30, 5998-6008. https://arxiv.org/abs/1706.03762

2. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv preprint arXiv:1810.04805. https://arxiv.org/abs/1810.04805

3. **Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2019).** *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* arXiv preprint arXiv:1910.10683. https://arxiv.org/abs/1910.10683

4. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).** *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv preprint arXiv:2106.09685. https://arxiv.org/abs/2106.09685

5. **Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., ... & Wei, J. (2022).** *Scaling Instruction-Finetuned Language Models.* arXiv preprint arXiv:2210.11416. https://arxiv.org/abs/2210.11416

6. **Longpre, S., Hou, L., Vu, T., Webson, A., Chung, H. W., Tay, Y., Zhou, D., Le, Q. V., Zoph, B., Wei, J., & Roberts, A. (2023).** *The Flan Collection: Designing Data and Methods for Effective Instruction Tuning.* arXiv preprint arXiv:2301.13688. https://arxiv.org/abs/2301.13688

7. **Han, Z., Gao, C., Liu, J., Zhang, J., & Zhang, S. Q. (2024).** *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* arXiv preprint arXiv:2403.14608. https://arxiv.org/abs/2403.14608

8. **Weidinger, L., Mellor, J., Rauh, M., Griffin, C., Uesato, J., Huang, P. S., Cheng, M., Glaese, M., Balle, B., Kasirzadeh, A., Kenton, Z., Brown, S., Hawkins, W., Stepleton, T., Biles, C., Birhane, A., Haas, J., Rimell, L., Hendricks, L. A., ... & Gabriel, I. (2021).** *Ethical and Social Risks of Harm from Language Models.* DeepMind. https://storage.googleapis.com/deepmind-media/research/language-research/Ethical%20and%20social%20risks.pdf

9. **Bengio, Y., Mindermann, S., Privitera, D., Besiroglu, T., Bommasani, R., Casper, S., Choi, Y., Goldfarb, D., Heidari, H., Khalatbari, L., Longpre, S., Mavroudis, V., Mazeika, M., Ng, K. Y., Okolo, C. T., Raji, D., Skeadas, T., Tramèr, F., Adekanmbi, B., ... & Zhou, D. (2024).** *International Scientific Report on the Safety of Advanced AI (Interim Report).* arXiv preprint arXiv:2412.05282. https://arxiv.org/abs/2412.05282

10. **Amodei, D., Ananthanarayanan, S., Bapna, R., Chen, Z., Du, E., Goodfellow, I., ... & Sutskever, I. (2016).** *Concrete Problems in AI Safety.* arXiv preprint arXiv:1606.06565. https://arxiv.org/abs/1606.06565

11. **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2024).** *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv preprint arXiv:2305.14314. https://arxiv.org/abs/2305.14314

12. **Zhang, Y., Yang, X., Cai, Y., & Giannakis, G. B. (2025).** *ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning.* arXiv preprint arXiv:2510.23818. https://arxiv.org/abs/2510.23818

13. **Wang, R., Dvijotham, K. D., & Manchester, I. R. (2025).** *Norm-Bounded Low-Rank Adaptation.* arXiv preprint arXiv:2501.19050. https://arxiv.org/abs/2501.19050

14. **Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T. A., Bernhard, M., ... & Houlsby, N. (2022).** *LoRA+: Efficient Low Rank Adaptation of Large Models.* arXiv preprint arXiv:2402.12354. https://arxiv.org/abs/2402.12354

15. **Laakso, A., Kemell, K. K., & Nurminen, J. K. (2024).** *Ethical Issues in Large Language Models: A Systematic Literature Review.* CEUR Workshop Proceedings, 3901. https://ceur-ws.org/Vol-3901/paper_4.pdf

16. **Bosma, M., & Wei, J. (2021).** *Introducing FLAN: More Generalizable Language Models with Instruction Fine-Tuning.* Google AI Blog. https://research.google/blog/introducing-flan-more-generalizable-language-models-with-instruction-fine-tuning/

17. **Roberts, A., & Raffel, C. (2020).** *Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer.* Google AI Blog. https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/

18. **Lester, B., Al-Rfou, R., & Wang, L. (2021).** *The Power of Scale for Parameter-Efficient Prompt Tuning.* Proceedings of EMNLP 2021. https://arxiv.org/abs/2104.08691

19. **Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020).** *Language Models are Few-Shot Learners.* Advances in Neural Information Processing Systems, 33, 1877-1901. https://arxiv.org/abs/2005.14165

---

*Bài viết được viết dựa trên tài liệu khóa học Fine-Tuning Large Language Models và các bài báo khoa học liên quan.*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[Nghiên Cứu Về Fine-Tuning Large Language Models: Từ Transformer Đến LoRA](fine_tuning_scientific_article.md)** | [Xem bài viết →](fine_tuning_scientific_article.md) |
| [Tài liệu tham khảo](references.md) | [Xem bài viết →](references.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
