# Các Retriever Nâng Cao trong LLAMAIndex

## Giới Thiệu

Chào mừng bạn đến với bài học về các retriever nâng cao trong LLAMAIndex. Trong bài học này, chúng ta sẽ tìm hiểu về các loại index khác nhau trong LLAMAIndex, các retriever cốt lõi và nâng cao, các kỹ thuật fusion kết hợp kết quả từ nhiều truy vấn, cũng như các trường hợp sử dụng phù hợp nhất cho từng loại retriever. LLAMAIndex cung cấp một bộ công cụ phong phú để xây dựng các pipeline truy xuất thông minh và linh hoạt.

## Các Loại Index Cốt Lõi trong LLAMAIndex

LLAMAIndex cung cấp ba loại index cốt lõi, mỗi loại phù hợp với các mục đích sử dụng khác nhau.

### VectorStoreIndex

VectorStoreIndex được sử dụng cho tìm kiếm ngữ nghĩa dựa trên ý nghĩa của văn bản. Index này lưu trữ các vector embedding cho mỗi đoạn tài liệu (document chunk). Đây là loại index phổ biến nhất trong các ứng dụng liên quan đến Large Language Models (LLM) và đặc biệt phù hợp cho việc truy xuất ngữ nghĩa (semantic retrieval).

### DocumentSummaryIndex

DocumentSummaryIndex tạo và lưu trữ các tóm tắt của tài liệu tại thời điểm indexing. Các tóm tắt này được sử dụng để lọc tài liệu trước khi truy xuất nội dung đầy đủ. Loại index này đặc biệt hữu ích khi làm việc với các tập hợp tài liệu lớn và đa dạng không thể chứa trong context window của LLM. Bằng cách sử dụng tóm tắt để lọc trước, hệ thống có thể giảm đáng kể lượng tài liệu cần xử lý.

### KeywordTableIndex

KeywordTableIndex trích xuất các từ khóa từ tài liệu và ánh xạ các từ khóa đó đến các đoạn nội dung cụ thể. Index này lý tưởng cho việc khớp từ khóa chính xác (exact keyword matching) và các kịch bản tìm kiếm lai (hybrid) hoặc dựa trên quy tắc (rule-based).

## Các Loại Retriever trong LLAMAIndex

### Vector Index Retriever

Vector Index Retriever sử dụng vector embedding để tìm nội dung liên quan về mặt ngữ nghĩa. Đây là loại retriever phổ biến nhất cho tìm kiếm mục đích chung và được sửng rộng rãi trong các pipeline Retrieval Augmented Generation (RAG).

### BM25 Retriever

BM25 Retriever là một phương pháp dựa trên từ khóa để xếp hạng tài liệu. Nó truy xuất nội dung dựa trên khớp từ khóa chính xác (exact keyword match) thay vì tương đồng ngữ nghĩa. BM25 cải thiện TF-IDF bằng cách giải quyết một số hạn chế của nó.

#### Tìm Hiểu TF-IDF

TF-IDF là nền tảng của tìm kiếm dựa trên từ khóa:

- **Term Frequency (TF)**: Đo lường tần suất một từ xuất hiện trong tài liệu.
- **Inverse Document Frequency (IDF)**: Đo lường độ hiếm của từ đó trong tất cả các tài liệu.
- **Điểm TF-IDF**: Là tích của hai giá trị này, làm nổi bật các từ thường xuất hiện trong một tài liệu nhưng hiếm trong toàn bộ bộ sưu tập.

#### Cải Tiến của BM25

BM25 cải thiện TF-IDF bằng hai cách chính:

1. **Term Frequency Saturation**: Giảm tác động của các từ lặp lại nhiều lần, tránh tình trạng một từ xuất hiện quá nhiều lần chiếm ưu thế.
2. **Document Length Normalization**: Điều chỉnh cho độ dài tài liệu, làm cho phương pháp hiệu quả hơn cho tìm kiếm dựa trên từ khóa.

### Document Summary Index Retriever

Loại retriever này sử dụng tóm tắt tài liệu thay vì tài liệu thực để tìm nội dung liên quan. Có hai phiên bản:

1. **Phiên bản sử dụng LLM**: Sử dụng LLM để tìm nội dung phù hợp nhất, nhưng có thể tốn thời gian và chi phí hơn.
2. **Phiên bản sử dụng tương đồng ngữ nghĩa**: Sử dụng tương đồng ngữ nghĩa giữa embedding của truy vấn và tóm tắt, hiệu quả hơn cho các bộ sưu tập lớn.

Dù sử dụng phiên bản nào, Document Summary Index Retriever luôn trả về tài liệu gốc, không phải tóm tắt của chúng.

### Auto Merging Retriever

Auto Merging Retriever được thiết kế để bảo preservation ngữ cảnh trong các tài liệu dài bằng cách sử dụng cấu trúc phân cấp. Nó sử dụng hierarchical chunking để chia tài liệu thành các node cha và node con. Nếu có đủ các node con từ cùng một node cha được truy xuất, retriever sẽ trả về node cha thay vì các node con. Điều này giúp konsolidieren nội dung liên quan và bảo preservation ngữ cảnh rộng hơn.

### Recursive Retriever

Recursive Retriever được thiết kế để theo dõi các mối quan hệ giữa các node thông qua các tham chiếu. Nó có thể theo dõi các tham chiếu từ một node đến node khác, như các trích dẫn trong bài báo khoa học hoặc các liên kết metadata. Recursive Retriever hỗ trợ cả tham chiếu chunk và tham chiếu metadata, cho phép truy xuất nội dung liên quan xuyên suốt các tài liệu hoặc các lớp trừu tượng khác nhau.

### Query Fusion Retriever

Query Fusion Retriever được sử dụng để kết hợp kết quả từ các retriever khác nhau, như các phương pháp dựa trên vector và dựa trên từ khóa. Nó cũng tùy chọn tạo nhiều biến thể của một truy vấn bằng cách sử dụng LLM để cải thiện độ bao phủ. Kết quả được hợp nhất bằng các chiến lược fusion như reciprocal rank fusion hoặc relative score fusion để cải thiện khả năng recall.

## Các Chiến Lược Fusion

LLAMAIndex's Query Fusion Retriever hỗ trợ một số chiến lược fusion:

### Reciprocal Rank Fusion

Reciprocal Rank Fusion kết hợp các danh sách đã xếp hạng bằng cách gán điểm cao hơn cho các tài liệu xuất hiện ở đầu bất kỳ danh sách nào. Phương pháp này mạnh mẽ và không phụ thuộc vào độ lớn của điểm số.

### Relative Score Fusion

Relative Score Fusion chuẩn hóa các điểm trong mỗi tập kết quả bằng cách chia cho điểm tối đa. Điều này bảo preservation sự tự tin tương đối của mỗi retriever.

### Distribution-Based Fusion

Distribution-Based Fusion sử dụng các kỹ thuật thống kê như z-score normalization hoặc percentile ranking để kết hợp kết quả, làm cho nó lý tưởng để xử lý sự biến thiên của điểm số.

## Khuyến Nghị Sử Dụng Theo Trường Hợp

### Hỏi Đáp Chung (General Q&A)

Nên sử dụng Vector Index Retriever, có thể kết hợp với BM25 Retriever. Sự kết hợp này kết hợp sự liên quan ngữ nghĩa với việc khớp từ khóa.

### Tài Liệu Kỹ Thuật

Đặc biệt cho các tài liệu kỹ thuật nơi các thuật ngữ chính xác cần được ưu tiên, hãy cân nhắc sử dụng BM25 Retriever làm retriever chính, với Vector Index Retriever bổ sung linh hoạt ngữ cảnh như retriever thứ cấp.

### Tài Liệu Dài

Auto Merging Retriever là một lựa chọn tuyệt vời vì nó sẽ truy xuất các phiên bản cha dài hơn chỉ khi đủ các phiên bản con ngắn hơn được truy xuất.

### Bài Báo Nghiên Cứu

Sử dụng Recursive Retriever để truy xuất nội dung liên quan từ các bài báo được trích dẫn.

### Bộ Tài Liệu Lớn

Cân nhắc sử dụng Document Summary Index Retriever để thu hẹp số lượng tài liệu liên quan, sau đó sử dụng Vector Search trong tập hợp con còn lại để truy xuất nội dung phù hợp nhất.

## Tổng Kết

Trong bài học này, chúng ta đã tìm hiểu:

1. **Các loại index cốt lõi của LLAMAIndex**:
   - VectorStoreIndex: Lưu trữ vector embedding, phù hợp cho truy xuất ngữ nghĩa
   - DocumentSummaryIndex: Tạo và lưu tóm tắt tài liệu, hữu ích cho tập tài liệu lớn
   - KeywordTableIndex: Trích xuất từ khóa, phù hợp cho tìm kiếm lai

2. **Các loại retriever**:
   - VectorIndexRetriever: Tìm kiếm ngữ nghĩa
   - BM25 Retriever: Tìm kiếm dựa trên từ khóa
   - Document Summary Index Retriever: Sử dụng tóm tắt
   - Auto Merging Retriever: Bảo preservation ngữ cảnh phân cấp
   - Recursive Retriever: Theo dõi mối quan hệ giữa các node
   - Query Fusion Retriever: Kết hợp kết quả từ nhiều retriever

3. **Các chiến lược fusion**: Reciprocal Rank Fusion, Relative Score Fusion, và Distribution-Based Fusion

Việc lựa chọn đúng loại retriever phụ thuộc vào đặc điểm của dữ liệu và yêu cầu cụ thể của ứng dụng. LLAMAIndex cung cấp sự linh hoạt để kết hợp nhiều phương pháp nhằm đạt được kết quả tốt nhất.

---

## Tài Liệu Tham Khảo

1. LlamaIndex Documentation. (2025). *Index Types*. https://docs.llamaindex.ai/api-reference/index/
2. LlamaIndex Documentation. (2025). *Retrievers*. https://docs.llamaindex.ai/api-reference/retriever/
3. LlamaIndex Documentation. (2025). *Query Fusion Retriever*. https://docs.llamaindex.ai/api-reference/retriever/query_fusion_retriever/
4. IBM. (2025). *Advanced RAG with Vector Databases and Retrievers*. Coursera.
5. Robertson, S., & Zaragoza, H. (2009). *The probabilistic relevance framework: BM25 and beyond*. Foundations and Trends in Information Retrieval, 3(4), 333-389.
