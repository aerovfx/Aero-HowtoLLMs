# Tổng Kết Khóa Học: Advanced RAG với Cơ Sở Dữ Liệu Vector và Retrievers

## Chúc Mừng

Xin chúc mừng, bạn đã đến cuối khóa học này. Hãy dành một chút thời gian để suy ngẫm về hành trình của bạn, nơi bạn đã có được những kiến thức cần thiết về việc sử dụng cơ sở dữ liệu vector và các retriever nâng cao trong việc phát triển các ứng dụng Retrieval-Augmented Generation (RAG).

## Tổng Quan Những Gì Đã Học

### Retriever trong LangChain

LangChain retriever là một interface trả về tài liệu dựa trên truy vấn không có cấu trúc. Mục đích của nó không nhất thiết là lưu trữ tài liệu mà là truy xuất chúng hoặc các đoạn (chunks) của chúng. Một LangChain retriever chấp truy vấn dưới dạng chuỗi làm đầu vào và trả về danh sách các tài liệu hoặc đoạn làm đầu ra.

**Vector Store-Based Retriever**: Loại retriever đơn giản nhất, truy xuất tài liệu từ cơ sở dữ liệu vector. Có thể được tạo trực tiếp từ đối tượng vector store bằng phương thức retriever sử dụng similarity search hoặc maximum marginal relevance (MMR).

**Multi-Query Retriever**: Tương tự như vector store-based retriever, ngoại trừ việc sử dụng LLM để tạo các phiên bản khác nhau của truy vấn, tạo ra tập hợp tài liệu được truy xuất phong phú hơn.

**Self-Query Retriever**: Chuyển đổi truy vấn thành hai thành phần: một chuỗi để tra cứu ngữ nghĩa và một bộ lọc metadata đi kèm.

**Parent Document Retriever**: Có hai bộ chia văn bản: parent splitter chia văn bản thành các chunk lớn để truy xuất và child splitter chia tài liệu thành các chunk nhỏ để tạo embedding có ý nghĩa. Trong quá trình truy xuất, parent document retriever đầu tiên fetch các chunk nhỏ, tra cứu ID cha của chúng và trả về tài liệu lớn hơn mà các chunk nhỏ tồn tại.

### Index và Retriever trong LlamaIndex

**VectorStoreIndex**: Lưu trữ vector embedding cho mỗi chunk, phù hợp nhất cho truy xuất ngữ nghĩa và thường được sử dụng trong các pipeline liên quan đến large language models.

**DocumentSummaryIndex**: Tạo và lưu trữ tóm tắt của tài liệu. Các tóm tắt này được sử dụng để lọc tài liệu trước khi truy xuất nội dung đầy đủ. Loại index này hữu ích khi làm việc với các tập tài liệu lớn và đa dạng.

**KeywordTableIndex**: Trích xuất từ khóa từ tài liệu và ánh xạ chúng đến các đoạn nội dung cụ thể, hữu ích trong các kịch bản tìm kiếm lai hoặc dựa trên quy tắc.

**Vector Index Retriever**: Sử dụng vector embedding để tìm nội dung liên quan về mặt ngữ nghĩa, lý tưởng cho tìm kiếm mục đích chung và pipeline RAG.

**BM25 Retriever**: Phương pháp dựa trên từ khóa để xếp hạng tài liệu. Nó truy xuất nội dung dựa trên khớp từ khóa chính xác thay vì tương đồng ngữ nghĩa.

**Document Summary Index Retriever**: Sử dụng tóm tắt tài liệu thay vì tài liệu thực để tìm nội dung liên quan bằng cách sử dụng LLM hoặc tương đồng ngữ nghĩa.

**Auto Merging Retriever**: Bảo preservation ngữ cảnh trong các tài liệu dài bằng cách sử dụng cấu trúc phân cấp và hierarchical chunking để chia tài liệu thành các node cha và node con.

**Recursive Retriever**: Theo dõi các mối quan hệ giữa các node và sử dụng các tham chiếu như trích dẫn trong bài báo khoa học hoặc liên kết metadata.

**Query Fusion Retriever**: Kết hợp kết quả từ các retriever khác nhau bằng các chiến lược fusion.

### Các Chiến Lược Fusion

**Reciprocal Rank Fusion**: Kết hợp các danh sách đã xếp hạng bằng cách gán điểm cao hơn cho các tài liệu xuất hiện ở đầu bất kỳ danh sách nào.

**Relative Score Fusion**: Chuẩn hóa các điểm trong mỗi tập kết quả bằng cách chia cho điểm tối đa.

**Distribution-Based Fusion**: Sử dụng các kỹ thuật thống kê như z-score normalization hoặc percentile ranking để kết hợp kết quả.

### FAISS và Chroma DB

**FAISS (Facebook AI Similarity Search)**: Thư viện do Meta tạo ra để tìm kiếm vector nhanh. Nó lý tưởng khi bạn muốn toàn quyền kiểm soát và hiệu suất cao. FAISS là một thư viện, trong khi Chroma DB là một cơ sở dữ liệu đầy đủ. FAISS được thiết kế cho hoạt động trên một node duy nhất và không cung cấp khả năng mở rộng phân tán gốc. FAISS cung cấp nhiều tùy chọn indexing hơn, trong khi Chroma DB chỉ hỗ trợ HNSW. FAISS không hỗ trợ metadata một cách gốc, trong khi Chroma DB hỗ trợ lưu trữ metadata và lọc dựa trên các thẻ metadata. Cả FAISS và Chroma DB đều hoạt động với LangChain và LlamaIndex.

**Chroma DB**: Cơ sở dữ liệu vector được xây dựng cho các trường hợp sử dụng AI. Nó lưu trữ cả vector và metadata như thẻ hoặc mô tả. Chroma DB hỗ trợ cả triển khai trên một node và triển khai phân tán cho các workload lớn.

### Các Loại Index trong FAISS

**Flat Index**: So sánh khoảng cách (sử dụng khoảng cách Euclidean hoặc dot product) giữa embedding truy vấn và embedding của mọi vector trong vector store bằng cách sử dụng tìm kiếm brute force. Sau đó truy xuất 'k-nearest vectors' được sắp xếp từ gần nhất đến xa nhất.

**Inverted File Index (IVF Index)**: Tăng tốc tìm kiếm vector bằng cách nhóm các vector bằng các phương pháp như k-means, hình thành các ô Voronoi xung quanh các centroid. Mỗi ô chứa các vector gần nhất với centroid của nó.

**Locality-Sensitive Hashing (LSH)**: Sử dụng các hàm băm ánh xạ các vector tương tự đến cùng một bucket. LSH tìm kiếm các vector trong các nhóm khớp gần nhất, cho phép tìm kiếm nhanh và hiệu quả bộ nhớ. Nó đặc biệt hữu ích cho dữ liệu thưa thớt chiều cao như text embeddings.

**Hierarchical Navigable Small World (HNSW)**: Tổ chức các vector thành một hệ thống phân cấp các lớp. Các lớp trên cùng thưa thớt và chỉ chứa một số ít vector. Các lớp dưới cung cấp các kết nối cục bộ chi tiết hơn, cho phép thuật toán tinh chỉnh tìm kiếm của nó. Tìm kiếm bắt đầu ở lớp trên cùng và di chuyển xuống dưới, sử dụng ứng viên tốt nhất từ mỗi lớp làm điểm vào cho lớp tiếp theo. Cách tiếp cận phân cấp này làm cho HNSW vừa nhanh vừa chính xác, đặc biệt cho các bộ dữ liệu lớn.

## Bước Tiếp Theo

Bây giờ bạn đã xem lại một số ý tưởng cơ bản được trình bày trong khóa học này, hãy nhớ các glossary và cheat sheets của mỗi module. Bạn có thể sử dụng các tài sản này để nhanh chóng tham khảo nhiều điều bạn đã học.

Nếu bạn chưa đăng ký Chương trình Chứng chỉ Chuyên nghiệp RAG và AI Tác nhân của IBM, mà khóa học này là một phần, chúng tôi khuyến khích bạn làm điều đó. Tùy thuộc vào lịch trình và số lượng khóa học trong chương trình, chương trình này sẽ mất khoảng 1-2 tháng để hoàn thành. Mỗi chương trình chứa nhiều bài lab thực hành và một dự án cuối cùng. Các Chương trình Chứng chỉ Chuyên nghiệp cũng có một khóa học capstone nơi bạn tổng hợp và trình bày tất cả các kỹ năng bạn đã học trong suốt chương trình.

## Lời Kết

Xin chúc mừng bạn đã hoàn thành khóa học này và cảm ơn bạn đã trở thành một phần của hành trình học tập này. Như một bước tiếp theo, chúng tôi khuyên bạn tiếp tục hành trình học tập của mình và tiếp tục áp dụng các kỹ năng mới của bạn. Chúc bạn mọi điều tốt đẹp nhất!

---

## Tài Liệu Tham Khảo

1. LangChain Documentation. (2025). *Retrievers*. https://python.langchain.com/docs/modules/data_connection/retrievers/
2. LlamaIndex Documentation. (2025). *Index and Retriever Types*. https://docs.llamaindex.ai/
3. Meta AI. (2025). *FAISS: Facebook AI Similarity Search*. https://faiss.ai/
4. Chromadb. (2025). *Chroma: The AI-native open-source embedding database*. https://www.trychroma.com/
5. IBM. (2025). *Advanced RAG with Vector Databases and Retrievers*. Coursera.
6. Robertson, S., & Zaragoza, H. (2009). *The probabilistic relevance framework: BM25 and beyond*. Foundations and Trends in Information Retrieval, 3(4), 333-389.
