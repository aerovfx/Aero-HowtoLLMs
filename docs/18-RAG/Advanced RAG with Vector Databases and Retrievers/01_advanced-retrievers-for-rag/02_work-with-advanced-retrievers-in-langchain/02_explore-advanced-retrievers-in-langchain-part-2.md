# Khám Phá Các Retriever Nâng Cao trong LangChain - Phần 2

## Giới Thiệu

Chào mừng bạn đến với phần tiếp theo của khóa học khám phá các retriever nâng cao trong LangChain. Trong phần này, chúng ta sẽ tìm hiểu về các loại retriever khác nhau trong LangChain, bao gồm multi-query retriever, self-query retriever, và parent document retriever. Mỗi loại retriever có những ưu điểm và ứng dụng riêng trong việc truy xuất tài liệu từ cơ sở dữ liệu vector.

## Tổng Quan về LangChain Retriever

Trong LangChain, retriever là một interface cho phép truy xuất các tài liệu dựa trên truy vấn không có cấu trúc (unstructured query). Loại retriever đơn giản nhất là vector store based retriever, loại này truy xuất tài liệu từ cơ sở dữ liệu vector bằng cách so sánh độ tương đồng giữa embedding của truy vấn và embedding của các tài liệu được lưu trữ.

## Multi-Query Retriever

### Nguyên Lý Hoạt Động

Multi-query retriever là một cải tiến của vector store based retriever. Thay vì chỉ sử dụng một truy vấn duy nhất, retriever này sử dụng một Large Language Model (LLM) để tạo ra nhiều phiên bản khác nhau của cùng một truy vấn. Điều này giúp tạo ra một tập hợp phong phú hơn các tài liệu được truy xuất.

### Lý Do Sử Dụng

Có hai lý do chính khiến multi-query retriever trở nên hữu ích:

1. **Sự khác biệt trong cách diễn đạt truy vấn**: Một thay đổi nhỏ trong cách viết truy vấn có thể dẫn đến các kết quả khác nhau đáng kể.

2. **Hạn chế của embedding**: Đôi khi các embedding không nắm bắt tốt ngữ nghĩa của dữ liệu, dẫn đến việc truy xuất thiếu chính xác.

### Triển Khai

Để triển khai multi-query retriever, trước tiên cần tạo một instance LLM. Trong ví dụ này, sử dụng Watson XLLM với mô hình Mixtral 8x7B để tạo các phiên bản truy vấn khác nhau. Sau đó, đối tượng multi-query retriever được tạo bằng cách sử dụng lớp MultiQueryRetriever từ LangChain.

Lớp này chấp nhận một tham số retriever, đó là vector store based retriever được sử dụng để truy xuất kết quả cho mỗi truy vấn. Trong ví dụ, similarity search retriever đơn giản được sử dụng, nhưng các retriever khác như MMR retriever cũng có thể được sử dụng.

### Kết Quả Truy Xuất

Với mỗi truy vấn được tạo ra, multi-query retriever truy xuất một tập hợp các tài liệu liên quan. Sau đó, nó thực hiện phép hợp nhất duy nhất (unique union) giữa tất cả các truy vấn để có được một tập hợp lớn hơn các tài liệu có thể liên quan. Điều này giúp tăng độ bao quát và giảm thiểu việc bỏ sót các tài liệu quan trọng.

## Self-Query Retriever

### Vấn Đề với Metadata

Trong nhiều ứng dụng thực tế, các tài liệu không chỉ chứa văn bản thuần túy mà còn đi kèm với metadata bổ sung. Ví dụ, khi làm việc với dữ liệu phim, mỗi tài liệu có thể chứa thông tin như năm phát hành, đạo diễn, và đánh giá IMDB. Các retriever trước đó không có khả năng truy cập metadata này vì chỉ có văn bản tài liệu được xem xét.

### Giải Pháp của Self-Query Retriever

Self-query retriever giải quyết vấn đề này bằng cách chuyển đổi truy vấn thành hai thành phần:

1. **Phần ngữ nghĩa (Semantic part)**: Một chuỗi để tìm kiếm theo ngữ nghĩa trong không gian vector.
2. **Phần metadata filter**: Bộ lọc metadata đi kèm để lọc tài liệu dựa trên các thuộc tính bổ sung.

### Triển Khai

Để triển khai self-query retriever, cần thực hiện các bước sau:

1. **Tạo vector store**: Chuyển đổi các tài liệu thành vector store để có thể truy xuất tài liệu.
2. **Định nghĩa metadata**: Mô tả các trường metadata cho các tài liệu trong vector store. Ví dụ, thuộc tính year được mô tả là một số nguyên cho biết năm phát hành của bộ phim.
3. **Tạo self-query retriever**: Sử dụng lớp SelfQueryRetriever từ LangChain với các tham số bao gồm LLM, vector database, document description, và metadata field description.

### Ví Dụ Sử Dụng

Khi sử dụng truy vấn "Tôi muốn xem một bộ phim có đánh giá cao hơn 8.5", self-query retriever sẽ chuyển đổi truy vấn này thành phần ngữ nghĩa để tìm kiếm và bộ lọc metadata (rating > 8.5) để lọc các bộ phim phù hợp. Kết quả là các bộ phim có đánh giá lớn hơn 8.5 được truy xuất thành công.

## Parent Document Retriever

### Xung Đột Trong Yêu Cầu

Khi chia nhỏ tài liệu để truy xuất, thường có những yêu cầu mâu thuẫn:

- **Yêu cầu thứ nhất**: Các tài liệu nhỏ để embedding có thể phản ánh chính xác ý nghĩa của chúng.
- **Yêu cầu thứ hai**: Các tài liệu đủ dài để giữ được ngữ cảnh của mỗi đoạn (chunk).

### Giải Pháp của Parent Document Retriever

Parent document retriever giải quyết xung đột này bằng cách sử dụng hai bộ chia văn bản (text splitters):

1. **Parent splitter**: Chia văn bản thành các chunk lớn để truy xuất.
2. **Child splitter**: Chia tài liệu thành các chunk nhỏ để tạo embedding có ý nghĩa.

### Cơ Chế Hoạt Động

Trong quá trình truy xuất, parent document retriever thực hiện các bước sau:

1. **Truy xuất chunk nhỏ**: Đầu tiên, fetches các chunk nhỏ từ vector store.
2. **Tìm parent IDs**: Look up các ID của tài liệu gốc (parent) chứa các chunk nhỏ.
3. **Trả về tài liệu gốc**: Trả về các tài liệu lớn hơn mà các chunk nhỏ thuộc về.

### Triển Khai

Để triển khai parent document retriever, cần:

1. **Hai text splitters**: Một parent splitter và một child splitter.
2. **Vector store**: Để lưu trữ các embedding.
3. **Store cho parent documents**: Để lưu trữ các tài liệu gốc.
4. **Tạo parent retriever object**: Và thêm tài liệu vào đó bằng phương thức add_documents.

### Ví Dụ Sử Dụng

Với truy vấn "smoking policy", parent document retriever truy xuất các chunk lớn được tạo bởi parent splitter, không phải các chunk được tạo bởi child splitter. Điều này đảm bảo rằng ngữ cảnh đầy đủ của chính sách được giữ nguyên.

## So Sánh Các Loại Retriever

| Loại Retriever | Mục Đích Chính | Ứng Dụng |
|----------------|----------------|----------|
| Multi-Query Retriever | Tạo nhiều phiên bản truy vấn | Khi cần tăng độ bao quát kết quả |
| Self-Query Retriever | Chuyển đổi truy vấn thành ngữ nghĩa + metadata | Khi tài liệu có metadata quan trọng |
| Parent Document Retriever | Chia nhỏ theo hai cấp độ | Khi cần giữ ngữ cảnh đầy đủ |

## Kết Luận

Trong bài học này, chúng ta đã tìm hiểu ba loại retriever nâng cao trong LangChain:

1. **Multi-Query Retriever**: Sử dụng LLM để tạo các phiên bản khác nhau của truy vấn, tạo ra tập hợp tài liệu phong phú hơn.

2. **Self-Query Retriever**: Chuyển đổi truy vấn thành hai thành phần - chuỗi để tìm kiếm ngữ nghĩa và bộ lọc metadata đi kèm.

3. **Parent Document Retriever**: Sử dụng hai text splitters - parent splitter chia văn bản thành các chunk lớn để truy xuất, và child splitter chia tài liệu thành các chunk nhỏ để tạo embedding có ý nghĩa.

Mỗi loại retriever có những ưu điểm riêng và phù hợp với các tình huống khác nhau trong ứng dụng thực tế. Việc lựa chọn đúng loại retriever phụ thuộc vào yêu cầu cụ thể của ứng dụng và đặc điểm của dữ liệu.

---

## Tài Liệu Tham Khảo

1. LangChain Documentation. (2025). *Retrievers*. https://python.langchain.com/docs/modules/data_connection/retrievers/
2. LangChain Documentation. (2025). *Multi-Query Retriever*. https://python.langchain.com/docs/modules/data_connection/retrievers/multi_query
3. LangChain Documentation. (2025). *Self-Query Retriever*. https://python.langchain.com/docs/modules/data_connection/retrievers/self_query
4. LangChain Documentation. (2025). *Parent Document Retriever*. https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever
5. IBM. (2025). *Advanced RAG with Vector Databases and Retrievers*. Coursera.
