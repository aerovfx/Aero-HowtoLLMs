
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
# Bố Cục Đồ Thị Mạng (Network Graph) Thông Qua Ma Trận Cosine Similarity

## Tóm tắt

Phân tích Mạng Lưới Điểm (Graph Network Analysis) là nền móng của Khoa học dữ liệu nhằm tìm ra các chuỗi liên kết cộng sinh trong cụm mô hình văn bản (Clustering Tokens). Thay vì sử dụng một bảng Pixel vuông (Heatmap Matrix) rất phổ biến nhưng thiếu chiều sâu thị giác, bài báo khoa học này thiết lập một thuật toán ánh xạ các điểm token thành mạng lưới vũ trụ ly tâm hình tròn. Thông qua mặt nạ phân cực (Binary mask thresholds), chúng ta có thể trực quan hệ mô hình sự cộng hưởng tính chất ngữ nghĩa giữa nhiều điểm vector nhúng.

---

## 1. Cơ Sở Thiết Lập Mặt Nạ Lân Cận (Spatial Thresholding Mask)

Để thiết lập cấu trúc cạnh liên kết (Edge) giữa $N$ phân tử (Nodes - Tokens), chúng ta cần khởi tạo Ma trận khoảng cách Tương quan góc Cosine $N \times N$, biểu thị độ trùng lặp đặc trưng góc của từng bộ vector:
$$
S(i,j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}
$$

**Tính Ngưỡng Chặn Dòng (Cut-off Threshold):** 
Trong Mạng nơ-ron, sự tương quan của $S$ luôn dày đặc ở mức $\sim 0.2$, sinh ra vô vàn rác kết nối nhiễu. Ta cần thanh tẩy đồ thị bằng việc tính toán Ngưỡng Độ bão hòa dựa trên hàm Phương sai Bán chuẩn (Median + 1 Standard Deviation) chuyên bắt tín hiệu bất thường cường độ cao:
$$ 
\text{Threshold } (T) = \text{Median}(S_{\text{upper-triangular}}) + \sigma(S_{\text{upper-triangular}}) 
$$
Tất cả những điểm $S(i, j) < T$ hoàn toàn bị thay thế bằng mặt nạ nhị phân câm (Binary mask $= 0$). Chỉ những kết nối siêu cường ($S(i, j) \geq T$) được lộ diện trong mạng lý tưởng, chuyển hóa tập hợp vector rối rắm (Dense matrix) thành Cấu trúc thưa thớt logic (Sparse matrix). Đừng quên ép hàm đường chéo chính (Diagonal tự tương quan) bằng 0.

---

## 2. Tính Toán Không Gian Vệ Tinh Tròn (Circular Graph Coordinates)

Thay vì rải loạn ngẫu nhiên x-y, sơ đồ Tròn (Ring Plot) được chọn để san bằng tính phân cấp thứ bậc, đồng dạng mọi token cách đều tâm.

### 2.1 Ma Trận Vector Phân Cực:
Giả thiết số lượng $N$ tokens sẽ được phân chia đều nhau đính trên một bán kính $R=1$, chúng ta sử dụng hệ tọa độ cực để tìm góc pha $d\theta$ và tọa độ $\theta$ mỗi góc chèn:
$$ 
\Delta \theta = \frac{2\pi}{N} 
$$
Dải Vector Pha Góc (Phase Angles): $\theta \in \left[ 0, ~ 2\pi - \Delta \theta \right]$. *Tại sao lại kết thúc ở $2\pi - \Delta \theta$? Vì kết thúc đúng tại $2\pi$ tương ứng góc $360^\circ$ sẽ gây ra sự tự chèn lớp đè lên điểm đếm gốc số $0$.*

Từ đó, hoành độ vi phân hiển thị ra tọa độ 2D của mỗi Token Node:
$$
x_i = \cos(\theta_i) 
$$
$$
y_i = \sin(\theta_i)
$$

### 2.2 Quy Hoạch Bậc Kết Nối (Degree Size Scaling):
Trong Graph Theory, "Sức hút" của một đỉnh vòng (Node Size) được tính bằng Bậc (Degree) - Tức là số lượng cạnh liên đới dính vào nó. Ở bài toán này, Đám mây cỡ hạt được quy định thông qua việc Đếm tần số vượt ngưỡng $T$ (Suprathreshold counts) của một Vector hàng:

$$ 
\text{DotSize}_i \propto 3 \times \sqrt{\sum_{j=1}^{N} \mathbb{I}(S(i, j) \geq T)} 
$$

*(Chuyển biến tỷ lệ thu phóng căn bậc hai giúp phân tán hình ảnh hài hòa và êm mắt).*

### 2.3 Đường Mạch (Color Mapping Edges):
Những sợi chỉ đường ranh giới thẳng đứng sẽ nối tọa độ $(x_i, y_i)$ và $(x_j, y_j)$ với tham chiếu màu thay đổi trượt theo hệ thang nóng (Plasma colormap). Đường mạch màu tím có nghĩa Cosine dư ở mức thấp, đường màu vàng nóng thể hiện những dòng xoáy điểm tựa ngữ nghĩa mãnh liệt móc xích từ vựng lại với nhau.

---

## 3. Ứng Dụng Xuyên Mạng Graph

Đồ thị Cosine không chỉ đơn thuần là bộ màu lòe loẹt. Khi thả vào văn bản chứa kiến thức hạt nhân (Physics, Networking), sơ đồ vệ tinh sẽ rẽ nhánh các cộng đồng (Communities Detection). Tính hiệu lực sinh học tập trung vào sự trồi lên của một lượng ít Nút vệ tinh siêu đại diện (Hub hubs) với hệ mạng chằng chịt, kéo theo các Nút vệ tinh vệ quốc (vệ tinh nhược kết nối) quay quần xung quanh, minh họa sự đa pha phân mảng trong cơ học diễn giải (Mechanistic Interpretability).

---

## Tài liệu tham khảo

1. **Newman, M. E. J. (2003).** *The structure and function of complex networks.* SIAM Review.
2. **Bastian, M., et al. (2009).** *Gephi: An open source software for exploring and manipulating networks.* ICWSM.
3. Tài liệu đào tạo bài giảng *Investigating token embeddings - Graph representation of cosine similarities.*
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
