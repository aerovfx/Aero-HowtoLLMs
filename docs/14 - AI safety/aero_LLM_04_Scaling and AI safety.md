# Định Luật Mở Rộng (Scaling Laws) và Sự Phát Triển Của An Toàn Trí Tuệ Nhân Tạo

## Tóm tắt

Liệu sự phát triển của Mô hình Ngôn ngữ Lớn (LLMs) sẽ đi tới đâu? Bài viết phân tích các Định luật mở rộng (Scaling laws) trong Trí tuệ Nhân tạo, từ đó phản biện lại khuynh hướng ngoại suy sự gia tăng tuyến tính trong tương lai. Dựa trên khung phương pháp của Diễn giải Cơ chế (Mechanistic Interpretability), bài viết cũng giải thích lý do toán học khiến các mô hình cực lớn phát sinh các hiện tượng phi tuyến tính như Chồng chập (Superposition), khiến công tác đánh giá an toàn AI ngày một khó khăn.

---

## 1. Định Luật Mở Rộng (Scaling Laws) Là Gì?

Bản chất của "Scaling Laws" (Định luật mở rộng) không bắt nguồn từ một phương trình vật lý bất di bất dịch, mà thay vào đó là **một quan sát thực nghiệm** dựa trên dữ liệu quá khứ. 

Trong lịch sử máy tính, định luật Moore chứng minh rằng số lượng bóng bán dẫn hoặc khả năng tính toán trên một vi mạch sẽ tăng gấp đôi xấp xỉ mỗi hai năm. Tương tự, trong Học sâu (Deep Learning), các phép đo thực nghiệm cho thấy hàm mất mát (loss function) của LLMs tiếp tục giảm đều khi ta tăng số lượng tham số $N$, lượng dữ liệu $D$, hoặc chi phí tính toán huấn luyện (Compute) $C$ theo tỷ lệ log-log.

Cụ thể, định luật chia sẻ dạng luật luỹ thừa (power-law):
$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha}
$$
Trong đó, $L(N)$ là hàm mất mát test loss, $N_c$ là hằng số và $\alpha$ thể hiện tốc độ cải thiện. Theo phương trình này, đồ thị trên trục log-log sẽ là một đường thẳng đi xuống.

---

## 2. Ảo Tưởng Ngoại Suy Trong An Toàn AI

Nhiều chuyên gia dự báo dựa trên chuỗi giá trị này (được gọi là các "Boomers") cho rằng tốc độ tăng trưởng của AI sẽ tiếp diễn theo phương tuyến tính này tới vô cực (infinity), dẫn tới các siêu trí tuệ nhân tạo. Tuy nhiên, ngoại suy cho các hệ thống phức tạp (complex dynamical systems) là một kỹ thuật thường dẫn tới kết luận sai lệch. Có các rào cản vật lý và thực tiễn:
1. **Giới Hạn Dữ Liệu:** Theo Epoch AI, nhân loại đã tiến gần tới giới hạn khai thác lượng văn bản chất lượng cao trên internet. Dữ liệu chất lượng thấp sinh ra bởi chính AI đang làm nhiễu loạn phân phối xác suất.
2. **Khuynh hướng phi bậc (Diminishing Returns):** Các kỹ năng không tuyến tính, giống như tuổi dậy thì của con người, sự cải thiện sẽ ở mức cực đại trong vài năm đầu, sau đó chững lại (plateau).

Vì vậy, việc đầu tư nâng cấp phần cứng không đảm bảo giải quyết được các giới hạn của thuật toán trong việc hiểu biết ngữ nghĩa ở cấp độ con người. Trọng tâm nên được san sẻ cho **An toàn AI (AI Safety)** thay vì chạy đua quy mô mô hình một cách mù quáng.

---

## 3. Khủng Hoảng Quy Mô: Chuyển Vị Toán Học và Hiện Tượng Chồng Chập (Superposition)

Tại sao các mô hình càng to thì An toàn AI càng khó diễn giải?
Dưới lăng kính của Mechanistic Interpretability, điều này xuất phát từ hiện tượng Chồng chập Không gian (Superposition).

### 3.1 Vấn đề Số Chiều (Curse of Dimensionality)
Khi một LLM học, nó sẽ thiết lập một không gian đặc trưng ảo kích thước khổng lồ $M$. Tuy nhiên, quy mô các lớp ẩn (đại diện bởi số neuron $N$) không thể tăng tiến cùng tốc độ. Do $M \gg N$, mô hình buộc phải "nén" các tri thức vào không gian vector không trực giao (non-orthogonal representation). 

$$
x \approx \sum_{i=1}^{M} c_i W_{in}^T W_{out} \cdot e_i
$$

### 3.2 Lượng Hóa Chồng Chập (Superposition) qua Mã Hóa Thưa Thớt
Thay vì mỗi neuron biểu diễn 1 khái niệm, $N$ neuron sẽ biểu diễn một tổ hợp $M$ tính năng kích hoạt thưa (sparse features) bằng cách giải bài toán tối ưu hoá tối đa:

$$
\mathcal{L} = \mathbb{E}_{x} \left[ \left\| x - \sum_{i=1}^{m} f_i(x) d_i \right\|^2_2 + \lambda \sum_{i=1}^{m} | f_i(x) | \right]
$$

Hệ quả là, các khái niệm an toàn, hành vi độc hại hay đạo đức bị ép chặt lên nhau vào cùng một cụm tham số vector ảo $d_i$. Nghĩa là việc bóc tách một hành vi xấu (như ý định tống tiền) mà không làm tổn thương năng lực nói chung của AI gặp sự nhiễu loạn đa chiều (interference). Do đó, sự phát triển quy mô (Scaling) vô tình kích hoạt sự chống đối lại tính minh bạch của chính mô hình đó.

---

## 4. Kết luận

Các định luật Scaling Laws cung cấp khung tham chiếu tuyệt vời để ước lượng khả năng phần mềm, nhưng hoàn toàn sai lầm nếu dùng để tiên đoán tương lai AI và bỏ qua các bất định cơ bản. Việc tăng trưởng kích thước tỷ lệ thuận với mức độ trừu tượng hóa toán học trong không gian nơ-ron (chồng chập). Chính vì vậy, để hệ thống thực sự vừa thông minh vừa an toàn, các nhà khoa học phải tiếp tục mở khóa bí mật tại tầng ma trận bằng Khả năng diễn giải cơ chế thay vì chỉ tập trung vào việc bổ sung sức mạnh tính toán.

---

## Tài liệu tham khảo

1. **Kaplan, J., et al. (2020).** *Scaling Laws for Neural Language Models.* Open AI. arXiv:2001.08361.
2. **Elhage, N., et al. (2022).** *Toy Models of Superposition.* Distill/Anthropic.
3. **Bostrom, N. (2014).** *Superintelligence: Paths, Dangers, Strategies.*
4. **Hoffmann, J., et al. (2022).** *Training Compute-Optimal Large Language Models (Chinchilla).* DeepMind.
