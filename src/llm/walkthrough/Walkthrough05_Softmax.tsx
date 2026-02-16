import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough05_Softmax(args: IWalkthroughArgs) {
  let { walkthrough: wt, state } = args;

  if (wt.phase !== Phase.Input_Detail_Softmax) {
    return;
  }

  setInitialCamera(state, new Vec3(-24.350, 0.000, -1702.195), new Vec3(283.100, 0.600, 1.556));

  let c0 = commentary(wt, null, 0)`

Phép toán softmax được sử dụng như một phần của self-attention, như đã thấy trong phần trước, và nó
cũng sẽ xuất hiện ở cuối cùng của mô hình.

Mục tiêu của nó là lấy một vector và chuẩn hóa các giá trị sao cho tổng của chúng bằng 1.0. Tuy nhiên, nó không đơn giản
chỉ là chia cho tổng. Thay vào đó, mỗi giá trị đầu vào trước tiên được lũy thừa hóa (exponentiated).

  a = exp(x_1)

Điều này có tác dụng làm cho tất cả các giá trị trở thành dương. Khi đã có vector các giá trị
đã lũy thừa, sau đó chúng ta chia mỗi giá trị cho tổng của tất cả các giá trị. Điều này sẽ đảm bảo rằng tổng
các giá trị là 1.0. Vì tất cả các giá trị lũy thừa đều dương, chúng ta biết rằng các giá trị
kết quả sẽ nằm giữa 0.0 và 1.0, cung cấp một phân phối xác suất trên các giá trị ban đầu.

Đó là tất cả về softmax: đơn giản lũy thừa các giá trị rồi chia cho tổng.

Tuy nhiên, có một vấn đề nhỏ. Nếu bất kỳ giá trị đầu vào nào khá lớn, thì các
giá trị lũy thừa sẽ rất lớn. Chúng ta sẽ chia một số lớn cho một số rất lớn,
và điều này có thể gây ra vấn đề với số học dấu phẩy động.

Một tính chất hữu ích của phép toán softmax là nếu chúng ta cộng một hằng số vào tất cả các giá trị đầu vào,
kết quả sẽ giống nhau. Vì vậy chúng ta có thể tìm giá trị lớn nhất trong vector đầu vào và trừ nó
khỏi tất cả các giá trị. Điều này đảm bảo rằng giá trị lớn nhất là 0.0, và softmax vẫn ổn định
về mặt số học.

Hãy xem xét phép toán softmax trong ngữ cảnh của lớp self-attention. Vector đầu vào
của chúng ta cho mỗi phép toán softmax là một hàng của ma trận self-attention (nhưng chỉ đến đường chéo).

Giống như với chuẩn hóa lớp, chúng ta có một bước trung gian lưu trữ một số giá trị tổng hợp
để giữ cho quá trình hiệu quả.

Với mỗi hàng, chúng ta lưu giá trị max trong hàng và tổng của các giá trị đã dịch chuyển & lũy thừa.
Sau đó, để tạo ra hàng đầu ra tương ứng, chúng ta có thể thực hiện một tập hợp nhỏ các phép toán: trừ đi
max, lũy thừa, và chia cho tổng.

Tại sao lại có tên "softmax"? Phiên bản "cứng" (hard) của phép toán này, gọi là argmax, đơn giản tìm
giá trị lớn nhất, đặt nó thành 1.0, và gán 0.0 cho tất cả các giá trị khác. Ngược lại, phép toán softmax
hoạt động như một phiên bản "mềm" (soft) hơn. Do phép lũy thừa trong softmax, giá trị
lớn nhất được nhấn mạnh và đẩy về phía 1.0, trong khi vẫn duy trì phân phối xác suất
trên tất cả các giá trị đầu vào. Điều này cho phép biểu diễn tinh tế hơn, không chỉ nắm bắt lựa chọn
có khả năng nhất mà còn cả khả năng tương đối của các lựa chọn khác.
`;

}
