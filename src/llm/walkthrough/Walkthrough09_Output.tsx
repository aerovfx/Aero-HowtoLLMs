import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough09_Output(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Output) {
        return;
    }

    setInitialCamera(state, new Vec3(-20.203, 0.000, -1642.819), new Vec3(281.600, -7.900, 2.298));

    let c0 = commentary(wt, null, 0)`

Cuối cùng, chúng ta đến phần cuối của mô hình. Đầu ra của khối Transformer cuối cùng được truyền qua
một lớp chuẩn hóa, và sau đó chúng ta sử dụng biến đổi tuyến tính (phép nhân ma trận), lần này không có bias.

Phép biến đổi cuối cùng này đưa mỗi vector cột của chúng ta từ độ dài C đến độ dài nvocab. Do đó,
nó thực sự tạo ra một điểm số cho mỗi từ trong từ vựng cho mỗi cột của chúng ta. Những
điểm số này có một tên đặc biệt: logits.

Tên "logits" xuất phát từ "log-odds", tức là logarit của tỷ lệ của mỗi token. "Log" được
sử dụng vì softmax mà chúng ta áp dụng tiếp theo thực hiện phép lũy thừa để chuyển đổi sang "odds" (tỷ lệ) hoặc xác suất.

Để chuyển đổi những điểm số này thành xác suất đẹp mắt, chúng ta truyền chúng qua phép toán softmax. Bây giờ, cho
mỗi cột, chúng ta có xác suất mà mô hình gán cho mỗi từ trong từ vựng.

Trong mô hình cụ thể này, nó đã học được tất cả các câu trả lời cho câu hỏi làm thế nào để sắp xếp
ba chữ cái, vì vậy các xác suất bị nghiêng nặng về câu trả lời đúng.

Khi chúng ta đang bước qua mô hình theo thời gian, chúng ta sử dụng xác suất của cột cuối cùng để xác định
token tiếp theo để thêm vào chuỗi. Ví dụ, nếu chúng ta đã cung cấp sáu token vào mô hình, chúng ta sẽ
sử dụng xác suất đầu ra của cột thứ 6.

Đầu ra của cột này là một loạt xác suất, và chúng ta thực sự phải chọn một trong số chúng để sử dụng
làm token tiếp theo trong chuỗi. Chúng ta làm điều này bằng cách "lấy mẫu từ phân phối". Nghĩa là, chúng ta ngẫu nhiên
chọn một token, được trọng số bởi xác suất của nó. Ví dụ, một token có xác suất 0.9 sẽ được
chọn 90% số lần.

Tuy nhiên, còn có các tùy chọn khác ở đây, chẳng hạn như luôn chọn token có xác suất cao nhất.

Chúng ta cũng có thể kiểm soát "độ mượt" của phân phối bằng cách sử dụng tham số nhiệt độ (temperature). Nhiệt độ cao hơn
sẽ làm cho phân phối đồng đều hơn, và nhiệt độ thấp hơn sẽ làm cho nó tập trung hơn
vào các token có xác suất cao nhất.

Chúng ta làm điều này bằng cách chia logits (đầu ra của phép biến đổi tuyến tính) cho nhiệt độ trước khi
áp dụng softmax. Vì phép lũy thừa trong softmax có tác động lớn đến các số lớn hơn,
làm cho tất cả chúng gần nhau hơn sẽ giảm tác động này.
`;

}
