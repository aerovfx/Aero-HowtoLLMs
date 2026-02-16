import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough08_Transformer(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Transformer) {
        return;
    }

    setInitialCamera(state, new Vec3(-135.531, 0.000, -353.905), new Vec3(291.100, 13.600, 5.706));

    let c0 = commentary(wt, null, 0)`

Và đó là một khối Transformer hoàn chỉnh!

Những khối này tạo nên phần lớn của bất kỳ mô hình GPT nào và được lặp lại nhiều lần, với đầu ra của một
khối được đưa vào khối tiếp theo, tiếp tục đường dẫn dư (residual pathway).

Như thường thấy trong deep learning, khó có thể nói chính xác từng lớp này đang làm gì, nhưng chúng ta
có một số ý tưởng chung: các lớp đầu có xu hướng tập trung vào việc học
các đặc trưng và mẫu ở mức thấp, trong khi các lớp sau học cách nhận biết và hiểu
các trừu tượng và mối quan hệ ở mức cao hơn. Trong ngữ cảnh xử lý ngôn ngữ tự nhiên, các
lớp thấp hơn có thể học ngữ pháp, cú pháp và liên kết từ đơn giản, trong khi các lớp cao hơn
có thể nắm bắt các mối quan hệ ngữ nghĩa phức tạp hơn, cấu trúc diễn ngôn và ý nghĩa phụ thuộc ngữ cảnh.

`;

}
