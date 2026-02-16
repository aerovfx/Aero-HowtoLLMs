import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { lerp, lerpSmoothstep } from "@/src/utils/math";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough06_Projection(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { breakAfter, afterTime, c_blockRef, c_dimRef, cleanup } } = args;

    if (wt.phase !== Phase.Input_Detail_Projection) {
        return;
    }

    setInitialCamera(state, new Vec3(-73.167, 0.000, -270.725), new Vec3(293.606, 2.613, 1.366));
    let block = layout.blocks[0];
    wt.dimHighlightBlocks = [...block.heads.map(h => h.vOutBlock), block.projBias, block.projWeight, block.attnOut];

    let outBlocks = block.heads.map(h => h.vOutBlock);

    commentary(wt, null, 0)`

Sau quá trình self-attention, chúng ta có các đầu ra từ mỗi head. Những đầu ra này là các
vector V đã được kết hợp thích hợp, chịu ảnh hưởng bởi các vector Q và K.

Để kết hợp ${c_blockRef('các vector đầu ra', outBlocks)} từ mỗi head, chúng ta đơn giản xếp chúng lên trên nhau. Vì vậy, tại thời điểm
${c_dimRef('t = 4', DimStyle.T)}, chúng ta chuyển từ 3 vector có độ dài ${c_dimRef('A = 16', DimStyle.A)} thành 1 vector có độ dài ${c_dimRef('C = 48', DimStyle.C)}.`;

    breakAfter();

    let t_fadeOut = afterTime(null, 1.0, 0.5);
    // let t_zoomToStack = afterTime(null, 1.0);
    let t_stack = afterTime(null, 1.0);

    breakAfter();

    commentary(wt)`

Đáng chú ý rằng trong GPT, độ dài của các vector trong một head (${c_dimRef('A = 16', DimStyle.A)}) bằng ${c_dimRef('C', DimStyle.C)} / num_heads.
Điều này đảm bảo rằng khi chúng ta xếp chúng lại với nhau, chúng ta được độ dài ban đầu, ${c_dimRef('C', DimStyle.C)}.

Từ đây, chúng ta thực hiện projection để nhận được đầu ra của lớp. Đây là một phép nhân ma trận-vector
đơn giản theo từng cột, với bias được cộng thêm.`;

    breakAfter();

    let t_process = afterTime(null, 3.0);

    breakAfter();

    commentary(wt)`

Bây giờ chúng ta có đầu ra của lớp self-attention. Thay vì truyền đầu ra này trực tiếp sang giai đoạn
tiếp theo, chúng ta cộng nó theo từng phần tử vào input embedding. Quy trình này, được biểu thị bằng
mũi tên dọc màu xanh lá, được gọi là _residual connection_ (kết nối dư) hoặc _residual pathway_ (đường dẫn dư).
`;

    breakAfter();

    let t_zoomOut = afterTime(null, 1.0, 0.5);
    let t_processResid = afterTime(null, 3.0);

    cleanup(t_zoomOut, [t_fadeOut, t_stack]);

    breakAfter();

    commentary(wt)`

Giống như chuẩn hóa lớp, đường dẫn dư (residual pathway) rất quan trọng để cho phép việc học hiệu quả trong các
mạng neural sâu.

Bây giờ với kết quả của self-attention trong tay, chúng ta có thể truyền nó sang phần tiếp theo của transformer:
mạng feed-forward.
`;

    breakAfter();

    if (t_fadeOut.active) {
        for (let head of block.heads) {
            for (let blk of head.cubes) {
                if (blk !== head.vOutBlock) {
                    blk.opacity = lerpSmoothstep(1, 0, t_fadeOut.t);
                }
            }
        }
    }

    if (t_stack.active) {
        let targetZ = block.attnOut.z;
        for (let headIdx = 0; headIdx < block.heads.length; headIdx++) {
            let head = block.heads[headIdx];
            let targetY = head.vOutBlock.y + head.vOutBlock.dy * (headIdx - block.heads.length + 1);
            head.vOutBlock.y = lerp(head.vOutBlock.y, targetY, t_stack.t);
            head.vOutBlock.z = lerp(head.vOutBlock.z, targetZ, t_stack.t);
        }
    }

    let processInfo = startProcessBefore(state, block.attnOut);

    if (t_process.active) {
        processUpTo(state, t_process, block.attnOut, processInfo);
    }

    moveCameraTo(state, t_zoomOut, new Vec3(-8.304, 0.000, -175.482), new Vec3(293.606, 2.623, 2.618));

    if (t_processResid.active) {
        processUpTo(state, t_processResid, block.attnResidual, processInfo);
    }
}
