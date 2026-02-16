import { duplicateGrid, splitGrid } from "../Annotations";
import { getBlockValueAtIdx } from "../components/DataFlow";
import { IBlkDef } from "../GptModelLayout";
import { drawText, IFontOpts, measureText } from "../render/fontRender";
import { lerp } from "@/src/utils/math";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough02_Embedding(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, tools: { c_str, c_blockRef, c_dimRef, afterTime, cleanup, breakAfter }, layout } = args;
    let render = state.render;

    if (wt.phase !== Phase.Input_Detail_Embedding) {
        return;
    }

    setInitialCamera(state, new Vec3(15.654, 0.000, -80.905), new Vec3(287.000, 14.500, 3.199));
    wt.dimHighlightBlocks = [layout.idxObj, layout.tokEmbedObj, layout.posEmbedObj, layout.residual0];

    commentary(wt)`
Chúng ta đã thấy trước đó cách các token được ánh xạ vào một chuỗi số nguyên bằng cách sử dụng bảng tra cứu đơn giản.
Những số nguyên này, ${c_blockRef('_token indices_', state.layout.idxObj, DimStyle.TokenIdx)}, là lần đầu tiên và duy nhất chúng ta thấy số nguyên trong mô hình.
Từ đây trở đi, chúng ta sử dụng số thực (số thập phân).

Hãy xem token thứ 4 (chỉ số 3) được sử dụng như thế nào để tạo ra vector cột thứ 4 của ${c_blockRef('_input embedding_', state.layout.residual0)}.`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.0);
    let t0_splitEmbedAnim = afterTime(null, 0.3);

    breakAfter();

    commentary(wt)`
Chúng ta sử dụng chỉ số token (trong trường hợp này ${c_str('B', DimStyle.Token)} = ${c_dimRef('1', DimStyle.TokenIdx)}) để chọn cột thứ 2 của ${c_blockRef('_ma trận token embedding_', state.layout.tokEmbedObj)} ở bên trái.
Lưu ý chúng ta đang dùng chỉ số bắt đầu từ 0, vì vậy cột đầu tiên ở chỉ số 0.

Điều này tạo ra một vector cột có kích thước ${c_dimRef('_C_ = 48', DimStyle.C)}, mà chúng ta gọi là token embedding.
    `;
    breakAfter();

    let t1_fadeEmbedAnim = afterTime(null, 0.3);
    let t2_highlightTokenEmbed = afterTime(null, 0.8);

    breakAfter();

    commentary(wt)`
Và vì chúng ta đang xem xét token ${c_str('B', DimStyle.Token)} ở _vị trí_ thứ 4 (t = ${c_dimRef('3', DimStyle.T)}), chúng ta sẽ lấy cột thứ 4 của ${c_blockRef('_ma trận position embedding_', state.layout.posEmbedObj)}.

Điều này cũng tạo ra một vector cột có kích thước ${c_dimRef('_C_ = 48', DimStyle.C)}, mà chúng ta gọi là position embedding.
    `;
    breakAfter();

    let t4_highlightPosEmbed = afterTime(null, 0.8);

    breakAfter();

    commentary(wt)`
Lưu ý rằng cả position và token embeddings này đều được học trong quá trình huấn luyện (được chỉ ra bởi màu xanh của chúng).

Bây giờ chúng ta đã có hai vector cột này, chúng ta đơn giản cộng chúng lại với nhau để tạo ra một vector cột khác có kích thước ${c_dimRef('_C_ = 48', DimStyle.C)}.
`;

    breakAfter();

    let t3_moveTokenEmbed = afterTime(null, 0.8);
    let t5_movePosEmbed = afterTime(null, 0.8);
    let t6_plusSymAnim = afterTime(null, 0.8);
    let t7_addAnim = afterTime(null, 0.8);
    let t8_placeAnim = afterTime(null, 0.8);
    let t9_cleanupInstant = afterTime(null, 0.0);
    let t10_fadeAnim = afterTime(null, 0.8);

    breakAfter();

    commentary(wt)`
Bây giờ chúng ta chạy quy trình tương tự cho tất cả các token trong chuỗi đầu vào, tạo ra một tập hợp các vector kết hợp cả giá trị token và vị trí của chúng.

`;

    breakAfter();

    let t11_fillRest = afterTime(null, 5.0);

    breakAfter();

    commentary(wt)`
Hãy thử di chuột qua các ô riêng lẻ trên ma trận ${c_blockRef('_input embedding_', state.layout.residual0)} để xem các phép tính và nguồn gốc của chúng.

Chúng ta thấy rằng chạy quy trình này cho tất cả các token trong chuỗi đầu vào tạo ra một ma trận có kích thước ${c_dimRef('_T_', DimStyle.T)} x ${c_dimRef('_C_', DimStyle.C)}.
${c_dimRef('_T_', DimStyle.T)} viết tắt cho ${c_dimRef('_time_', DimStyle.T)} (thời gian), tức là bạn có thể nghĩ các token xuất hiện sau trong chuỗi như là xuất hiện muộn hơn trong thời gian.
${c_dimRef('_C_', DimStyle.C)} viết tắt cho ${c_dimRef('_channel_', DimStyle.C)} (kênh), nhưng cũng được gọi là "feature" (tính năng) hoặc "dimension" (chiều) hoặc "embedding size" (kích thước embedding). Độ dài ${c_dimRef('_C_', DimStyle.C)} này
là một trong nhiều "hyperparameters" (siêu tham số) của mô hình, và được người thiết kế chọn để cân bằng giữa kích thước mô hình và hiệu suất.

Ma trận này, mà chúng ta gọi là ${c_blockRef('_input embedding_', state.layout.residual0)}, bây giờ đã sẵn sàng để được truyền xuống qua mô hình.
Tập hợp ${c_dimRef('T', DimStyle.T)} cột, mỗi cột có độ dài ${c_dimRef('C', DimStyle.C)}, sẽ trở thành một hình ảnh quen thuộc xuyên suốt hướng dẫn này.
`;

    cleanup(t9_cleanupInstant, [t3_moveTokenEmbed, t5_movePosEmbed, t6_plusSymAnim, t7_addAnim, t8_placeAnim]);
    cleanup(t10_fadeAnim, [t0_splitEmbedAnim, t1_fadeEmbedAnim, t2_highlightTokenEmbed, t4_highlightPosEmbed]);

    moveCameraTo(state, t_moveCamera, new Vec3(7.6, 0, -33.1), new Vec3(290, 15.5, 0.8));

    let residCol: IBlkDef = null!;
    let exampleIdx = 3;
    if ((t0_splitEmbedAnim.t > 0.0 || t10_fadeAnim.t > 0.0) && t11_fillRest.t === 0) {
        splitGrid(layout, layout.idxObj, Dim.X, exampleIdx + 0.5, t0_splitEmbedAnim.t * 4.0);

        layout.residual0.access!.disable = true;
        layout.residual0.opacity = lerp(1.0, 0.1, t1_fadeEmbedAnim.t);

        residCol = splitGrid(layout, layout.residual0, Dim.X, exampleIdx + 0.5, t0_splitEmbedAnim.t * 4.0)!;
        residCol.highlight = 0.3;

        residCol.opacity = lerp(1.0, 0.0, t1_fadeEmbedAnim.t);

    }

    let tokValue = getBlockValueAtIdx(layout.idxObj, new Vec3(exampleIdx, 0, 0)) ?? 1;


    let tokColDupe: IBlkDef | null = null;
    let posColDupe: IBlkDef | null = null;

    if (t2_highlightTokenEmbed.t > 0.0) {
        let tokEmbedCol = splitGrid(layout, layout.tokEmbedObj, Dim.X, tokValue + 0.5, t2_highlightTokenEmbed.t * 4.0)!;

        tokColDupe = duplicateGrid(layout, tokEmbedCol);
        tokColDupe.t = 'i';
        tokEmbedCol.highlight = 0.3;

        let startPos = new Vec3(tokEmbedCol.x, tokEmbedCol.y, tokEmbedCol.z);
        let targetPos = new Vec3(residCol.x, residCol.y, residCol.z).add(new Vec3(-2.0, 0, 3.0));

        let pos = startPos.lerp(targetPos, t3_moveTokenEmbed.t);

        tokColDupe.x = pos.x;
        tokColDupe.y = pos.y;
        tokColDupe.z = pos.z;
    }


    if (t4_highlightPosEmbed.t > 0.0) {
        let posEmbedCol = splitGrid(layout, layout.posEmbedObj, Dim.X, exampleIdx + 0.5, t4_highlightPosEmbed.t * 4.0)!;

        posColDupe = duplicateGrid(layout, posEmbedCol);
        posColDupe.t = 'i';
        posEmbedCol.highlight = 0.3;

        let startPos = new Vec3(posEmbedCol.x, posEmbedCol.y, posEmbedCol.z);
        let targetPos = new Vec3(residCol.x, residCol.y, residCol.z).add(new Vec3(2.0, 0, 3.0));

        let pos = startPos.lerp(targetPos, t5_movePosEmbed.t);

        posColDupe.x = pos.x;
        posColDupe.y = pos.y;
        posColDupe.z = pos.z;
    }

    if (t6_plusSymAnim.t > 0.0 && tokColDupe && posColDupe && t7_addAnim.t < 1.0) {
        for (let c = 0; c < layout.shape.C; c++) {
            let plusCenter = new Vec3(
                (tokColDupe.x + tokColDupe.dx + posColDupe.x) / 2,
                tokColDupe.y + layout.cell * (c + 0.5),
                tokColDupe.z + tokColDupe.dz / 2);

            let isActive = t6_plusSymAnim.t > (c + 1) / layout.shape.C;
            let opacity = lerp(0.0, 1.0, isActive ? 1 : 0);

            let fontOpts: IFontOpts = { color: new Vec4(0, 0, 0, 1).mul(opacity), size: 1.5, mtx: Mat4f.fromTranslation(plusCenter) };
            let w = measureText(render.modelFontBuf, '+', fontOpts);

            drawText(render.modelFontBuf, '+', -w / 2, -fontOpts.size / 2, fontOpts);
        }
    }

    let origResidPos = residCol ? new Vec3(residCol.x, residCol.y, residCol.z) : new Vec3();
    let offsetResidPos = origResidPos.add(new Vec3(0.0, 0, 3.0));

    if (t7_addAnim.t > 0.0 && tokColDupe && posColDupe) {
        let targetPos = offsetResidPos;
        let tokStartPos = new Vec3(tokColDupe.x, tokColDupe.y, tokColDupe.z);
        let posStartPos = new Vec3(posColDupe.x, posColDupe.y, posColDupe.z);

        let tokPos = tokStartPos.lerp(targetPos, t7_addAnim.t);
        let posPos = posStartPos.lerp(targetPos, t7_addAnim.t);

        tokColDupe.x = tokPos.x;
        tokColDupe.y = tokPos.y;
        tokColDupe.z = tokPos.z;
        posColDupe.x = posPos.x;
        posColDupe.y = posPos.y;
        posColDupe.z = posPos.z;

        if (t7_addAnim.t > 0.95) {
            tokColDupe.opacity = 0.0;
            posColDupe.opacity = 0.0;
            residCol.opacity = 1.0;
            residCol.highlight = 0.0;
            residCol.access!.disable = false;
            residCol.x = targetPos.x;
            residCol.y = targetPos.y;
            residCol.z = targetPos.z;
        }
    }

    if (t8_placeAnim.t > 0.0) {
        let startPos = offsetResidPos;
        let targetPos = origResidPos;
        let pos = startPos.lerp(targetPos, t8_placeAnim.t);
        residCol.x = pos.x;
        residCol.y = pos.y;
        residCol.z = pos.z;
    }

    if (t9_cleanupInstant.t > 0.0 && residCol) {
        residCol.opacity = 1.0;
        residCol.highlight = 0.0;
        residCol.access!.disable = false;
    }

    if (t11_fillRest.t > 0.0) {
        layout.residual0.access!.disable = true;

        let prevInfo = startProcessBefore(state, layout.residual0);
        processUpTo(state, t11_fillRest, layout.residual0, prevInfo);
    }
    // new Vec3(-6.9, 0, -36.5), new Vec3(281.5, 5.5, 0.8)
}
