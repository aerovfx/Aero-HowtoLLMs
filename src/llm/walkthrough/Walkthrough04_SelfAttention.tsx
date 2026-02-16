import React from 'react';
import { dimProps, duplicateGrid, findSubBlocks, splitGrid, splitGridAll } from "../Annotations";
import { drawDataFlow, getBlockValueAtIdx } from "../components/DataFlow";
import { BlkSpecial, getBlkDimensions, IBlkDef, setBlkPosition } from "../GptModelLayout";
import { drawDependences } from "../Interaction";
import { IProgramState } from "../Program";
import { drawText, IFontOpts, measureText } from "../render/fontRender";
import { clamp, makeArray } from "@/src/utils/data";
import { lerp, lerpSmoothstep } from "@/src/utils/math";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";
import { embedInline } from "./Walkthrough01_Prelim";
import { Colors, commentary, DimStyle, dimStyleColor, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { BlockText } from '../components/CommentaryHelpers';
import clsx from 'clsx';

export function walkthrough04_SelfAttention(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_str, c_blockRef, c_dimRef, breakAfter, cleanup } } = args;
    let { C, A, nHeads } = layout.shape;

    if (wt.phase !== Phase.Input_Detail_SelfAttention) {
        return;
    }

    let block0 = layout.blocks[0];
    let head2 = block0.heads[2];

    setInitialCamera(state, new Vec3(-125.258, 0.000, -178.805), new Vec3(294.000, 12.800, 2.681));
    wt.dimHighlightBlocks = [layout.residual0, block0.ln1.lnResid, ...head2.cubes];

    commentary(wt, null, 0)`
Lớp self-attention có lẽ là trái tim của Transformer và GPT. Đây là giai đoạn mà các
cột trong ma trận input embedding "nói chuyện" với nhau. Cho đến nay, và trong tất cả các giai đoạn khác,
các cột có thể được xem xét độc lập.

Lớp self-attention được tạo thành từ nhiều head (đầu), và chúng ta sẽ tập trung vào một trong số chúng ngay bây giờ.`;
    breakAfter();
    let t_moveCamera = afterTime(null, 1.0);
    let t_highlightHeads = afterTime(null, 2.0);
    let t_moveCamera2 = afterTime(null, 1.0);
    let t_focusHeads = focusSelfAttentionHeadTimers(args, 3.0);

    breakAfter();
    commentary(wt)`
Bước đầu tiên là tạo ra ba vector cho mỗi trong số ${c_dimRef('T', DimStyle.T)} cột từ ${c_blockRef('ma trận input embedding đã chuẩn hóa', block0.ln1.lnResid)}.
Những vector này là các vector Q, K, và V:

${embedInline(<ul>
        <li>Q: <BlockText blk={head2.qBlock}>Vector Query (Truy vấn)</BlockText></li>
        <li>K: <BlockText blk={head2.kBlock}>Vector Key (Khóa)</BlockText></li>
        <li>V: <BlockText blk={head2.vBlock}>Vector Value (Giá trị)</BlockText></li>
    </ul>)}

Để tạo ra một trong những vector này, chúng ta thực hiện phép nhân ma trận-vector với một bias được cộng thêm. Mỗi
ô đầu ra là một sự kết hợp tuyến tính của vector đầu vào. Ví dụ: đối với các ${c_blockRef('vector Q', head2.qBlock)}, điều này được thực hiện bằng một tích vô hướng (dot product) giữa
một hàng của ${c_blockRef('ma trận trọng số Q', head2.qWeightBlock)} và một cột của ${c_blockRef('ma trận đầu vào', block0.ln1.lnResid)}.`;
    breakAfter();

    let t_focusQCol = afterTime(null, 1.0);
    let t_qIterColDot = afterTime(null, 3.0);

    breakAfter();
    commentary(wt)`
Phép toán tích vô hướng (dot product), mà chúng ta sẽ thấy rất nhiều, khá đơn giản: Chúng ta ghép mỗi phần tử từ
vector thứ nhất với phần tử tương ứng của vector thứ hai, nhân các cặp đó lại với nhau
rồi sau đó cộng tất cả kết quả lại.`;
    breakAfter();

    let t_moveDotCells = afterTime(null, 2.0, 0.5);
    let t_dotCellsZoomClose = afterTime(null, 1.0, 0.5);
    let t_collapseDotCellsA = afterTime(null, 2.0);
    let t_collapseDotCellsB = afterTime(null, 2.0, 0.5);
    let t_dotCellsZoomOut = afterTime(null, 1.0, 0.5);
    let t_addBias = afterTime(null, 2.0, 0.5);
    let t_moveToDest = afterTime(null, 0.5);

    breakAfter();
    commentary(wt)`
Đây là một cách tổng quát và đơn giản để đảm bảo mỗi phần tử đầu ra có thể bị ảnh hưởng bởi tất cả các
phần tử trong vector đầu vào (trong đó sự ảnh hưởng đó được xác định bởi các trọng số). Chính vì thế nó thường xuyên
xuất hiện trong các mạng neural.

Chúng ta lặp lại phép toán này cho mỗi ô đầu ra trong các vector Q, K, V:`;
    breakAfter();

    let t_revertFocusCol = afterTime(null, 0.25, 0.5);
    cleanup(t_revertFocusCol, [t_focusQCol]);
    let t_processQkv = afterTime(null, 5.0);

    breakAfter();
    commentary(wt)`
Chúng ta làm gì với các vector Q (query), K (key), và V (value)? Tên gọi của chúng
cho chúng ta một gợi ý: "key" (khóa) và "value" (giá trị) gợi nhớ đến một từ điển (dictionary) trong phần mềm, với các khóa ánh xạ đến
các giá trị. Khi đó "query" (truy vấn) là những gì chúng ta sử dụng để tra cứu giá trị.

${embedInline(<div className='ml-4'>
        <div className='mt-1 text-center italic'>Sự tương đồng trong phần mềm</div>
        <div className='text-sm mt-1 mb-1 text-gray-600'>Bảng tra cứu:</div>
        <div className='font-mono'>{'table = { "key0": "value0", "key1": "value1", ... }'}</div>
        <div className='text-sm mt-1 mb-1 text-gray-600'>Quá trình truy vấn:</div>
        <div className='font-mono'>{'table["key1"] => "value1"'}</div>
    </div>)}

Trong trường hợp của self-attention, thay vì trả về một mục duy nhất, chúng ta trả về một sự kết hợp có trọng số
của các mục. Để tìm trọng số đó, chúng ta thực hiện tích vô hướng giữa một vector Q và mỗi
vector K. Chúng ta chuẩn hóa trọng số đó, trước khi cuối cùng sử dụng nó để nhân với
vector V tương ứng, và sau đó cộng tất cả chúng lại.

${embedInline((() => {
        let keyCol = dimStyleColor(DimStyle.Intermediates);
        let valCol = dimStyleColor(DimStyle.A);
        let qCol = dimStyleColor(DimStyle.Aggregates);

        return <div className='ml-4'>
            <div className='mt-1 text-center italic'>Self Attention</div>
            <div className='text-sm mt-2 mb-1 text-gray-600'>Bảng tra cứu:</div>
            <div className='font-mono flex items-center'>K:
                <div className='mx-2 my-1'>{makeTextVector(keyCol)}</div>
                <div className='mx-2 my-1'>{makeTextVector(keyCol)}</div>
                <div className='mx-2 my-1'>{makeTextVector(keyCol)}</div>
            </div>
            <div className='font-mono flex items-center'>V:
                <div className='mx-2 my-1'>{makeTextVector(valCol)}</div>
                <div className='mx-2 my-1'>{makeTextVector(valCol)}</div>
                <div className='mx-2 my-1'>{makeTextVector(valCol)}</div>
            </div>
            <div className='text-sm mt-2 mb-1 text-gray-600'>Quá trình truy vấn:</div>
            <div className='font-mono flex items-center'>
                <div className='flex items-center'>Q: <div className='mx-2 my-1'>{makeTextVector(qCol)}</div></div>
            </div>
            <div className='font-mono flex items-center -ml-2 mt-2'>
                <div className='flex items-center mx-2'>w0 = <div className='m-1'>{makeTextVector(qCol)}</div>.<div className='m-1'>{makeTextVector(keyCol)}</div></div>
                <div className='flex items-center mx-2'>w1 = <div className='m-1'>{makeTextVector(qCol)}</div>.<div className='m-1'>{makeTextVector(keyCol)}</div></div>
                <div className='flex items-center mx-2'>w2 = <div className='m-1'>{makeTextVector(qCol)}</div>.<div className='m-1'>{makeTextVector(keyCol)}</div></div>
            </div>
            <div className='font-mono flex items-center my-2'>
                [w0n, w1n, w2n] =&nbsp;<span className='italic'>chuẩn hóa</span>([w0, w1, w2])
            </div>
            <div className='font-mono flex items-center'>
                kết quả =
                w0n * <div className='ml-2 mr-2 my-1'>{makeTextVector(valCol)}</div>&nbsp;+&nbsp;
                w1n * <div className='ml-2 mr-2 my-1'>{makeTextVector(valCol)}</div>&nbsp;+&nbsp;
                w2n * <div className='ml-2 mr-2 my-1'>{makeTextVector(valCol)}</div>
            </div>

        </div>;
    })())}

Để có một ví dụ cụ thể hơn, hãy nhìn vào cột thứ 6 (${c_dimRef('t = 5', DimStyle.T)}), nơi
chúng ta sẽ thực hiện truy vấn:`;
    breakAfter();

    let t_focusQKVCols = afterTime(null, 1.0);

    breakAfter();
    // It would like to find relevant information from other columns and extract their values. The other
    // columns each have a K (key) vector, which represents the information that that column has, and our
    // Q (query) vector is what information is relevant to us.
    commentary(wt)`
Các mục {K, V} trong bảng tra cứu của chúng ta là 6 cột trong quá khứ, và giá trị Q là thời điểm hiện tại.

Đầu tiên chúng ta tính tích vô hướng giữa ${c_blockRef('vector Q', head2.qBlock)} của cột hiện tại (${c_dimRef('t = 5', DimStyle.T)}) và ${c_blockRef('các vector K', head2.kBlock)}
của mỗi cột trước đó. Kết quả sau đó được lưu vào hàng tương ứng (${c_dimRef('t = 5', DimStyle.T)})
của ${c_blockRef('ma trận attention', head2.attnMtx)}.`;
    breakAfter();

    let t_processAttnRow = afterTime(null, 3.0);

    breakAfter();
    commentary(wt)`
Những tích vô hướng này là một cách đo lường sự tương đồng giữa hai vector. Nếu chúng rất
giống nhau, tích vô hướng sẽ lớn. Nếu chúng rất khác nhau, tích vô hướng sẽ nhỏ hoặc
mang giá trị âm.

Ý tưởng chỉ sử dụng query đối với các key trong quá khứ làm cho đây là self-attention _nhân quả_ (causal). Tức là,
các token không thể "nhìn thấy tương lai".

Một yếu tố khác là sau khi lấy tích vô hướng, chúng ta chia cho căn bậc hai của ${c_dimRef('A', DimStyle.A)}, trong đó
${c_dimRef('A', DimStyle.A)} là chiều dài của các vector Q/K/V. Việc tỷ lệ (scaling) này được thực hiện để ngăn các giá trị quá lớn
áp đặt quá trình chuẩn hóa (softmax) ở bước tiếp theo.

Chúng ta sẽ bỏ qua chi tiết về phép toán softmax (sẽ giải thích sau); nói tóm lại, mỗi hàng được chuẩn hóa để tổng
bằng 1.`;
    breakAfter();

    let t_processAttnSmAggRow = afterTime(null, 1.0);
    let t_processAttnSmRow = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
Cuối cùng, chúng ta có thể tạo ra vector đầu ra cho cột của mình (${c_dimRef('t = 5', DimStyle.T)}). Ta nhìn vào hàng (${c_dimRef('t = 5', DimStyle.T)}) của
${c_blockRef('ma trận self-attention đã chuẩn hóa', head2.attnMtxSm)} và với mỗi phần tử, nhân với ${c_blockRef('vector V', head2.vBlock)} tương ứng
của các cột khác theo từng phần tử.`;
    breakAfter();

    let t_zoomVOutput = afterTime(null, 0.4, 0.5);
    // multi-step process here, displaying the multiplication of each element of the row with each column of V
    let t_expandVCols = afterTime(null, 1.0, 0.5);
    let t_moveAttnVals = afterTime(null, 1.0, 0.5);
    let t_applyMultiplies = afterTime(null, 1.0, 0.5);
    let t_applyAdds = afterTime(null, 1.0, 0.5);
    let t_placeVOutput = afterTime(null, 1.0, 0.5);
    let t_finalizeVOutput = afterTime(null, 0.5, 0.5);

    breakAfter();
    commentary(wt)`
Sau đó chúng ta cộng các giá trị này lại để tạo ra vector đầu ra. Do đó, vector đầu ra sẽ bị chi phối bởi
các vector V từ những cột có điểm số cao.

Bây giờ ta đã biết quy trình, hãy chạy nó cho tất cả các cột.`;

    breakAfter();

    let t_processRemainZoom = afterTime(null, 0.5, 0.5);
    cleanup(t_processRemainZoom, [t_focusQKVCols]);
    let t_processRemain = afterTime(null, 8.0);

    breakAfter();
    commentary(wt)`
Và đó là quy trình cho một head của lớp self-attention. Vậy mục tiêu chính của self-attention là
mỗi cột muốn tìm thông tin liên quan từ các cột khác và trích xuất giá trị của chúng, và
thực hiện điều này bằng cách so sánh vector _query_ (truy vấn) của nó với các _keys_ (khóa) của những cột khác. Với rằng buộc
là nó chỉ có thể nhìn vào quá khứ.
`;

    // Running this process for all the columns produces our self-attention matrix, which is a square
    // matrix, T x T, and due to the causal nature of the process, is a lower triangular matrix.

    moveCameraTo(state, t_moveCamera, new Vec3(-192.1, 0, -214.8), new Vec3(293.5, 49, 2.3));
    moveCameraTo(state, t_moveCamera2, new Vec3(-92.7, 0, -219), new Vec3(286, 12.8, 1.4));


    if (t_highlightHeads.t > 0.0) {
        block0.selfAttendLabel.visible = lerp(0, 1, t_highlightHeads.t * 10);
        let headPos = t_highlightHeads.t * nHeads;
        let headIdx = clamp(Math.floor(headPos), 0, nHeads - 1);
        let headFrac = headPos - headIdx;

        let labelOpacity = lerp(0.0, 1.0, headFrac / 0.3); // * lerp(1.0, 0.0, (1.0 - headFrac) / 0.3);

        let head = block0.heads[headIdx];
        head.headLabel.visible = labelOpacity;

        for (let blk of head.headLabel.cubes) {
            blk.highlight = labelOpacity * 0.4;
        }
    }

    if (t_focusHeads.t0_dissolveHeads.t > 0.0) {
        let head = block0.heads[2];
        let t = t_focusHeads.t0_dissolveHeads.t;
        for (let blk of head.headLabel.cubes) {
            blk.highlight = lerp(1.0, 0.0, t * 4) * 0.4;
        }
        head2.qBlock.access!.disable = true;
        head2.kBlock.access!.disable = true;
        head2.vBlock.access!.disable = true;
    }

    focusSelfAttentionHead(args, t_focusHeads);

    moveCameraTo(state, t_dotCellsZoomClose, new Vec3(-53, 0, -155.5), new Vec3(274.1, 8.5, 0.4));
    moveCameraTo(state, t_dotCellsZoomOut, new Vec3(-92.7, 0, -219), new Vec3(286, 12.8, 1.4));

    let exampleIdx = 3;
    if (t_focusQCol.t > 0) {
        let otherOpacity = lerp(1.0, 0.2, t_focusQCol.t);
        head2.qBlock.opacity = otherOpacity;
        head2.kBlock.opacity = otherOpacity;
        head2.vBlock.opacity = otherOpacity;
        block0.ln1.lnResid.opacity = otherOpacity;

        let splitAmt = lerp(0.0, 2.0, t_focusQCol.t);
        let qCol = splitGrid(layout, head2.qBlock, Dim.X, exampleIdx + 0.5, splitAmt)!;
        let inputCol = splitGrid(layout, block0.ln1.lnResid, Dim.X, exampleIdx + 0.5, splitAmt)!;

        qCol.opacity = 1.0;
        inputCol.opacity = 1.0;

        if (t_qIterColDot.t > 0) {
            let aPos = t_qIterColDot.t * A;
            let aIdx = clamp(Math.floor(aPos), 0, A - 1);
            let destIdx = new Vec3(exampleIdx, aIdx, 0);
            let pinIdx = new Vec3(exampleIdx, 0, 0);

            drawDependences(state, head2.qBlock, destIdx);
            drawDataFlow(state, head2.qBlock, destIdx, pinIdx);
            splitGrid(layout, qCol, Dim.Y, aIdx + 0.5, 0);
            for (let b of findSubBlocks(qCol, Dim.Y, null, aIdx)) {
                b.access!.disable = false;
            }
            inputCol.highlight = 0.3;
        }

        let targetTop = new Vec3(inputCol.x - 26, inputCol.y, inputCol.z + 5);
        let addTarget = new Vec3(targetTop.x + layout.cell, targetTop.y - layout.cell * 12, targetTop.z);
        let biasTarget = new Vec3(addTarget.x - layout.cell * 3, addTarget.y, addTarget.z);

        if (t_moveDotCells.t > 0 && t_moveToDest.t === 0.0) {
            let qWeightRow = findSubBlocks(head2.qWeightBlock, Dim.Y, A - 1, null)[0];

            let qCells = splitGridAll(layout, qWeightRow, Dim.X);
            let inCells = splitGridAll(layout, inputCol, Dim.Y);

            let cellMovePct = 0.5;
            let prevCY = 0;
            for (let c = 0; c < C; c++) {
                let cPos = c / (C - 1);
                let startT = (1.0 - cellMovePct) * (1.0 - cPos);
                let cellMoveT = inverseLerp(startT, startT + cellMovePct, t_moveDotCells.t);

                let qInitial = getBlkDimensions(qCells[c]);
                let qFinal = targetTop.add(new Vec3(0, c * layout.cell * 1.2));

                let inInitial = getBlkDimensions(inCells[c]);
                let inFinal = targetTop.add(new Vec3(layout.cell * 2, c * layout.cell * 1.2));

                if (t_dotCellsZoomOut.t > 0) {
                    qFinal = inFinal = new Vec3(targetTop.x + layout.cell, targetTop.y - layout.cell * 12, qFinal.z);
                }

                // animate the cells from the initial grid position to be lined up next to each other
                setBlkPosition(qCells[c], qInitial.tl.lerp(qFinal, cellMoveT));
                setBlkPosition(inCells[c], inInitial.tl.lerp(inFinal, cellMoveT));

                let transitionPt = 0.15;
                let collapsDotCellsT = lerp(0.0, transitionPt, t_collapseDotCellsA.t) + lerp(0, 1 - transitionPt, t_collapseDotCellsB.t);

                let startT2 = 0.9 * cPos;
                let cellTimesSymT = inverseLerp(startT2, startT2 + 0.1, collapsDotCellsT);

                // move the cells together in the times/add operation, with them ending up in the same place
                if (cellTimesSymT > 0.0 && t_dotCellsZoomOut.t == 0.0) {
                    let qCurr = getBlkDimensions(qCells[c]);
                    let inCurr = getBlkDimensions(inCells[c]);

                    let qCellPos = new Vec3(
                        lerp(qCurr.tl.x, addTarget.x, cellTimesSymT * 2.0),
                        lerp(qCurr.tl.y, addTarget.y, cellTimesSymT),
                        qCurr.tl.z
                    );
                    let inCellPos = new Vec3(
                        lerp(inCurr.tl.x, addTarget.x, cellTimesSymT * 2.0),
                        lerp(inCurr.tl.y, addTarget.y, cellTimesSymT),
                        qCurr.tl.z
                    );

                    setBlkPosition(qCells[c], qCellPos);
                    setBlkPosition(inCells[c], inCellPos);

                    if (c > 0 && cellTimesSymT > 0.0) {

                        let midPt = new Vec3(
                            lerp(qCurr.br.x, inCurr.tl.x, 0.5),
                            lerp(prevCY, qCellPos.y, 0.5),
                            qCurr.tl.z + layout.cell / 2);

                        let mtx = Mat4f.fromTranslation(midPt);
                        let fontOpts: IFontOpts = { color: Colors.Black, size: 1.5, mtx };
                        let w = measureText(state.render.modelFontBuf, "+", fontOpts);
                        drawText(state.render.modelFontBuf, "+", -w / 2, -fontOpts.size / 2, fontOpts);
                    }
                    prevCY = qCellPos.y + layout.cell;
                }

                // draw times symbol between blocks (provided they're next to each other)
                // done after the other operations so that they are always positioned correctly
                if (cellMoveT >= 1.0) {
                    drawSymbolBetweenBlocks(args, qCells[c], inCells[c], Dim.X, "x", { size: 1.5, color: Colors.Black });
                }
            }

            // move the bias block next to the resulting dotprod cell, draw a plus between them, and then join them together
            if (t_addBias.t >= 0.0) {
                let qBiasCell = findSubBlocks(head2.qBiasBlock, Dim.Y, A - 1, null)[0];

                let qBiasInitial = getBlkDimensions(qBiasCell);
                let qBiasPos = qBiasInitial.tl.lerp(biasTarget, inverseLerp(0, 0.4, t_addBias.t));
                setBlkPosition(qBiasCell, qBiasPos);

                let moveTogetherT = inverseLerp(0.6, 1.0, t_addBias.t);
                qBiasInitial = getBlkDimensions(qBiasCell);
                qBiasPos = qBiasInitial.tl.lerp(addTarget, moveTogetherT);
                setBlkPosition(qBiasCell, qBiasPos);

                if (t_addBias.t > 0.4) {
                    drawSymbolBetweenBlocks(args, qBiasCell, qCells[qCells.length - 1], Dim.X, "+", { size: 1.5, color: Colors.Black });
                }
            }
        }

        // move the resulting dotprod cell to the final destination
        if (t_moveToDest.t > 0) {
            let qWeightRow = findSubBlocks(head2.qWeightBlock, Dim.Y, A - 1, null)[0];
            let qBiasCell = findSubBlocks(head2.qBiasBlock, Dim.Y, A - 1, null)[0];
            qBiasCell.opacity = t_moveToDest.t;
            qWeightRow.opacity = t_moveToDest.t;
            inputCol.opacity = t_moveToDest.t;

            let qResultCell = findSubBlocks(qCol, Dim.Y, A - 1, null)[0];
            let qResultInitial = getBlkDimensions(qResultCell);
            let qResultPos = qResultInitial.tl.lerp(addTarget, 1.0 - t_moveToDest.t);

            setBlkPosition(qResultCell, qResultPos);
        }
    }

    // simple run-through and process of each of the Q, K, V blocks
    if (t_processQkv.t > 0) {
        let processStart = startProcessBefore(state, head2.qBlock);
        processUpTo(state, t_processQkv, head2.vBlock, processStart);
    }

    // highlight the example index column for the Q block, and previous columns for K and V
    let attnExampleIdx = 5;
    if (t_focusQKVCols.t > 0 && t_processRemain.t <= 0) {
        let ignoreOpacity = lerp(1.0, 0.2, t_focusQKVCols.t);
        head2.qBlock.opacity = ignoreOpacity;
        head2.kBlock.opacity = ignoreOpacity;
        head2.vBlock.opacity = ignoreOpacity;

        let qCol = splitGrid(layout, head2.qBlock, Dim.X, attnExampleIdx + 0.5, 0)!;
        splitGrid(layout, head2.kBlock, Dim.X, attnExampleIdx + 0.5, 0)!;
        splitGrid(layout, head2.vBlock, Dim.X, attnExampleIdx + 0.5, 0)!;

        let kBeforeCols = findSubBlocks(head2.kBlock, Dim.X, null, attnExampleIdx);
        let vBeforeCols = findSubBlocks(head2.vBlock, Dim.X, null, attnExampleIdx);

        for (let col of [...kBeforeCols, ...vBeforeCols, qCol]) {
            col.opacity = 1.0;
        }

        head2.attnMtx.access!.disable = true;
        head2.attnMtxSm.access!.disable = true;
        head2.attnMtxAgg1.access!.disable = true;
        head2.attnMtxAgg2.access!.disable = true;

        head2.qBlock.opacity = 1.0;
        head2.kBlock.opacity = 1.0;
        head2.vBlock.opacity = 1.0;
    }

    moveCameraTo(state, t_focusQKVCols, new Vec3(-91.5, 0, -227.9), new Vec3(270.1, -38.4, 0.8));

    if (t_processAttnRow.t > 0 && t_processRemain.t <= 0) {
        let aIdx = clamp(Math.floor(t_processAttnRow.t * (attnExampleIdx + 1)), 0, attnExampleIdx);
        let destIdx = new Vec3(aIdx, attnExampleIdx, 0);
        let pinIdx = new Vec3(attnExampleIdx, 0, 0);

        if (t_processAttnSmAggRow.t <= 0) {
            drawDependences(state, head2.attnMtx, destIdx);
            drawDataFlow(state, head2.attnMtx, destIdx, pinIdx);
        }

        let attnRow = splitGrid(layout, head2.attnMtx, Dim.Y, attnExampleIdx, 0)!;
        splitGrid(layout, attnRow, Dim.X, aIdx, 0)!;
        let attnRowStart = findSubBlocks(attnRow, Dim.X, null, aIdx);

        for (let blk of attnRowStart) {
            blk.access!.disable = false;
        }
    }

    if (t_processAttnSmAggRow.t > 0 && t_processRemain.t <= 0) {
        let agg0T = inverseLerp(0, 0.5, t_processAttnSmAggRow.t);
        let agg1T = inverseLerp(0.5, 1.0, t_processAttnSmAggRow.t);
        let hidePopup = t_processAttnSmRow.t > 0;
        processDim(state, head2.attnMtxAgg2, Dim.Y, attnExampleIdx, agg0T, { pinIdx: new Vec3(5, 0, 0), clamp: true, hidePopup });

        if (agg1T > 0) {
            processDim(state, head2.attnMtxAgg1, Dim.Y, attnExampleIdx, agg1T, { pinIdx: new Vec3(-12, 0, 0), clamp: true, hidePopup });
        }
    }

    if (t_processAttnSmRow.t > 0 && t_processRemain.t <= 0) {
        let hidePopup = t_zoomVOutput.t > 0;
        processDim(state, head2.attnMtxSm, Dim.Y, attnExampleIdx, t_processAttnSmRow.t, { pinIdx: new Vec3(5, 0, 0), clamp: true, maxIdx: attnExampleIdx + 1, hidePopup });
    }

    if (t_zoomVOutput.t > 0 && t_processRemain.t <= 0) {
        head2.vOutBlock.access!.disable = true;
    }

    // Do the attn_sm times V vectors to get the output vectors animation
    {
        moveCameraTo(state, t_zoomVOutput, new Vec3(-91.9, 0, -267.9), new Vec3(270.1, -7.5, 0.7));

        let topLeftPos = getBlkDimensions(head2.vBlock).tl.add(new Vec3(0, 4, 5));
        let midLeftPos = topLeftPos.add(new Vec3(0, layout.cell * (A / 2 - 0.5)));

        if (t_expandVCols.t > 0 && t_placeVOutput.t <= 0) {
            let allVCols = [];
            let vBeforeCols = findSubBlocks(head2.vBlock, Dim.X, null, attnExampleIdx);
            for (let col of vBeforeCols) {
                allVCols.push(...splitGridAll(layout, col, Dim.X));
            }

            let allAttnCells = [];
            let attnRow = findSubBlocks(head2.attnMtxSm, Dim.Y, attnExampleIdx, attnExampleIdx)[0];
            let attnCellsBefore = findSubBlocks(attnRow, Dim.X, null, attnExampleIdx);
            for (let cell of attnCellsBefore) {
                for (let subCell of splitGridAll(layout, cell, Dim.X)) {
                    allAttnCells.push(duplicateGrid(layout, subCell));
                }
            }

            for (let i = 0; i < attnExampleIdx + 1; i++) {
                let attnVal = getBlockValueAtIdx(head2.attnMtxSm, new Vec3(i, attnExampleIdx, 0)) ?? 0.2;

                let initColPos = getBlkDimensions(allVCols[i]).tl;
                let destColPos = topLeftPos.add(new Vec3(i * layout.cell * 5, 0, 0));

                setBlkPosition(allVCols[i], initColPos.lerp(destColPos, t_expandVCols.t));

                let initAttnPos = getBlkDimensions(allAttnCells[i]).tl;
                let destAttnPos = midLeftPos.add(new Vec3(i * layout.cell * 5 - 2 * layout.cell, 0));

                setBlkPosition(allAttnCells[i], initAttnPos.lerp(destAttnPos, t_moveAttnVals.t));

                if (t_applyMultiplies.t > 0) {
                    initAttnPos = destAttnPos;
                    destAttnPos = initAttnPos.add(new Vec3(layout.cell * 2, 0));
                    setBlkPosition(allAttnCells[i], initAttnPos.lerp(destAttnPos, t_applyMultiplies.t));

                    allAttnCells[i].opacity = 1.0 - t_applyMultiplies.t;
                    allVCols[i].highlight = lerp(0.0, attnVal * 1.5, t_applyMultiplies.t);
                }

                if (t_moveAttnVals.t > 0.8 && t_applyMultiplies.t < 0.7) {
                    drawSymbolBetweenBlocks(args, allVCols[i], allAttnCells[i], Dim.X, 'x', { color: Colors.Black, size: 1.5 });
                }

                if (t_applyAdds.t > 0) {
                    initColPos = destColPos;
                    destColPos = topLeftPos.add(new Vec3(0, 0, attnVal * 1.0));

                    setBlkPosition(allVCols[i], initColPos.lerp(destColPos, t_applyAdds.t));
                }

                if (t_applyMultiplies.t > 0.6 && i > 0 && t_applyAdds.t < 0.7) {
                    drawSymbolBetweenBlocks(args, allVCols[i - 1], allVCols[i], Dim.X, '+', { color: Colors.Black, size: 1.5 });
                }
            }
        }

        if (t_placeVOutput.t > 0 && t_finalizeVOutput.t <= 0) {
            let prepareT = inverseLerp(0, 0.5, t_placeVOutput.t);

            let vOutCol = splitGrid(layout, head2.vOutBlock, Dim.X, attnExampleIdx + 0.5, prepareT * 2.0)!;
            vOutCol.access!.disable = true;
            vOutCol.opacity = lerp(1.0, 0.0, prepareT);

            for (let col of findSubBlocks(head2.vBlock, Dim.X, null, attnExampleIdx)) {
                col.opacity = t_placeVOutput.t;
            }

            let vOutColDupe = duplicateGrid(layout, vOutCol);
            vOutColDupe.access!.disable = false;
            vOutColDupe.opacity = 1.0;

            let colInitialPos = topLeftPos;
            let colFinalPos = getBlkDimensions(vOutCol).tl;

            setBlkPosition(vOutColDupe, colInitialPos.lerp(colFinalPos, t_placeVOutput.t));
        }

        if (t_finalizeVOutput.t > 0) {
            let splitAmt = lerp(1.0, 0.0, t_finalizeVOutput.t) * 2.0;
            let vOutCol = splitGrid(layout, head2.vOutBlock, Dim.X, attnExampleIdx + 0.5, splitAmt)!;
            vOutCol.access!.disable = false;
        }
    }

    moveCameraTo(state, t_processRemainZoom, new Vec3(-99.7, 0, -230.1), new Vec3(275.6, -4.4, 1.2));

    if (t_processRemain.t > 0) {
        for (let blk of [head2.attnMtx, head2.attnMtxSm, head2.attnMtxAgg1, head2.attnMtxAgg2, head2.vOutBlock]) {
            blk.access!.disable = true;
        }

        let processStart = startProcessBefore(state, head2.attnMtx);
        processUpTo(state, t_processRemain, head2.vOutBlock, processStart);
    }

    // if (t_processAllAttn.t > 0) {
    //     let processStart = startProcessBefore(state, head2.attnMtx);
    //     processUpTo(state, t_processAllAttn, head2.attnMtx, processStart);
    // }

}

// when t < edge0, returns 0
// when t > edge1, returns 1
// when t is between edge0 and edge1, returns a value between 0 and 1
// note that edge1 must be greater than edge0
export function inverseLerp(edge0: number, edge1: number, t: number) {
    return (clamp(t, edge0, edge1) - edge0) / (edge1 - edge0);
}

export interface IProcessDimOpts {
    pinIdx?: Vec3;
    clamp?: boolean;
    maxIdx?: number;
    hidePopup?: boolean;
}

export function processDim(state: IProgramState, block: IBlkDef, dim: Dim, destIdx: number, t: number, options: IProcessDimOpts = {}) {
    let { layout } = state;
    let { pinIdx, clamp: keep, maxIdx, hidePopup } = options;
    let otherDim = dim === Dim.X ? Dim.Y : Dim.X;

    let { cx: cxOther } = dimProps(block, otherDim);

    pinIdx ||= new Vec3(0, 0, 0);

    let rowCol = splitGrid(layout, block, dim, destIdx, 0);

    if (!rowCol) {
        return;
    }

    let maxPos = maxIdx ?? cxOther; // for attn matrix, reduce to the row number
    let cellPos = t * maxPos;

    if (keep) {
        cellPos = clamp(cellPos, 0, maxPos - 1);
    }

    let cellIdx = Math.floor(cellPos);

    if (cellIdx >= maxPos) {
        return;
    }

    splitGrid(layout, rowCol, otherDim, cellIdx + 0.5, 0);

    // cell!.highlight = 0.2;
    let destIdxVec = new Vec3(0, 0, 0);
    destIdxVec.setAt(dim, destIdx);
    destIdxVec.setAt(otherDim, cellIdx);

    if (rowCol && !hidePopup) {
        drawDependences(state, block, destIdxVec);
        drawDataFlow(state, block, destIdxVec, pinIdx);
    }

    for (let blk of findSubBlocks(rowCol, otherDim, null, cellIdx)) {
        blk.access!.disable = false;
    }
}

export function focusSelfAttentionHeadTimers(args: IWalkthroughArgs, duration: number) {
    let afterTime = args.tools.afterTime;

    let totalTime = 1.5 * 2;
    let timeScale = duration / totalTime;

    let t0_dissolveHeads = afterTime(null, 1 * timeScale, 0.5 * timeScale);
    let t2_alignqkv = afterTime(null, 1.0 * timeScale, 0.5 * timeScale);

    return { t0_dissolveHeads, t2_alignqkv };
}

export function focusSelfAttentionHead(args: IWalkthroughArgs, timers: ReturnType<typeof focusSelfAttentionHeadTimers>) {
    let { layout } = args;
    let { t0_dissolveHeads, t2_alignqkv } = timers;

    let targetHeadIdx = 2;
    let targetHead = layout.blocks[0].heads[targetHeadIdx];

    let block = layout.blocks[0];
    {
        for (let headIdx = 0; headIdx < block.heads.length; headIdx++) {
            if (headIdx == targetHeadIdx) {
                continue;
            }
            for (let cube of block.heads[headIdx].cubes) {
                cube.opacity = lerpSmoothstep(1.0, 0.0, t0_dissolveHeads.t);
            }
        }
    }

    {
        let headZ = targetHead.attnMtx.z;
        let targetHeadZ = block.ln1.lnResid.z;
        let deltaZ = lerpSmoothstep(0, (targetHeadZ - headZ), t2_alignqkv.t);
        for (let cube of targetHead.cubes) {
            cube.z += deltaZ;
        }
    }

    {
        let qkv = [
            [targetHead.qBlock, targetHead.qWeightBlock, targetHead.qBiasBlock],
            [targetHead.kBlock, targetHead.kWeightBlock, targetHead.kBiasBlock],
            [targetHead.vBlock, targetHead.vWeightBlock, targetHead.vBiasBlock],
        ];
        let targetZ = block.ln1.lnResid.z;
        let strideY = targetHead.qBlock.dy + layout.margin;
        let baseY = targetHead.qBlock.y;
        let qkvYPos = [-strideY * 2, -strideY, 0];

        for (let i = 0; i < 3; i++) {
            let y = lerpSmoothstep(qkv[i][0].y, baseY + qkvYPos[i], t2_alignqkv.t);
            let z = lerpSmoothstep(qkv[i][0].z, targetZ, t2_alignqkv.t);
            for (let cube of qkv[i]) {
                cube.y = y;
                cube.z = z;
            }
        }

        let blockMidY = (blk: IBlkDef) => blk.y + blk.dy / 2;

        let resid0Idx = layout.cubes.indexOf(block.ln1.lnResid);
        let yDelta = lerpSmoothstep(0, blockMidY(block.ln1.lnResid) - blockMidY(targetHead.kBlock), t2_alignqkv.t);

        for (let i = 0; i < resid0Idx; i++) {
            let targetOpacity = 0.2;
            layout.cubes[i].opacity = lerpSmoothstep(1.0, targetOpacity, t2_alignqkv.t);
        }

        let afterAttn = false;
        for (let i = resid0Idx + 1; i < layout.cubes.length; i++) {
            let cube = layout.cubes[i];
            cube.y += yDelta;
            if (afterAttn) {
                cube.opacity = Math.min(lerpSmoothstep(1.0, 0.2, t2_alignqkv.t), cube.opacity ?? 1.0);
            }
            afterAttn = afterAttn || cube === targetHead.vOutBlock;
        }
    }

}

export function drawSymbolBetweenBlocks(args: IWalkthroughArgs, block1: IBlkDef, block2: IBlkDef, dim: Dim, symbol: string, opts: { color: Vec4, size: number }) {
    let { color, size } = opts;

    let block1Dim = getBlkDimensions(block1);
    let block2Dim = getBlkDimensions(block2);

    let midPt: Vec3;

    if (dim === Dim.X) {
        midPt = new Vec3(
            lerp(block1Dim.br.x, block2Dim.tl.x, 0.5),
            (block1Dim.tl.y + block1Dim.br.y + block2Dim.tl.y + block2Dim.br.y) * 0.25,
            block1Dim.tl.z + args.layout.cell / 2)
    } else {
        midPt = new Vec3(
            (block1Dim.tl.x + block1Dim.br.x + block2Dim.tl.x + block2Dim.br.x) * 0.25,
            lerp(block1Dim.br.y, block2Dim.tl.y, 0.5),
            block1Dim.tl.z + args.layout.cell / 2);
    }

    let mtx = Mat4f.fromTranslation(midPt);
    let fontOpts: IFontOpts = { color, size, mtx };
    let w = measureText(args.state.render.modelFontBuf, symbol, fontOpts);
    drawText(args.state.render.modelFontBuf, symbol, -w / 2, -fontOpts.size / 2, fontOpts);
}

function makeTextVector(col: Vec4, count: number = 3) {
    let bgColor = new Vec4(col.x, col.y, col.z, 0.5);
    return <div className='flex flex-col pb-[2px]'>
        {makeArray(3, 0).map((_, i) => {
            return <div key={i} className={"w-3 h-3 border-2 mb-[-2px]"} style={{ borderColor: col.toHexColor(), backgroundColor: bgColor.toHexColor() }} />;
        })}
    </div>;
}
