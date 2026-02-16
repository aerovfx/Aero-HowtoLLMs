import { IBlockLayerLink, IGptModelLink, ILayerNormLayerLink, IModelShape, IGpuMoEBlockLayer } from "./GptModel";
import { isNil } from "@/src/utils/data";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3 } from "@/src/utils/vector";
import { IBufferTex } from "@/src/utils/renderPhases";
import { dimProps } from "./Annotations";
import { DimStyle } from "./walkthrough/WalkthroughTools";

export interface IBlkDef {
    idx: number; // index in the layout.cubes array
    t: 'w' | 'i' | 'a', // weights; intermediate value; aggregate (just LN & softmax)
    x: number;
    y: number;
    z: number;
    dx: number; // units: model-space
    dy: number;
    dz: number;
    cx: number; // units: number of cells
    cy: number;
    cz: number;
    access?: IBlkAccess;
    deps?: IBlkDeps;
    dimX: DimStyle;
    dimY: DimStyle;
    name: string;
    small: boolean; // small enough to not be worth rendering in large models
    // implicit dimZ = DimStyle.Batch for t === 'i'

    // fields that are post-added by the walk-through for various rendering configurations

    localMtx?: Mat4f; // for creating blocks that are sub-parts of a block
    // what to do for different axes?
    rangeOffsetsX?: [number, number][]; // if this block has been split, map from [[s0, xOff], [s1, xOff], ...] to the original block
    rangeOffsetsY?: [number, number][];
    rangeOffsetsZ?: [number, number][];
    highlight: number; // 0 - 1 (0 = no highlight, 1 = full highlight)
    opacity: number; // 0 - 1 (0 = transparent, 1 = opaque)
    special: BlkSpecial;
    transpose?: boolean; // transpose the process direction
    subs?: IBlkDef[]; // substitutes for this block (i.e. render these instead)
    offX?: number; // offset from the original block
    offY?: number;
    offZ?: number;
    sizeX?: number; // size of the sub block
    sizeY?: number;
    sizeZ?: number;
}

export enum BlkSpecial {
    None,
    Attention,
}

// define how a cell is computed from other blocks
// matrix-mulplication: cell(x, y, b) = sum_i(A[i, y] * B[x, i, b]) + C[0, y]
export interface IBlkDeps {
    dot?: [IBlkCellDep, IBlkCellDep];
    dotLen?: number;
    add?: IBlkCellDep[];
    special: BlKDepSpecial;
    lowerTri?: boolean;
}

export interface IBlkCellDep {
    src: IBlkDef;
    srcIdxMtx: Mat4f; // inputs: [x, y, b, [i]], outputs: [x, y, b]
}

interface IBlkDepArgs {
    dot?: [[IBlkDef, string], [IBlkDef, string]];
    dotLen?: number;
    add?: [IBlkDef, string][];
    lowerTri?: boolean; // only use the lower triangle of the matrix (causal attention matrices)
    special?: BlKDepSpecial;
}

export enum BlKDepSpecial {
    None,
    Softmax,
    Gelu,
    LayerNorm,
    InputEmbed,
    LayerNormMu,
    LayerNormSigma,
    SoftmaxAggMax,
    SoftmaxAggExp,
    Attention,
}

let depIdxVars = '0xybi';
function parseDepIdxStr(str: string): Mat4f {
    let mtx = Mat4f.zeros();
    for (let destI = 0; destI < str.length; destI++) {
        let srcIdx = depIdxVars.indexOf(str[destI]);
        if (srcIdx > 0) {
            mtx.s(destI, srcIdx - 1, 1.0);
        }
    }
    return mtx;
}

function depArgsToDeps(args: IBlkDepArgs): IBlkDeps {
    let makeBlkDeps = (src: IBlkDef, depStr: string) => ({ src, srcIdxMtx: parseDepIdxStr(depStr) });
    return {
        dot: args.dot && args.dot.map(([src, depStr]) => makeBlkDeps(src, depStr)) as [IBlkCellDep, IBlkCellDep],
        dotLen: args.dotLen,
        add: args.add && args.add.map(([src, depStr]) => makeBlkDeps(src, depStr)),
        special: args.special ?? BlKDepSpecial.None,
        lowerTri: args.lowerTri,
    };
}

export function getBlkDimensions(blk: IBlkDef) {
    let { x, y, z, dx, dy, dz } = blk;
    return {
        tl: new Vec3(x, y, z),
        br: new Vec3(x + dx, y + dy, z + dz),
    };
}

export function setBlkPosition(blk: IBlkDef, pos: Vec3) {
    blk.x = pos.x;
    blk.y = pos.y;
    blk.z = pos.z;
}

export interface IBlkAccess {
    src: IBufferTex;
    channel: 'r' | 'g' | 'b';
    scale: number;
    mat: Mat4f; // actually using the first two columns for a 3x2 matrix: mapping (x, y, z) integer cell coord to (x, y) src tex coord
    disable?: boolean;
}

interface IBlkAccessDefArgs {
    src?: IBufferTex;
    channel?: 'r' | 'g' | 'b';
    scale?: number;
    x: number[];
    y: number[];
}

interface IBlkDefArgs {
    t: 'w' | 'i' | 'a', // weights; intermediate value
    xL?: number; // pos of Left edge
    xR?: number; // Right
    xM?: number; // Middle
    zF?: number; // Front
    zB?: number; // Back
    zM?: number; // Middle
    name?: string;
    y: number;
    cx: number; // units: number of cells
    cz: number;
    cy: number;
    dimX: DimStyle;
    dimY: DimStyle;
    special?: BlkSpecial;
    access?: IBlkAccessDefArgs;
    deps?: IBlkDepArgs;
    small?: boolean;
    hidden?: boolean;
    transpose?: boolean;
}

export interface IBlkLabel {
    visible: number;
    cubes: IBlkDef[];
}

export interface IModelLayout {
    cell: number;
    height: number;
    margin: number;
    cubes: IBlkDef[];
}

export function cellPosition(layout: IModelLayout, blk: IBlkDef, dim: Dim, index: number) {
    let { x, rangeOffsets } = dimProps(blk, dim);
    let base = x + layout.cell * index;
    if (!rangeOffsets) {
        return base;
    }
    for (let [s, xOff] of rangeOffsets!) {
        if (index < s) {
            return base + xOff;
        }
    }
    return base;
}

export type IGptModelLayout = ReturnType<typeof genGptModelLayout>;
export type IGptLayerNormLayout = IGptModelLayout['ln_f'];

export function genGptModelLayout(shape: IModelShape, gptGpuModel: IGptModelLink | null = null, offset: Vec3 = new Vec3(0, 0, 0), detailLevel: number = 2) {
    let { B, T, C, vocabSize, nHeads, A, nBlocks } = shape;

    // work our way downwards from the top
    // x is to the left and right
    // y is positive going down, and the stack advances down from the top (at (0, 0, 0))
    // z is coming out of the page

    // a single batch of the residual pathway goes down the x-z plane
    // weights & off-residual pathways are left & right of the residual pathway (i.e. along x)
    // those blocks might have y-depth but that's OK: still have space to add batches
    // x = 0 is just to the left of time-cell t=0

    let isLargeModel = shape.nBlocks > 12;

    let y = 0;

    let cell = 1.5;
    let margin = Math.max(30, C / 4);

    function mk(args: IBlkDefArgs): IBlkDef {
        let xDef = [args.xL, args.xR, args.xM].map(a => +!isNil(a)).reduce((a, b) => a + b, 0);
        let yDef = [args.zF, args.zB, args.zM].map(a => +!isNil(a)).reduce((a, b) => a + b, 0);
        if (xDef !== 1 || yDef !== 1) {
            throw new Error(`Must supply exactly 1 x arg & 1 y arg: ${JSON.stringify(args)}`);
        }
        let dx = args.cx * cell;
        let dy = args.cz * cell;
        let x = !isNil(args.xL) ? args.xL : !isNil(args.xR) ? args.xR - dx : args.xM! - dx / 2;
        let z = !isNil(args.zB) ? args.zB : !isNil(args.zF) ? args.zF - dy : args.zM! - dy / 2;

        function ensure4(a: number[]) {
            return a.length === 4 ? a : [...a, 0];
        }

        return {
            dx: args.cx * cell,
            dy: args.cy * cell,
            dz: args.cz * cell,
            t: args.t,
            x: x,
            y: args.y,
            z: z,
            cx: args.cx,
            cy: args.cy,
            cz: args.cz,
            dimX: args.dimX,
            dimY: args.dimY,
            name: args.name ?? "<unknown>",
            access: args.access?.src ? {
                channel: args.access.channel ?? 'r',
                src: args.access.src,
                scale: args.access.scale ?? 1.0,
                mat: Mat4f.fromColMajor([...ensure4(args.access.x), ...ensure4(args.access.y), 0, 0, 0, 0, 0, 0, 0, 0]),
            } : undefined,
            deps: args.deps ? depArgsToDeps(args.deps) : undefined,
            opacity: args.hidden ? 0.0 : 1.0,
            highlight: 0.0,
            small: args.small ?? false,
            special: args.special ?? BlkSpecial.None,
            transpose: args.transpose,
            idx: -1,
        };
    }

    function mkLabel(init: number, cubes?: IBlkDef[]): IBlkLabel {
        return { visible: 0, cubes: cubes ?? [] };
    }

    let cubes: IBlkDef[] = [];

    let idxObj = mk({
        t: 'i', cx: T, cz: B, cy: 1, y: y,
        xM: 0, zM: 0,
        access: { src: gptGpuModel?.inputTokens, x: [0, 1, 0], y: [1, 0, T], scale: 1 / vocabSize },
        dimX: DimStyle.T, dimY: DimStyle.None,
        name: 'Token (Đầu vào)',
    });
    cubes.push(idxObj);

    // Multimodal Input: Images
    // We'll place them to the left of the token sequence for now
    if (shape.n_images && shape.n_images > 0) {
        let imgSize = 32; // Placeholder size for now
        let imgObj = mk({
            t: 'i', cx: imgSize, cz: 1, cy: imgSize, y: y,
            xR: -T * cell / 2 - margin * 4, zM: 0,
            access: { src: gptGpuModel?.inputImages, x: [0, 0, 0], y: [1, 32, 0], scale: 1.0 },
            dimX: DimStyle.None, dimY: DimStyle.None,
            name: 'Ảnh đầu vào',
        });
        cubes.push(imgObj);
    }

    let leftX = -T * cell / 2 - margin;
    let rightX = T * cell / 2 + margin;

    y += cell + margin;

    let tokEmbedObj = mk({
        t: 'w',
        xR: leftX, zM: 0, y: y,
        cx: vocabSize, cz: 1, cy: C, // src has shape [vocabSize, C]
        access: { src: gptGpuModel?.vocabEmbed.weight, x: [0, 1, 0], y: [1, 0, 0], scale: 10 },
        dimX: DimStyle.n_vocab, dimY: DimStyle.C,
        name: 'Nhúng Token (Embeddings)',
    });

    let posEmbedObj = mk({
        t: 'w',
        xL: rightX, zM: 0, y: y,
        cx: T, cz: 1, cy: C,
        access: { src: gptGpuModel?.posEmbed.weight, x: [0, 1, 0], y: [1, 0, 0], scale: 10 },
        dimX: DimStyle.T, dimY: DimStyle.C,
        name: 'Nhúng Vị trí (Position)',
    });

    let residual0 = mk({
        t: 'i',
        xM: 0, zM: 0, y: y,
        cx: T, cz: B, cy: C,
        access: { src: gptGpuModel?.add.output, x: [0, 1, 0], y: [1, 0, T], scale: 10 },
        deps: { add: [[tokEmbedObj, 'iy'], [posEmbedObj, 'xy'], [idxObj, 'x0']], special: BlKDepSpecial.InputEmbed }, // the i comes from the idxObj lookup
        dimX: DimStyle.T, dimY: DimStyle.C,
        name: 'Nhúng đầu vào',
    });
    cubes.push(idxObj, tokEmbedObj, posEmbedObj, residual0);

    let embedLabel = mkLabel(y, [idxObj, tokEmbedObj, posEmbedObj, residual0]);

    y += C * cell + margin;

    function createLn(x: number, src: IBlkDef, target?: ILayerNormLayerLink) {
        let lnLeftX = leftX + x;
        let resLeftX = lnLeftX - T * cell - margin;

        let lnAgg1 = mk({
            t: 'a', cx: T, cz: B, cy: 1, y: y,
            xR: lnLeftX, zM: 0,
            access: { src: target?.normAgg, x: [0, 1, 0], y: [1, 0, T], scale: 10.0, channel: 'r' },
            deps: { add: [[src, 'xi']], special: shape.isRMSNorm ? BlKDepSpecial.None : BlKDepSpecial.LayerNormMu },
            dimX: DimStyle.T, dimY: DimStyle.None, small: true,
            name: shape.isRMSNorm ? 'RMS Agg: RMS' : 'LN Agg: μ, σ',
            hidden: !!shape.isRMSNorm, // Hide Mu if it's RMSNorm
        });
        let lnAgg2 = mk({
            t: 'a', cx: T, cz: B, cy: 1, y: y + cell,
            xR: lnLeftX, zM: 0,
            access: { src: target?.normAgg, x: [0, 1, 0], y: [1, 0, T], scale: 10.0, channel: 'g' },
            deps: { add: [[src, 'xi']], special: shape.isRMSNorm ? BlKDepSpecial.None : BlKDepSpecial.LayerNormSigma },
            dimX: DimStyle.T, dimY: DimStyle.None, small: true,
            name: '',
        });

        y += 2 * cell + margin;

        let lnSigma = mk({
            t: 'w', cx: 1, cz: 1, cy: C, y: y,
            xR: resLeftX, zM: 0,
            access: { src: target?.normWeight, x: [1, 0, 0], y: [0, 1, 0], scale: 0.5 }, // mostly around 1.0
            dimX: DimStyle.None, dimY: DimStyle.C,
            name: 'γ', small: true,
        });
        let lnMu = mk({
            t: 'w', cx: 1, cz: 1, cy: C, y: y,
            xR: resLeftX - cell * 1 - margin, zM: 0,
            access: { src: target?.normBias, x: [1, 0, 0], y: [0, 1, 0] },
            dimX: DimStyle.None, dimY: DimStyle.C,
            name: 'β', small: true,
        });
        let lnResid = mk({
            t: 'i', cx: T, cz: B, cy: C, y: y,
            xR: lnLeftX, zM: 0,
            access: { src: target?.output, x: [0, 1, 0], y: [1, 0, T], scale: 1.0 },
            deps: { add: [[src, 'xy'], [lnAgg1, 'xi'], [lnAgg2, 'xi'], [lnSigma, '0y'], [lnMu, '0y']], special: BlKDepSpecial.LayerNorm }, // lnSigma is really mul rather than add
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: shape.isRMSNorm ? 'RMS Norm' : 'Chuẩn hóa lớp (Layer Norm)',
        });
        let lnCubes = [lnAgg1, lnAgg2, lnSigma, lnMu, lnResid];
        return { lnAgg1, lnAgg2, lnResid, lnSigma, lnMu, cubes: lnCubes };
    }

    let lnLeftX = leftX - (T + 2) * cell - 3 * margin;

    function createLayer(src: IBlkDef, target: IBlockLayerLink | IGpuMoEBlockLayer | undefined) {
        let ln1 = createLn(0, src, target?.ln_1);

        let interHeadMargin = 3 * margin + (C * cell) / 16;
        let qkvMargin = 1 * margin + (C * cell) / 16;
        /* ... continued in next edit ... */

        let headWidth = 3 * B * cell + qkvMargin * 2 + (isLargeModel ? 0 : interHeadMargin);

        let attn1Y = y + A * cell + margin + (isLargeModel ? 2 * A * cell : 0);
        let attn2Y = attn1Y; // + T * cell + margin;
        let vOutY = attn2Y + T * cell + margin;

        let attnLeftX = lnLeftX; // leftX - ((T + 2) * cell + 3 * margin);
        let qkvValLeftX = attnLeftX - T * cell - margin;
        let qkvBiasLeftX = qkvValLeftX - C * cell - margin;
        let stepPerHeadY = 0; // A * cell;

        let attnTarget = target?.attn;

        let heads = [];
        for (let i = 0; i < nHeads; i++) {
            let headZMid = headWidth * i - (nHeads - 1) * headWidth / 2;
            let qMid = headZMid + B * cell + qkvMargin;
            let kMid = headZMid;
            let vMid = headZMid - B * cell - qkvMargin;

            // x: C, y: A
            let qWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y,
                xR: qkvValLeftX, zM: qMid,
                access: { src: attnTarget?.qkvWeight, x: [1, 0, 0], y: [0, 1, 0, 0 * C + A * i], scale: C * 0.25 },
                dimX: DimStyle.C, dimY: DimStyle.A,
                name: 'Trọng số Q',
                hidden: detailLevel < 2,
            });

            let kWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y,
                xR: qkvValLeftX, zM: kMid,
                access: { src: attnTarget?.qkvWeight, x: [1, 0, 0], y: [0, 1, 0, 1 * C + A * i], scale: C * 0.25 },
                dimX: DimStyle.C, dimY: DimStyle.A,
                name: 'Trọng số K',
                hidden: detailLevel < 2,
            });

            let vWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y,
                xR: qkvValLeftX, zM: vMid,
                access: { src: attnTarget?.qkvWeight, x: [1, 0, 0], y: [0, 1, 0, 2 * C + A * i], scale: C * 0.25 },
                dimX: DimStyle.C, dimY: DimStyle.A,
                name: 'Trọng số V',
                hidden: detailLevel < 2,
            });

            let qkvWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A * 3, y: y,
                xR: qkvValLeftX, zM: kMid,
                dimX: DimStyle.C, dimY: DimStyle.C,
                name: 'Trọng số QKV',
                hidden: detailLevel < 2,
            });

            let qBiasBlock = mk({
                t: 'w', cx: 1, cz: 1, cy: A, y: y,
                xR: qkvBiasLeftX, zM: qMid,
                access: { src: attnTarget?.qkvBias, x: [1, 0, 0], y: [0, 1, 0, 0 * C + A * i] },
                dimX: DimStyle.None, dimY: DimStyle.A, small: true,
                name: 'Bias Q',
                hidden: detailLevel < 2 || shape.noBias,
            });

            let kBiasBlock = mk({
                t: 'w', cx: 1, cz: 1, cy: A, y: y,
                xR: qkvBiasLeftX, zM: kMid,
                access: { src: attnTarget?.qkvBias, x: [1, 0, 0], y: [0, 1, 0, 1 * C + A * i] },
                dimX: DimStyle.None, dimY: DimStyle.A, small: true,
                name: 'Bias K',
                hidden: detailLevel < 2 || shape.noBias,
            });

            let vBiasBlock = mk({
                t: 'w', cx: 1, cz: 1, cy: A, y: y,
                xR: qkvBiasLeftX, zM: vMid,
                access: { src: attnTarget?.qkvBias, x: [1, 0, 0], y: [0, 1, 0, 2 * C + A * i] },
                dimX: DimStyle.None, dimY: DimStyle.A, small: true,
                name: 'Bias V',
                hidden: detailLevel < 2 || shape.noBias,
            });

            let qBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: y,
                xR: attnLeftX, zM: qMid,
                access: { src: attnTarget?.qkvOutput, x: [0, 1, 0, 0 * C + A * i], y: [1, 0, T], scale: 1.0 },
                deps: { dot: [[qWeightBlock, 'iy'], [ln1.lnResid, 'xi']], add: [[qBiasBlock, '0y']], dotLen: C },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: 'Vector Q',
            });

            let kBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: y,
                xR: attnLeftX, zM: kMid,
                access: { src: attnTarget?.qkvOutput, x: [0, 1, 0, 1 * C + A * i], y: [1, 0, T], scale: 1.0 },
                deps: { dot: [[kWeightBlock, 'iy'], [ln1.lnResid, 'xi']], add: [[kBiasBlock, '0y']], dotLen: C },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: 'Vector K',
            });

            let vBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: y,
                xR: attnLeftX, zM: vMid,
                access: { src: attnTarget?.qkvOutput, x: [0, 1, 0, 2 * C + A * i], y: [1, 0, T], scale: 1.0 },
                deps: { dot: [[vWeightBlock, 'iy'], [ln1.lnResid, 'xi']], add: [[vBiasBlock, '0y']], dotLen: C },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: 'Vector V',
            });

            let qkvBlock = mk({
                t: 'i', cx: T, cz: B, cy: A * 3, y: y,
                xR: attnLeftX, zM: kMid,
                dimX: DimStyle.T, dimY: DimStyle.C,
                name: 'Vector QKV',
            });

            let attn2LeftX = attnLeftX - (T + 2) * cell - 2 * margin;

            let attnMtx = mk({
                t: 'i', cx: T, cz: B, cy: T, y: attn1Y,
                xR: attnLeftX, zM: headZMid,
                access: { src: attnTarget?.attnMatrix, x: [1, 0, 0], y: [0, 1, nHeads * T, T * i], scale: 1.0 },
                deps: { dot: [[qBlock, 'yi'], [kBlock, 'xi']], lowerTri: true, dotLen: A, special: BlKDepSpecial.Attention },
                dimX: DimStyle.T, dimY: DimStyle.T,
                special: BlkSpecial.Attention,
                transpose: true,
                name: 'Ma trận Chú ý',
            });

            let attnMtxAgg1 = mk({
                t: 'a', cx: 1, cz: B, cy: T, y: attn1Y,
                xR: attnLeftX - T * cell - margin - cell, zM: headZMid,
                access: { src: attnTarget?.attnMatrixSoftmax, x: [0, 0, 0, 1], y: [0, 1, nHeads * T, T * i], channel: 'r' },
                deps: { add: [[attnMtx, 'iy']], special: BlKDepSpecial.SoftmaxAggExp },
                dimX: DimStyle.None, dimY: DimStyle.T, small: true,
                name: '',
            });

            let attnMtxAgg2 = mk({
                t: 'a', cx: 1, cz: B, cy: T, y: attn1Y,
                xR: attnLeftX - T * cell - margin, zM: headZMid,
                access: { src: attnTarget?.attnMatrixSoftmax, x: [0, 0, 0, 1], y: [0, 1, nHeads * T, T * i], channel: 'g' },
                deps: { add: [[attnMtx, 'iy']], special: BlKDepSpecial.SoftmaxAggMax },
                dimX: DimStyle.None, dimY: DimStyle.T, small: true,
                name: '',
            });

            let attnMtxSm = mk({
                t: 'i', cx: T, cz: B, cy: T, y: attn1Y,
                xR: attn2LeftX, zM: headZMid,
                access: { src: attnTarget?.attnMatrixSoftmax, x: [1, 0, 0], y: [0, 1, nHeads * T, T * i], scale: 2.0 },
                deps: { add: [[attnMtx, 'xy'], [attnMtxAgg1, 'iy'], [attnMtxAgg2, 'iy']], lowerTri: true, special: BlKDepSpecial.Softmax },
                dimX: DimStyle.T, dimY: DimStyle.T,
                special: BlkSpecial.Attention,
                transpose: true,
                name: 'Softmax Ma trận Chú ý',
            });

            let vOutBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: vOutY + i * stepPerHeadY,
                xR: attnLeftX, zM: headZMid,
                access: { src: attnTarget?.scaledVectors, x: [0, 1, 0, i * A], y: [1, 0, T] },
                deps: { dot: [[vBlock, 'iy'], [attnMtxSm, 'ix']], dotLen: A },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: 'Dau ra V',
            });

            let headCubes = [...isLargeModel ? [qkvWeightBlock, qkvBlock] : [qWeightBlock, kWeightBlock, vWeightBlock, qBlock, kBlock, vBlock],
                qBiasBlock, kBiasBlock, vBiasBlock,
                attnMtx, attnMtxAgg1, attnMtxAgg2, attnMtxSm, vOutBlock];

            let headLabel = mkLabel(1.0, headCubes);
            let qLabel = mkLabel(1.0, [qWeightBlock, qBiasBlock, qBlock]);
            let kLabel = mkLabel(1.0, [kWeightBlock, kBiasBlock, kBlock]);
            let vLabel = mkLabel(1.0, [vWeightBlock, vBiasBlock, vBlock]);
            let biasLabel = mkLabel(1.0, [qBiasBlock, kBiasBlock, vBiasBlock]);
            let mtxLabel = mkLabel(1.0, [attnMtx, attnMtxAgg1, attnMtxAgg2, attnMtxSm]);
            let vectorLabel = mkLabel(1.0, [vOutBlock]);

            let head = {
                qWeightBlock, kWeightBlock, vWeightBlock,
                qBiasBlock, kBiasBlock, vBiasBlock,
                qBlock, kBlock, vBlock,
                attnMtx, attnMtxAgg1, attnMtxAgg2, attnMtxSm, vOutBlock,
                qLabel, kLabel, vLabel, biasLabel, mtxLabel, vectorLabel, headLabel,
                cubes: headCubes,
                labels: [qLabel, kLabel, vLabel, biasLabel, mtxLabel, vectorLabel, headLabel],
            };
            heads.push(head);
        }

        let vOutCombined = mk({
            t: 'i', cx: T, cz: B, cy: C, y: vOutY,
            xR: attnLeftX, zF: - headWidth * nHeads / 2,
            dimX: DimStyle.T, dimY: DimStyle.C,
            hidden: true,
            name: 'Đầu ra V Tổng hợp',
        });

        let vFinalZ = Math.max(
            vOutY + stepPerHeadY * (nHeads - 1) + A * cell + 2 * margin,
            y + C * cell + margin, // in case the layer norm block is shorter
        );

        let projWeight = mk({
            t: 'w', cx: C, cz: 1, cy: C, y: vFinalZ,
            xR: qkvValLeftX, zM: 0,
            access: { src: attnTarget?.proj.weight, x: [1, 0, 0], y: [0, 1, 0], scale: C * 0.5 },
            dimX: DimStyle.C, dimY: DimStyle.C,
            name: 'Trọng số Chiếu',
            hidden: detailLevel < 2,
        });

        let projBias = mk({
            t: 'w', cx: 1, cz: 1, cy: C, y: vFinalZ,
            xR: qkvValLeftX - C * cell - margin, zM: 0,
            access: { src: attnTarget?.proj.bias!, x: [0, 0, 0], y: [0, 1, 0], scale: C * 0.5 },
            dimX: DimStyle.None, dimY: DimStyle.C, small: true,
            name: 'Bias Chieu',
            hidden: detailLevel < 2 || shape.noBias,
        });

        let attnOut = mk({
            t: 'i', cx: T, cz: B, cy: C, y: vFinalZ,
            xR: attnLeftX, zM: 0,
            access: { src: attnTarget?.proj.output, x: [0, 1, 0], y: [1, 0, T] },
            // deps: { dot: [[projWeight, 'iy'], [vOutCombined, 'xi']], dotLen: C },
            // vOutCombined isn't displayed atm, so add from the heads instead
            deps: {
                dot: [[projWeight, 'iy'], [vOutCombined, 'xi']], dotLen: C,
                add: [[projBias, '0y'], ...heads.map(h => [h.vOutBlock, 'xi'] as [IBlkDef, string])]
            },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: 'Đầu ra Chú ý',
        });

        let attnResidual = mk({
            t: 'i', cx: T, cz: B, cy: C, y: vFinalZ,
            xM: 0, zM: 0,
            access: { src: attnTarget?.output, x: [0, 1, 0], y: [1, 0, T] },
            deps: { add: [[attnOut, 'xy'], [src, 'xy']] },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: 'Đầu ra Chú ý (Cong don)',
        });

        y = vFinalZ + C * cell + margin;

        let ln2 = createLn(0, attnResidual, target?.ln_2);

        let mlpCubes: IBlkDef[] = [];
        let mlpLabelCubes: IBlkDef[] = [];
        let mlpResidual: IBlkDef;

        let mlpFc: IBlkDef | undefined;
        let mlpFcWeight: IBlkDef | undefined;
        let mlpFcBias: IBlkDef | undefined;
        let mlpAct: IBlkDef | undefined;
        let mlpProjWeight: IBlkDef | undefined;
        let mlpProjBias: IBlkDef | undefined;
        let mlpResult: IBlkDef | undefined;

        if (target && 'experts' in target) {
            // MoE Layout
            // 1. Gate (Linear + Softmax)
            // 2. Experts (Many MLPs)

            let moeBlock = target as IGpuMoEBlockLayer;
            const numExperts = moeBlock.experts.length;

            // Pre-calculate grid position for router centering
            const expertsPerRow = 4;
            const expertW = C * cell * 0.8;
            const expertSpacingX = expertW + margin * 3;
            const gridWidth = expertsPerRow * expertSpacingX;
            const gridCenterX = attnLeftX - gridWidth / 2 - margin * 4;

            // Gate (Router)
            let gateWeight = mk({
                t: 'w', cx: C, cz: 1, cy: numExperts, y: y,
                xM: gridCenterX, zM: 0,  // Centered above grid
                access: { src: moeBlock.gate.weight, x: [1, 0, 0], y: [0, 1, 0], scale: C * 0.5 },
                dimX: DimStyle.C, dimY: DimStyle.None,
                name: 'Router: Trong so',
                hidden: detailLevel < 2,
                small: true,
            });

            y += cell * numExperts + margin;

            // Gate Scores (routing logits)
            let gateScores = mk({
                t: 'i', cx: T, cz: B, cy: numExperts, y: y,
                xM: gridCenterX, zM: 0,
                access: { src: moeBlock.gate.output, x: [0, 1, 0], y: [1, 0, T] },
                deps: { dot: [[gateWeight, 'yi'], [ln2.lnResid, 'xi']], dotLen: C },
                dimX: DimStyle.T, dimY: DimStyle.None,
                name: 'Router: Diem so',
                small: true,
            });

            y += T * cell + margin;

            // Gate Softmax (TOP-K selection probabilities)
            let gateSoftmax = mk({
                t: 'i', cx: T, cz: B, cy: numExperts, y: y,
                xM: gridCenterX, zM: 0,
                access: { src: moeBlock.gateSoftmax.output, x: [0, 1, 0], y: [1, 0, T] },
                deps: { special: BlKDepSpecial.Softmax },
                dimX: DimStyle.T, dimY: DimStyle.None,
                name: 'Router: Top-K',
            });

            y += T * cell + margin;

            mlpCubes.push(gateWeight, gateScores, gateSoftmax);
            mlpLabelCubes.push(gateWeight, gateScores, gateSoftmax);

            // Experts
            // Simple stacking for now
            // We need to aggregate their outputs

            let expertsCubes = [];
            let expertOutputs = [];

            for (let i = 0; i < moeBlock.experts.length; i++) {
                const expert = moeBlock.experts[i];
                const h = expert.fcLayer.weight.shape![0]; // Hidden size

                // Calculate grid position (2 rows x 4 cols)
                const expertsPerRow = 4;
                const row = Math.floor(i / expertsPerRow);
                const col = i % expertsPerRow;

                // Compact grid layout
                const expertW = C * cell * 0.8;
                const expertH = (h + C) * cell + margin * 2;
                const spacingX = expertW + margin * 3;
                const spacingY = expertH + margin * 2;

                // Grid base position (to the left)
                const gridBaseX = attnLeftX - (expertsPerRow * spacingX);
                const gridBaseY = y + row * spacingY;

                const expertX = gridBaseX + col * spacingX;
                const expertY = gridBaseY;

                let expFcWeight = mk({
                    t: 'w', cx: C, cz: 1, cy: h, y: expertY,
                    xL: expertX, zM: 0,
                    access: { src: expert.fcLayer.weight, x: [1, 0, 0], y: [0, 1, 0], scale: 1.0 },
                    dimX: DimStyle.C, dimY: DimStyle.None,
                    name: `Chuy\u00ean gia ${i} W1`,
                    small: true,
                    hidden: detailLevel < 2,
                });

                // Color coding for Top-K routing visualization
                // GPT-4 uses Top-2: only first 2 experts typically active
                const topK = shape.expertsActive || 2;
                const isLikelyActive = i < topK;

                // Apply visual distinction: active (highlighted) vs inactive (dimmed)
                expFcWeight.highlight = isLikelyActive ? 0.3 : 0;
                expFcWeight.opacity = isLikelyActive ? 1.0 : 0.5;

                // ... other expert parts ...
                // For the sake of the plan, let's just show the Weights and the Output

                let expOut = mk({
                    t: 'i', cx: T, cz: B, cy: C, y: expertY + h * cell + margin,
                    xL: expertX, zM: 0,
                    access: { src: expert.projLayer.output, x: [0, 1, 0], y: [1, 0, T] },
                    // deps?
                    dimX: DimStyle.T, dimY: DimStyle.C,
                    name: `Chuy\u00ean gia ${i} \u0110\u1ea7u ra`,
                });

                // Apply same color coding to output
                expOut.highlight = isLikelyActive ? 0.3 : 0;
                expOut.opacity = isLikelyActive ? 1.0 : 0.5;

                expertsCubes.push(expFcWeight, expOut);
                expertOutputs.push(expOut);
            }

            // Update Y to be after the grid (2 rows)
            y += Math.ceil(moeBlock.experts.length / 4) * (C * cell * 2 + margin * 4);

            mlpCubes.push(...expertsCubes);
            mlpLabelCubes.push(...expertsCubes);

            // Simplified aggregation (sum of experts * gate)
            // Real MoE is sparse, but here we might just sum them for viz

            mlpResidual = mk({
                t: 'i', cx: T, cz: B, cy: C, y: y,
                xM: 0, zM: 0,
                access: { src: moeBlock.output, x: [0, 1, 0], y: [1, 0, T] },
                deps: { add: [[attnResidual, 'xy'], ...expertOutputs.map(e => [e, 'xy'] as [IBlkDef, string])] },
                dimX: DimStyle.T, dimY: DimStyle.C,
                name: 'MoE (C\u1ed9ng d\u1ed3n)',
            });

        } else {
            // Standard MLP
            mlpFcWeight = mk({
                t: 'w', cx: C * 4, cz: 1, cy: C, y: y,
                xR: attnLeftX, zM: 0,
                access: { src: target?.mlp.fcLayer.weight, x: [0, 1, 0], y: [1, 0, 0], scale: C * 0.5 },
                dimX: DimStyle.C4, dimY: DimStyle.C,
                name: 'Trọng số MLP',
                hidden: detailLevel < 2,
            });

            mlpFcBias = mk({
                t: 'w', cx: C * 4, cz: 1, cy: 1, y: y - 1 * cell - margin,
                xR: attnLeftX, zM: 0,
                access: { src: target?.mlp.fcLayer.bias!, x: [0, 1, 0], y: [1, 0, 0], scale: C * 0.5 },
                dimX: DimStyle.C4, dimY: DimStyle.None,
                name: 'Bias MLP', small: true,
                hidden: detailLevel < 2 || shape.noBias,
            });

            y += C * cell + margin;

            mlpFc = mk({
                t: 'i', cx: C * 4, cz: B, cy: T, y: y,
                xR: attnLeftX, zM: 0,
                access: { src: target?.mlp.fcLayer.output, x: [1, 0, 0], y: [0, 1, T], scale: 1.0 },
                deps: { dot: [[mlpFcWeight, 'xi'], [ln2.lnResid, 'yi']], dotLen: C, add: [[mlpFcBias, 'x']] },
                dimX: DimStyle.C4, dimY: DimStyle.T,
                name: 'MLP',
                transpose: true,
            });

            y += T * cell + margin;

            mlpAct = mk({
                t: 'i', cx: C * 4, cz: B, cy: T, y: y,
                xR: attnLeftX, zM: 0,
                access: { src: target?.mlp.mlpGelu, x: [1, 0, 0], y: [0, 1, T], scale: 1.0 },
                deps: { add: [[mlpFc, 'xy']], special: BlKDepSpecial.Gelu },
                dimX: DimStyle.C4, dimY: DimStyle.T,
                name: shape.isSquareReLU ? 'Square ReLU' : 'GELU (Kich hoat)',
                transpose: true,
            });

            y += T * cell + margin;

            mlpProjWeight = mk({
                t: 'w', cx: C * 4, cz: 1, cy: C, y: y,
                xR: attnLeftX, zM: 0,
                access: { src: target?.mlp.projLayer.weight, x: [1, 0, 0], y: [0, 1, 0], scale: C * 0.5 },
                dimX: DimStyle.C4, dimY: DimStyle.C,
                name: 'Trọng số Chiếu MLP',
                hidden: detailLevel < 2,
            });

            mlpProjBias = mk({
                t: 'w', cx: 1, cz: 1, cy: C, y: y,
                xR: attnLeftX - C * 4 * cell - margin, zM: 0,
                access: { src: target?.mlp.projLayer.bias!, x: [1, 0, 0], y: [0, 1, 0], scale: C * 0.5 },
                dimX: DimStyle.None, dimY: DimStyle.C, small: true,
                name: 'Bias Chieu MLP',
                hidden: detailLevel < 2 || shape.noBias,
            });

            mlpResult = mk({
                t: 'i', cx: T, cz: B, cy: C, y: y,
                xL: attnLeftX + margin, zM: 0,
                access: { src: target?.mlp.projLayer.output, x: [0, 1, 0], y: [1, 0, T] },
                deps: { dot: [[mlpProjWeight, 'iy'], [mlpAct, 'ix']], dotLen: C, add: [[mlpProjBias, '0y']] },
                dimX: DimStyle.T, dimY: DimStyle.C,
                name: 'Ket qua MLP',
            });

            mlpResidual = mk({
                t: 'i', cx: T, cz: B, cy: C, y: y,
                xM: 0, zM: 0,
                access: { src: target?.mlp.output, x: [0, 1, 0], y: [1, 0, T] },
                deps: { add: [[mlpResult, 'xy'], [attnResidual, 'xy']] },
                dimX: DimStyle.T, dimY: DimStyle.C,
                name: 'MLP (Cong don)',
            });

            mlpCubes.push(mlpFcWeight, mlpFcBias, mlpFc, mlpAct, mlpProjWeight, mlpProjBias, mlpResult);
            mlpLabelCubes.push(...mlpCubes);
        }

        mlpCubes.push(mlpResidual); // Common output block

        y += C * cell - margin;

        let blockCubes = [
            ...ln1.cubes,
            ...heads.flatMap(h => h.cubes),
            projWeight,
            projBias,
            attnOut,
            attnResidual,
            ...ln2.cubes,
            ...mlpCubes,
        ]

        let headCubes = [...ln1.cubes, ...heads.flatMap(h => h.cubes)];
        let projCubes = [projWeight, projBias, attnOut, attnResidual];

        let transformerLabel = mkLabel(1.0, blockCubes);
        let selfAttendLabel = mkLabel(1.0, [...headCubes, ...projCubes]);
        let projLabel = mkLabel(1.0, projCubes);
        let mlpLabel = mkLabel(1.0, [...ln2.cubes, ...mlpLabelCubes, mlpResidual]);

        cubes.push(...blockCubes);


        // Actually, I'll use multi_replace to insert declarations and update return.
        return {
            ln1,
            heads,
            labels: [transformerLabel, projLabel, selfAttendLabel, mlpLabel, ...heads.flatMap(h => h.labels)],
            cubes: blockCubes,
            transformerLabel,
            projLabel,
            selfAttendLabel,
            mlpLabel,
            projWeight,
            projBias,
            attnOut,
            attnResidual,
            mlpResidual,
            ln2,
            // Add new fields
            mlpFc, mlpFcWeight, mlpFcBias, mlpAct, mlpProjWeight, mlpProjBias, mlpResult,
        };
    }


    let blockHalfMargin = 2 * margin;

    y += blockHalfMargin;

    let numColumns = 1;
    let blocksPerColumn = 12;
    if (shape.nBlocks > blocksPerColumn) {
        numColumns = Math.ceil(shape.nBlocks / blocksPerColumn);
    }
    let columnWidth = (C * 14) * cell + margin * 2;
    let blockIdxInColumn = 0;
    let blockYTop = y;

    let blocks: ReturnType<typeof createLayer>[] = [];
    let blockSrc = residual0;
    for (let i = 0; i < nBlocks; i++) {
        if (blockIdxInColumn >= blocksPerColumn) {
            blockIdxInColumn = 0;
            y = blockYTop;
            lnLeftX += columnWidth;
            leftX += columnWidth;
            rightX += columnWidth;
        }

        let target = gptGpuModel?.blocks[i];
        y += blockHalfMargin;
        let block = createLayer(blockSrc, target);
        blocks.push(block);
        blockSrc = block.mlpResidual;
        y += blockHalfMargin;
        blockIdxInColumn++;
    }

    y += blockHalfMargin;
    let ln_f = createLn(0, blockSrc, gptGpuModel?.ln_f);

    cubes.push(...ln_f.cubes);

    let logitsTransposed = false;

    let lmHeadWeight: IBlkDef, logits: IBlkDef, logitsAgg1: IBlkDef, logitsAgg2: IBlkDef, logitsSoftmax: IBlkDef;

    if (logitsTransposed) {
        lmHeadWeight = mk({
            t: 'w', cx: vocabSize, cz: 1, cy: C, y: y,
            xR: lnLeftX, zM: 0,
            access: { src: gptGpuModel?.lm_head.weight, x: [0, 1, 0], y: [1, 0, 0], scale: 5.0 },
            dimX: DimStyle.n_vocab, dimY: DimStyle.C,
            name: 'Trọng số LM Head',
        });

        y += C * cell + margin;

        logits = mk({
            t: 'i', cx: vocabSize, cz: B, cy: T, y: y,
            xR: lnLeftX, zM: 0,
            access: { src: gptGpuModel?.lm_head.output, x: [1, 0, 0], y: [0, 1, T] },
            deps: { dot: [[lmHeadWeight, 'xi'], [ln_f.lnResid, 'yi']], dotLen: C },
            dimX: DimStyle.n_vocab, dimY: DimStyle.T,
            name: 'Logits (Diem so)',
        });

        // z += vocabSize * cell + margin;

        logitsAgg1 = mk({
            t: 'a', cx: 1, cz: B, cy: T, y: y,
            xL: lnLeftX + 1.5 * margin, zM: -3 * cell,
            access: { src: gptGpuModel?.softmaxFinal.agg, x: [1, 0, 0], y: [0, 1, T], channel: 'r' },
            deps: { add: [[logits, 'iy']], special: BlKDepSpecial.SoftmaxAggExp },
            dimX: DimStyle.None, dimY: DimStyle.T,
            name: 'Tong hop Softmax',
        });

        logitsAgg2 = mk({
            t: 'a', cx: 1, cz: B, cy: T, y: y,
            xL: lnLeftX + 1.5 * margin + cell, zM: -3 * cell,
            access: { src: gptGpuModel?.softmaxFinal.agg, x: [1, 0, 0], y: [0, 1, T], channel: 'g' },
            deps: { add: [[logits, 'iy']], special: BlKDepSpecial.SoftmaxAggMax },
            dimX: DimStyle.None, dimY: DimStyle.T,
            name: '',
        });

        y += T * cell + margin;

        logitsSoftmax = mk({
            t: 'i', cx: vocabSize, cz: B, cy: T, y: y,
            xR: lnLeftX, zM: 0,
            access: { src: gptGpuModel?.softmaxFinal.output, x: [1, 0, 0], y: [0, 1, T] },
            deps: { add: [[logits, 'xy'], [logitsAgg1, 'iy'], [logitsAgg2, 'iy']], special: BlKDepSpecial.Softmax },
            dimX: DimStyle.n_vocab, dimY: DimStyle.T,
            name: 'Xac suat (Softmax)',
        });

    } else {
        y += C * cell + margin;
        let leftX2 = leftX - T * cell - margin;

        lmHeadWeight = mk({
            t: 'w', cx: C, cy: vocabSize, cz: 1, y: y,
            xR: leftX2, zM: 0,
            access: { src: gptGpuModel?.lm_head.weight, x: [1, 0, 0], y: [0, 1, 0], scale: 5.0 },
            dimX: DimStyle.C, dimY: DimStyle.n_vocab,
            name: 'Trọng số LM Head',
        });


        logits = mk({
            t: 'i', cx: T, cy: vocabSize, cz: B, y: y,
            xR: leftX, zM: 0,
            access: { src: gptGpuModel?.lm_head.output, x: [0, 1, 0], y: [1, 0, T] },
            deps: { dot: [[lmHeadWeight, 'iy'], [ln_f.lnResid, 'xi']], dotLen: C },
            dimX: DimStyle.T, dimY: DimStyle.n_vocab,
            name: 'Logits (Diem so)',
        });

        y += vocabSize * cell + margin;

        logitsAgg2 = mk({
            t: 'a', cx: T, cy: 1, cz: B, y: y,
            xR: leftX, zM: 0,
            access: { src: gptGpuModel?.softmaxFinal.agg, x: [0, 1, 0], y: [1, 0, T], channel: 'g' },
            deps: { add: [[logits, 'xi']], special: BlKDepSpecial.SoftmaxAggMax },
            dimX: DimStyle.T, dimY: DimStyle.None,
            name: 'Tong hop Softmax',
        });

        logitsAgg1 = mk({
            t: 'a', cx: T, cy: 1, cz: B, y: y + cell,
            xR: leftX, zM: 0,
            access: { src: gptGpuModel?.softmaxFinal.agg, x: [0, 1, 0], y: [1, 0, T], channel: 'r' },
            deps: { add: [[logits, 'xi'], [logitsAgg2, 'x0']], special: BlKDepSpecial.SoftmaxAggExp },
            dimX: DimStyle.T, dimY: DimStyle.None,
            name: '',
        });

        y += 2 * cell + margin;

        logitsSoftmax = mk({
            t: 'i', cx: T, cy: vocabSize, cz: B, y: y,
            xR: leftX, zM: 0,
            access: { src: gptGpuModel?.softmaxFinal.output, x: [0, 1, 0], y: [1, 0, T] },
            deps: { add: [[logits, 'xy'], [logitsAgg1, 'xi'], [logitsAgg2, 'xi']], special: BlKDepSpecial.Softmax },
            dimX: DimStyle.T, dimY: DimStyle.n_vocab,
            name: 'Xac suat (Softmax)',
        });

    }

    // let logitsSoftmaxTopN = mk({
    //     t: 'i', cx: T, cz: B, cy: Math.min(32, vocabSize), y: y,
    //     xM: 0, zM: 0,
    // });

    let weightCount = vocabSize * C + T * C +
        nBlocks * ((2 * C + 4 * C * C + C + 3 * C) + // self attn
            (2 * C + 4 * C + 8 * C * C + C)) + 2 * C; // mlp

    // let decoderCount = vocabSize * C; (excluded from the weight count apparently)

    cubes.push(lmHeadWeight, logits, logitsAgg1, logitsAgg2, logitsSoftmax);

    for (let i = 0; i < cubes.length; i++) {
        cubes[i].idx = i;
    }

    return {
        cubes,
        cell,
        margin,
        idxObj,
        tokEmbedObj,
        posEmbedObj,
        residual0,
        ln_f,
        lmHeadWeight,
        logits,
        logitsAgg1,
        logitsAgg2,
        logitsSoftmax,
        embedLabel,
        blocks,
        height: y,
        logitsTransposed,
        model: gptGpuModel,
        labels: [embedLabel, ...blocks.flatMap(b => b.labels)],
        weightCount,
        shape,
        extraSources: {
            idx: gptGpuModel?.inputBuf,
            tokEmbedOut: gptGpuModel?.vocabEmbed.output,
            posEmbedOut: gptGpuModel?.posEmbed.output,
        },
    };
}

