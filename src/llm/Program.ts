import { genModelViewMatrices, ICamera, ICameraPos, updateCamera } from "./Camera";
import { drawAllArrows } from "./components/Arrow";
import { drawBlockLabels } from "./components/SectionLabels";
import { drawModelCard } from "./components/ModelCard";
import { IGptModelLink, IGpuGptModel, IModelShape } from "./GptModel";
import { genGptModelLayout, IBlkDef, IGptModelLayout } from "./GptModelLayout";
import { drawText, IFontAtlasData, IFontOpts, measureText } from "./render/fontRender";
import { initRender, IRenderState, IRenderView, renderModel, resetRenderBuffers } from "./render/modelRender";
import { beginQueryAndGetPrevMs, endQuery } from "./render/queryManager";
import { SavedState } from "./SavedState";
import { isNotNil } from "@/src/utils/data";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { initWalkthrough, runWalkthrough } from "./walkthrough/Walkthrough";
import { IColorMix } from "./Annotations";
import { Mat4f } from "@/src/utils/matrix";
import { runMouseHitTesting } from "./Interaction";
import { RenderPhase } from "./render/sharedRender";
import { drawBlockInfo } from "./components/BlockInfo";
import { NativeFunctions } from "./NativeBindings";
import { IWasmGptModel, stepWasmModel, syncWasmDataWithJsAndGpu } from "./GptModelWasm";
import { IMovementInfo, manageMovement } from "./components/MovementControls";
import { IBlockRender, initBlockRender } from "./render/blockRender";
import { ILayout } from "../utils/layout";
import { DimStyle } from "./walkthrough/WalkthroughTools";
import { Subscriptions } from "../utils/hooks";

export interface IProgramState {
    native: NativeFunctions | null;
    wasmGptModel: IWasmGptModel | null;
    stepModel: boolean;
    mouse: IMouseState;
    render: IRenderState;
    inWalkthrough: boolean;
    walkthrough: ReturnType<typeof initWalkthrough>;
    camera: ICamera;
    htmlSubs: Subscriptions;
    layout: IGptModelLayout;
    mainExample: IModelExample;
    examples: IModelExample[];
    currExampleId: number;
    shape: IModelShape;
    gptGpuModel: IGpuGptModel | null;
    jsGptModel: IGptModelLink | null;
    movement: IMovementInfo;
    display: IDisplayState;
    pageLayout: ILayout;
    markDirty: () => void;
    detailLevel: number;
}

export interface IModelExample {
    name: string;
    shape: IModelShape;
    enabled: boolean;
    layout?: IGptModelLayout;
    blockRender: IBlockRender;
    offset: Vec3;
    modelCardOffset: Vec3;
    camera?: ICameraPos;
}

export interface IMouseState {
    mousePos: Vec3;
}

export interface IDisplayState {
    tokenColors: IColorMix | null;
    tokenIdxColors: IColorMix | null;
    tokenOutputColors: IColorMix | null;
    tokenIdxModelOpacity?: number[];
    topOutputOpacity?: number;
    lines: string[];
    hoverTarget: IHoverTarget | null;
    blkIdxHover: number[] | null;
    dimHover: DimStyle | null;
}

export interface IHoverTarget {
    subCube: IBlkDef;
    mainCube: IBlkDef;
    mainIdx: Vec3;
}

export function initProgramState(canvasEl: HTMLCanvasElement, fontAtlasData: IFontAtlasData): IProgramState {

    let render = initRender(canvasEl, fontAtlasData);
    let walkthrough = initWalkthrough();

    let prevState = SavedState.state;
    let camera: ICamera = {
        angle: prevState?.camera.angle ?? new Vec3(296, 16, 13.5),
        center: prevState?.camera.center ?? new Vec3(-8.4, 50, -481.5),
        transition: {},
        modelMtx: new Mat4f(),
        viewMtx: new Mat4f(),
        lookAtMtx: new Mat4f(),
        camPos: new Vec3(),
        camPosModel: new Vec3(),
    }

    let shape: IModelShape = {
        B: 1,
        T: 11,
        C: 48,
        nHeads: 3,
        A: 48 / 3,
        nBlocks: 3,
        vocabSize: 3,
    };

    let gpt2ShapeSmall: IModelShape = {
        B: 1,
        T: 1024,
        C: 768,
        nHeads: 12,
        A: 768 / 12,
        nBlocks: 12,
        vocabSize: 50257,
    };

    let gpt2ShapeLarge: IModelShape = {
        B: 1,
        T: 1024,
        C: 1600,
        nHeads: 25,
        A: 1600 / 25,
        nBlocks: 48,
        vocabSize: 50257,
    };

    let gpt3Shape: IModelShape = {
        B: 1,
        T: 1024,
        C: 12288,
        nHeads: 96,
        A: 12288 / 96,
        nBlocks: 96,
        vocabSize: 50257,
    };

    // GPT-4 - MoE Architecture (estimated parameters)
    let gpt4Shape: IModelShape = {
        B: 1,
        T: 8192,            // Context window
        C: 12288,           // Hidden size (same as GPT-3)
        nHeads: 96,
        A: 12288 / 96,
        nBlocks: 120,       // More layers
        vocabSize: 100277,  // Larger vocabulary

        // MoE específico
        expertsPerLayer: 8,  // 8 experts per layer
        expertsActive: 2,    // Top-2 routing
        isMoE: true,
    };

    // GPT-4 Turbo - Extended context
    let gpt4TurboShape: IModelShape = {
        ...gpt4Shape,
        T: 128000,          // 128K context
    };

    // GPT-5 - Router + Thinking Architecture (estimates based on user info)
    let gpt5Shape: IModelShape = {
        B: 1,
        T: 272000,          // 272K input context (400k total via API)
        C: 16384,           // Larger hidden size
        nHeads: 128,
        A: 16384 / 128,
        nBlocks: 160,       // Deep reasoning
        vocabSize: 128000,  // New unified tokenizer

        isMoE: true,
        expertsPerLayer: 16, // More experts for routing
        expertsActive: 4,    // Higher diversity in reasoning
    };
    // microGPT - minimalist educational model (Karpathy)
    let microGptShape: IModelShape = {
        B: 1,
        T: 32,              // Context length
        C: 16,              // Small hidden size
        nHeads: 4,
        A: 16 / 4,
        nBlocks: 1,         // Single layer
        vocabSize: 32,      // Character level
        isRMSNorm: true,    // Special feature of microGPT
        noBias: true,       // Simplified architecture
        isSquareReLU: true, // Use Square ReLU as in microGPT
    };

    function makeCamera(center: Vec3, angle: Vec3): ICameraPos {
        return { center, angle };
    }

    let delta = new Vec3(10000, 0, 0);

    return {
        native: null,
        wasmGptModel: null,
        render: render!,
        inWalkthrough: true,
        walkthrough,
        camera,
        shape: shape,
        layout: genGptModelLayout(shape),
        currExampleId: -1,
        mainExample: {
            name: 'nano-gpt',
            enabled: true,
            shape: shape,
            offset: new Vec3(),
            modelCardOffset: new Vec3(),
            blockRender: null!,
            camera: makeCamera(new Vec3(42.771, 0.000, -569.287), new Vec3(284.959, 26.501, 12.867)),
        },
        examples: [{
            name: 'MICRO-GPT',
            enabled: true,
            shape: microGptShape,
            layout: genGptModelLayout(microGptShape),
            offset: delta.mul(-8.5),
            modelCardOffset: delta.mul(-3.5),
            blockRender: initBlockRender(render?.ctx ?? null),
            camera: makeCamera(new Vec3(-105200.0, 0, -85000.0), new Vec3(285, 15, 1500.0)),
        }, {
            name: 'GPT-2 (small)',
            enabled: true,
            shape: gpt2ShapeSmall,
            layout: genGptModelLayout(gpt2ShapeSmall),
            offset: delta.mul(-5),
            modelCardOffset: delta.mul(-2.0),
            blockRender: initBlockRender(render?.ctx ?? null),
            camera: makeCamera(new Vec3(-65141.321, 0.000, -69843.439), new Vec3(224.459, 24.501, 1574.240)),
        }, {
            name: 'GPT-2 (XL)',
            enabled: true,
            shape: gpt2ShapeLarge,
            layout: genGptModelLayout(gpt2ShapeLarge),
            offset: delta.mul(20),
            modelCardOffset: delta.mul(0.5),
            blockRender: initBlockRender(render?.ctx ?? null),
            camera: makeCamera(new Vec3(237902.688, 0.000, -47282.484), new Vec3(311.959, 23.501, 1382.449)),
        }, {
            name: 'GPT-3',
            enabled: false,
            shape: gpt3Shape,
            layout: genGptModelLayout(gpt3Shape),
            offset: delta.mul(50.0),
            modelCardOffset: delta.mul(15.0),
            blockRender: initBlockRender(render?.ctx ?? null),
            camera: makeCamera(new Vec3(837678.163, 0.000, -485242.286), new Vec3(238.959, 10.501, 12583.939)),
        }, {
            name: 'GPT-4',
            enabled: false,
            shape: gpt4Shape,
            layout: genGptModelLayout(gpt4Shape),
            offset: delta.mul(100.0),
            modelCardOffset: delta.mul(35.0),
            blockRender: initBlockRender(render?.ctx ?? null),
            camera: makeCamera(new Vec3(1500000.0, 0.000, -800000.0), new Vec3(270.0, 15.0, 25000.0)),
        }, {
            name: 'GPT-5 (Thinking)',
            enabled: false,
            shape: gpt5Shape,
            layout: genGptModelLayout(gpt5Shape),
            offset: delta.mul(180.0),
            modelCardOffset: delta.mul(60.0),
            blockRender: initBlockRender(render?.ctx ?? null),
            camera: makeCamera(new Vec3(3500000.0, 0.000, -1800000.0), new Vec3(270.0, 10.0, 45000.0)),
        }],
        gptGpuModel: null,
        jsGptModel: null,
        stepModel: false,
        markDirty: () => { },
        htmlSubs: new Subscriptions(),
        mouse: {
            mousePos: new Vec3(),
        },
        movement: {
            action: null,
            actionHover: null,
            target: [0, 0],
            depth: 1,
            cameraLerp: null,
        },
        display: {
            tokenColors: null,
            tokenIdxColors: null,
            tokenOutputColors: null,
            lines: [],
            hoverTarget: null,
            dimHover: null,
            blkIdxHover: null,
        },
        pageLayout: {
            height: 0,
            width: 0,
            isDesktop: true,
            isPhone: true,
        },
        detailLevel: 2,
    };
}

export function runProgram(view: IRenderView, state: IProgramState) {
    let timer0 = performance.now();

    if (!state.render) {
        return;
    }

    resetRenderBuffers(state.render);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    state.display.lines = [];
    state.display.hoverTarget = null;
    state.display.tokenColors = null;
    state.display.tokenIdxColors = null;

    if (state.wasmGptModel && state.jsGptModel) {
        syncWasmDataWithJsAndGpu(state.wasmGptModel, state.jsGptModel);
    }

    if (state.stepModel && state.wasmGptModel && state.jsGptModel) {
        state.stepModel = false;
        stepWasmModel(state.wasmGptModel, state.jsGptModel);
    }

    // generate the base model, incorporating the gpu-side model if available
    state.layout = genGptModelLayout(state.shape, state.jsGptModel, new Vec3(), state.detailLevel);

    // @TODO: handle different models in the same scene.
    // Maybe need to copy a lot of different things like the entire render state per model?
    for (let example of state.examples) {
        if (example.enabled && !example.layout) {
            let layout = genGptModelLayout(example.shape, null, example.offset, state.detailLevel);
            example.layout = layout;
        }
    }

    // Get the current active layout based on selected model
    let currentExample = state.examples[state.currExampleId] ?? state.mainExample;
    let activeLayout = currentExample.layout ?? state.layout;

    genModelViewMatrices(state, activeLayout!);

    let queryRes = beginQueryAndGetPrevMs(state.render.queryManager, 'render');
    if (isNotNil(queryRes)) {
        state.render.lastGpuMs = queryRes;
    }

    state.render.renderTiming = false; // state.pageLayout.isDesktop;

    // will modify layout; view; render a few things.
    if (state.inWalkthrough) {
        runWalkthrough(state, view);
    }

    updateCamera(state, view);

    drawBlockInfo(state);
    // these will get modified by the walkthrough (stored where?)
    drawAllArrows(state.render, activeLayout);

    drawModelCard(state, state.layout, 'nano-gpt', new Vec3());
    // drawTokens(state.render, state.layout, state.display);

    for (let example of state.examples) {
        if (example.enabled && example.layout) {
            drawModelCard(state, example.layout, example.name, example.offset.add(example.modelCardOffset));
        }
    }

    // manageMovement(state, view);
    runMouseHitTesting(state);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    drawBlockLabels(state.render, activeLayout);

    let lineNo = 1;
    let tw = state.render.size.x;
    state.render.sharedRender.activePhase = RenderPhase.Overlay2D;
    for (let line of state.display.lines) {
        let opts: IFontOpts = { color: new Vec4(), size: 14 };
        let w = measureText(state.render.modelFontBuf, line, opts);
        drawText(state.render.modelFontBuf, line, tw - w - 4, lineNo * opts.size * 1.3 + 4, opts)
        lineNo++;
    }

    // render everything; i.e. here's where we actually do gl draw calls
    // up until now, we've just been putting data in cpu-side buffers
    renderModel(state);

    endQuery(state.render.queryManager, 'render');
    state.render.gl.flush();

    state.render.lastJsMs = performance.now() - timer0;
}
