/**
 * LLM Visualization Engine (Vanilla WebGL)
 * Không sử dụng Three.js - Tập trung vào hiệu năng và kiểm soát tuyệt đối.
 */

export interface ITransform {
    posX: number;
    posY: number;
    posZ: number;
    rotX: number;
    rotY: number;
    rotZ: number;
    scale: number;
}

export enum PipelineStage {
    DataCollection,
    PreProcessing,
    ArchitectureDesign,
    PreTraining,
    FineTuning,
    RLHF,
    Evaluation,
}

/**
 * Quản lý trạng thái toàn cục của hệ thống Visualization
 */
export interface IVisualState {
    currentStage: PipelineStage;
    camera: ITransform;
    interaction: {
        hoveredId: string | null;
        selectedId: string | null;
        cursorPos: { x: number, y: number };
    };
    settings: {
        showGrid: boolean;
        showLabels: boolean;
        particleDensity: number;
        bloomIntensity: number;
    };
}

/**
 * Định nghĩa một thành phần trực quan (Visual Entity)
 */
export interface IVisualEntity {
    id: string;
    type: 'cube' | 'particle' | 'line' | 'mesh';
    transform: ITransform;
    color: { r: number, g: number, b: number, a: number };
    metadata: any;
}

/**
 * Shader cơ bản cho LLM Viz (GLSL)
 */
export const VIZ_SHADERS = {
    vertex: `
        attribute vec3 position;
        attribute vec4 color;
        uniform mat4 u_viewMatrix;
        uniform mat4 u_projectionMatrix;
        uniform mat4 u_modelMatrix;
        varying vec4 v_color;

        void main() {
            v_color = color;
            gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * vec4(position, 1.0);
            gl_PointSize = 2.0;
        }
    `,
    f18_ragment: `
        precision mediump float;
        varying vec4 v_color;
        uniform float u_opacity;

        void main() {
            gl_F18_ragColor = vec4(v_color.rgb, v_color.a * u_opacity);
        }
    `
};

/**
 * Manager điều phối các màn chơi (Stages)
 */
export class StageManager {
    private currentState: IVisualState;

    constructor(initialState: IVisualState) {
        this.currentState = initialState;
    }

    public setStage(stage: PipelineStage) {
        this.currentState.currentStage = stage;
        this.onStageChange(stage);
    }

    private onStageChange(stage: PipelineStage) {
        // Logic chuyển cảnh: Di chuyển camera, load assets, đổi walkthrough
        console.log(`Transitioning to stage: ${PipelineStage[stage]}`);
    }

    public update() {
        // Cập nhật logic từng frame (physics, animations)
    }

    public render(gl: WebGLRenderingContext) {
        // Vẽ toàn bộ scene
    }
}
