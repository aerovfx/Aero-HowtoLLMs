/**
 * Base Architecture Interface
 * Defines common structure for all LLM architectures
 */

export enum ModelType {
    DecoderOnly = 'decoder-only',
    EncoderDecoder = 'encoder-decoder',
    Multimodal = 'multimodal',
}

export enum RoutingStrategy {
    TopK = 'top-k',
    Learned = 'learned',
    Sparse = 'sparse',
}

export enum FusionMechanism {
    Early = 'early',      // Fuse before transformer
    Mid = 'mid',          // Fuse in middle layers
    Late = 'late',        // Fuse after transformer
}

export type Modality = 'text' | 'vision' | 'audio';

/**
 * Core architecture specification
 */
export interface IArchitectureSpec {
    // Basic info
    name: string;
    version: string;
    type: ModelType;
    description?: string;

    // Architecture parameters
    architecture: {
        layers: number;
        hiddenSize: number;
        attentionHeads: number;
        contextWindow: number;
        vocabularySize: number;

        // Advanced features (optional)
        kvHeads?: number;              // For Grouped Query Attention
        intermediateSize?: number;      // MLP hidden size
        normEpsilon?: number;

        // MoE specific (GPT-4, Mixtral)
        expertsPerLayer?: number;
        expertsActive?: number;        // Top-K experts
        routingStrategy?: RoutingStrategy;
        expertCapacity?: number;

        // Multimodal specific (Gemini, GPT-4V)
        modalities?: Modality[];
        fusionMechanism?: FusionMechanism;
        visionLayers?: number;
        audioLayers?: number;

        // Long context (Claude, Gemini)
        maxContextWindow?: number;
        slidingWindow?: number;
    };

    // Visualization configuration
    visualization: {
        primaryColor: string;
        secondaryColor?: string;
        layerSpacing: number;
        layerHeight: number;
        expertLayout?: 'grid' | 'circular' | 'linear';
        showWeights?: boolean;
        showBiases?: boolean;
        defaultDetailLevel?: number;
    };

    // Training info (optional, for display)
    training?: {
        parameters: string;          // e.g., "1.76T"
        dataTokens?: string;         // e.g., "13T tokens"
        trainedBy?: string;
        releaseDate?: string;
    };
}

/**
 * Registry of all supported models
 */
export class ModelRegistry {
    private static models: Map<string, IArchitectureSpec> = new Map();

    static register(spec: IArchitectureSpec): void {
        this.models.set(spec.name, spec);
    }

    static get(name: string): IArchitectureSpec | undefined {
        return this.models.get(name);
    }

    static getAll(): IArchitectureSpec[] {
        return Array.from(this.models.values());
    }

    static getByType(type: ModelType): IArchitectureSpec[] {
        return this.getAll().filter(m => m.type === type);
    }
}

/**
 * Helper to create architecture specs
 */
export function defineArchitecture(spec: IArchitectureSpec): IArchitectureSpec {
    ModelRegistry.register(spec);
    return spec;
}
