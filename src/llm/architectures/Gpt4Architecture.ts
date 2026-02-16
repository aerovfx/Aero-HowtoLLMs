/**
 * GPT-4 Architecture Specification
 * Based on publicly available information and research papers
 */

import { defineArchitecture, ModelType, RoutingStrategy, FusionMechanism } from './BaseArchitecture';

/**
 * GPT-4 - Mixture of Experts Architecture
 * 
 * Key features:
 * - 120 layers with MoE (Mixture of Experts)
 * - 8 experts per MoE layer
 * - Top-2 expert routing
 * - ~1.76T total parameters
 */
export const GPT4_ARCHITECTURE = defineArchitecture({
    name: 'GPT-4',
    version: '1.0',
    type: ModelType.DecoderOnly,
    description: 'Large-scale Mixture of Experts language model by OpenAI',

    architecture: {
        layers: 120,
        hiddenSize: 12288,          // d_model
        attentionHeads: 96,
        kvHeads: 96,                // Full attention (not GQA)
        contextWindow: 8192,
        maxContextWindow: 128000,   // GPT-4 Turbo
        vocabularySize: 100277,

        intermediateSize: 49152,    // 4 * hidden_size
        normEpsilon: 1e-5,

        // MoE configuration
        expertsPerLayer: 8,
        expertsActive: 2,           // Top-2 routing
        routingStrategy: RoutingStrategy.Learned,
        expertCapacity: 1.25,       // Capacity factor
    },

    visualization: {
        primaryColor: '#10a37f',    // OpenAI green
        secondaryColor: '#6e6e80',
        layerSpacing: 80,
        layerHeight: 40,
        expertLayout: 'grid',       // 2x4 grid for 8 experts
        showWeights: true,
        showBiases: false,
        defaultDetailLevel: 1,      // Start with medium detail
    },

    training: {
        parameters: '1.76T',
        dataTokens: '~13T tokens',
        trainedBy: 'OpenAI',
        releaseDate: '2023-03',
    },
});

/**
 * GPT-4 Turbo - Extended context variant
 */
export const GPT4_TURBO_ARCHITECTURE = defineArchitecture({
    name: 'GPT-4 Turbo',
    version: '1.0',
    type: ModelType.DecoderOnly,
    description: 'Extended context version of GPT-4 (128K tokens)',

    architecture: {
        ...GPT4_ARCHITECTURE.architecture,
        contextWindow: 128000,      // Main difference
    },

    visualization: {
        ...GPT4_ARCHITECTURE.visualization,
        primaryColor: '#0084ff',    // Turbo blue
    },

    training: {
        parameters: '1.76T',
        dataTokens: '~13T tokens',
        trainedBy: 'OpenAI',
        releaseDate: '2023-11',
    },
});

/**
 * GPT-4 Vision - Multimodal variant
 */
export const GPT4_VISION_ARCHITECTURE = defineArchitecture({
    name: 'GPT-4 Vision',
    version: '1.0',
    type: ModelType.Multimodal,
    description: 'Multimodal GPT-4 with vision capabilities',

    architecture: {
        ...GPT4_ARCHITECTURE.architecture,

        // Multimodal features
        modalities: ['text', 'vision'],
        fusionMechanism: FusionMechanism.Mid,     // Fuse in middle layers
        visionLayers: 24,           // Separate vision encoder
    },

    visualization: {
        ...GPT4_ARCHITECTURE.visualization,
        primaryColor: '#8b5cf6',    // Purple for multimodal
    },

    training: {
        parameters: '1.76T',
        dataTokens: '~13T tokens',
        trainedBy: 'OpenAI',
        releaseDate: '2023-09',
    },
});
