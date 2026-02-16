# LLM Architecture System

This directory contains architecture specifications for various Large Language Models (LLMs) supported by the visualization tool.

## Structure

```
architectures/
├── BaseArchitecture.ts      # Core interfaces and registry
├── Gpt4Architecture.ts      # GPT-4 specifications
├── ClaudeArchitecture.ts    # Claude specifications (TODO)
├── GeminiArchitecture.ts    # Gemini specifications (TODO)
└── index.ts                 # Central exports
```

## Adding a New Architecture

1. **Create specification file:**
   ```typescript
   // MyModelArchitecture.ts
   import { defineArchitecture, ModelType } from './BaseArchitecture';
   
   export const MY_MODEL_ARCHITECTURE = defineArchitecture({
       name: 'My Model',
       version: '1.0',
       type: ModelType.DecoderOnly,
       
       architecture: {
           layers: 32,
           hiddenSize: 4096,
           // ... other params
       },
       
       visualization: {
           primaryColor: '#ff0000',
           // ... viz config
       },
   });
   ```

2. **Export in index.ts:**
   ```typescript
   export * from './MyModelArchitecture';
   ```

3. **Use in visualization:**
   ```typescript
   import { getModel } from './architectures';
   
   const spec = getModel('My Model');
   // Use spec to configure visualization
   ```

## Architecture Parameters

### Required Fields

- `name`: Unique model identifier
- `version`: Model version
- `type`: ModelType enum value
- `architecture.layers`: Number of transformer layers
- `architecture.hiddenSize`: Hidden dimension size
- `architecture.attentionHeads`: Number of attention heads
- `architecture.contextWindow`: Maximum context length
- `architecture.vocabularySize`: Size of token vocabulary
- `visualization.primaryColor`: Main color for this model
- `visualization.layerSpacing`: Space between layers (px)
- `visualization.layerHeight`: Height of each layer (px)

### Optional Fields

#### MoE (Mixture of Experts)
- `expertsPerLayer`: Number of experts per layer
- `expertsActive`: How many experts are active (top-k)
- `routingStrategy`: How experts are selected
- `expertCapacity`: Capacity factor for load balancing

#### Multimodal
- `modalities`: Array of supported modalities
- `fusionMechanism`: When to fuse modalities
- `visionLayers`: Number of vision encoder layers
- `audioLayers`: Number of audio encoder layers

#### Advanced
- `kvHeads`: KV heads for Grouped Query Attention
- `intermediateSize`: MLP hidden size
- `maxContextWindow`: Extended context (if different from base)
- `slidingWindow`: Sliding window size for attention

## Visualization Configuration

### Colors
Use hex colors for consistency:
- **GPT-4**: `#10a37f` (OpenAI green)
- **Claude**: `#cc785c` (Anthropic orange)
- **Gemini**: `#4285f4` (Google blue)

### Layout
- `layerSpacing`: Vertical space between layers (60-100px recommended)
- `layerHeight`: Height of each layer visualization (30-50px)
- `expertLayout`: For MoE models ('grid', 'circular', 'linear')
- `defaultDetailLevel`: 0 (low), 1 (medium), 2 (high)

## Examples

### Simple Decoder-Only Model
```typescript
export const SIMPLE_MODEL = defineArchitecture({
    name: 'Simple GPT',
    version: '1.0',
    type: ModelType.DecoderOnly,
    architecture: {
        layers: 12,
        hiddenSize: 768,
        attentionHeads: 12,
        contextWindow: 2048,
        vocabularySize: 50257,
    },
    visualization: {
        primaryColor: '#3b82f6',
        layerSpacing: 60,
        layerHeight: 35,
    },
});
```

### MoE Model (GPT-4 style)
```typescript
export const MoE_MODEL = defineArchitecture({
    name: 'MoE Example',
    version: '1.0',
    type: ModelType.DecoderOnly,
    architecture: {
        layers: 32,
        hiddenSize: 4096,
        attentionHeads: 32,
        contextWindow: 8192,
        vocabularySize: 100000,
        
        // MoE specific
        expertsPerLayer: 8,
        expertsActive: 2,
        routingStrategy: RoutingStrategy.TopK,
    },
    visualization: {
        primaryColor: '#8b5cf6',
        layerSpacing: 80,
        layerHeight: 40,
        expertLayout: 'grid',
    },
});
```

### Multimodal Model
```typescript
export const MULTIMODAL_MODEL = defineArchitecture({
    name: 'Vision-Language Model',
    version: '1.0',
    type: ModelType.Multimodal,
    architecture: {
        layers: 24,
        hiddenSize: 1024,
        attentionHeads: 16,
        contextWindow: 4096,
        vocabularySize: 50000,
        
        // Multimodal specific
        modalities: ['text', 'vision'],
        fusionMechanism: FusionMechanism.Mid,
        visionLayers: 12,
    },
    visualization: {
        primaryColor: '#ec4899',
        layerSpacing: 70,
        layerHeight: 38,
    },
});
```

## Model Registry API

```typescript
// Get all models
const allModels = ModelRegistry.getAll();

// Get by name
const gpt4 = ModelRegistry.get('GPT-4');

// Get by type
const multimodalModels = ModelRegistry.getByType(ModelType.Multimodal);

// Check if model exists
if (ModelRegistry.get('My Model')) {
    // Model is registered
}
```

## Future Enhancements

- [ ] Model comparison utilities
- [ ] Architecture diff viewer
- [ ] Parameter count calculator
- [ ] Memory footprint estimator
- [ ] FLOPS calculator
- [ ] JSON export/import for architectures
- [ ] Validation schema

## References

- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- [Mixtral MoE Paper](https://arxiv.org/abs/2401.04088)
- [Gemini Technical Report](https://arxiv.org/abs/2312.11805)
