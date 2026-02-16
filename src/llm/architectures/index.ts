/**
 * LLM Architecture Registry
 * Central export point for all supported architectures
 */

// Base types and interfaces
export * from './BaseArchitecture';

// Specific architectures
export * from './Gpt4Architecture';

// Re-export for convenience
import { ModelRegistry } from './BaseArchitecture';
import { GPT4_ARCHITECTURE, GPT4_TURBO_ARCHITECTURE, GPT4_VISION_ARCHITECTURE } from './Gpt4Architecture';

// Auto-register all architectures on import
// (already registered via defineArchitecture in each file)

/**
 * Get all registered models
 */
export function getAllModels() {
    return ModelRegistry.getAll();
}

/**
 * Get model by name
 */
export function getModel(name: string) {
    return ModelRegistry.get(name);
}

/**
 * Available model names
 */
export const AVAILABLE_MODELS = {
    GPT4: 'GPT-4',
    GPT4_TURBO: 'GPT-4 Turbo',
    GPT4_VISION: 'GPT-4 Vision',
    // Add more as they're implemented
} as const;
