import React, { useState, useEffect, useRef } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import html2canvas from 'html2canvas';
import { LLM_FORMULAS, LatexFormula } from '../../utils/latex-formula-generator';

interface HybridMathRendererProps {
    formulaKey: string;
    mode?: 'simple' | 'billboard' | 'tooltip' | 'both' | 'all';
    interactive?: boolean;
    className?: string;
    style?: React.CSSProperties;
}

/**
 * Hybrid Math Renderer Component
 * Implements the 3-layer approach: 3D (ASCII), Billboard (Texture), and Tooltip (Full LaTeX)
 */
export const HybridMathRenderer: React.FC<HybridMathRendererProps> = ({
    formulaKey,
    mode = 'all',
    interactive = true,
    className,
    style
}) => {
    const formula = LLM_FORMULAS[formulaKey];
    const [showTooltip, setShowTooltip] = useState(false);
    const [billboardUrl, setBillboardUrl] = useState<string | null>(null);
    const billboardRef = useRef<HTMLDivElement>(null);

    if (!formula) return <div>Formula {formulaKey} not found</div>;

    // Phase 1: Simple ASCII for 3D/Fast rendering
    const renderSimple = () => (
        <code style={{
            fontFamily: 'monospace',
            color: '#c6d8e6',
            background: 'rgba(0,0,0,0.3)',
            padding: '2px 6px',
            borderRadius: '4px'
        }}>
            {formula.asciiMath || formula.name}
        </code>
    );

    // Phase 2: Tooltip with full KaTeX typography
    const renderTooltip = () => (
        <div style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(10, 26, 42, 0.95)',
            border: '1px solid rgba(42, 114, 163, 0.5)',
            padding: '15px',
            borderRadius: '12px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
            zIndex: 1000,
            width: 'max-content',
            maxWidth: '300px',
            backdropFilter: 'blur(10px)',
            marginBottom: '10px'
        }}>
            <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#f97279' }}>{formula.name}</h4>
            <div style={{ fontSize: '1.2em', marginBottom: '10px' }}>
                <BlockMath math={formula.latex} />
            </div>
            <p style={{ margin: 0, fontSize: '12px', opacity: 0.8, color: '#c6d8e6' }}>{formula.description}</p>
        </div>
    );

    // Phase 3: Billboard (Pre-rendering to texture simulator)
    // Note: In a real three.js scene, we would use the result of html2canvas as a texture
    useEffect(() => {
        if ((mode === 'billboard' || mode === 'all' || mode === 'both') && billboardRef.current) {
            html2canvas(billboardRef.current, {
                backgroundColor: null,
                scale: 2
            }).then(canvas => {
                setBillboardUrl(canvas.toDataURL());
            });
        }
    }, [formulaKey, mode]);

    return (
        <div
            className={`hybrid-math-container ${className || ''}`}
            style={{ position: 'relative', display: 'inline-block', ...style }}
            onMouseEnter={() => interactive && setShowTooltip(true)}
            onMouseLeave={() => interactive && setShowTooltip(false)}
        >
            {/* Layer 1: The Visual representation visible in 3D/Scene */}
            <div style={{ cursor: interactive ? 'help' : 'default' }}>
                {mode === 'simple' && renderSimple()}

                {(mode === 'billboard' || mode === 'all' || mode === 'both') && (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        {/* Hidden source for billboard rendering */}
                        <div ref={billboardRef} style={{
                            position: 'absolute',
                            visibility: 'hidden',
                            pointerEvents: 'none',
                            padding: '10px',
                            background: 'transparent'
                        }}>
                            <div style={{ color: '#fff', fontSize: '24px' }}>
                                <BlockMath math={formula.latex} />
                            </div>
                        </div>

                        {/* The rendered billboard (or fallback) */}
                        {billboardUrl ? (
                            <img src={billboardUrl} alt={formula.name} style={{ maxHeight: '60px', filter: 'drop-shadow(0 0 8px rgba(249, 114, 121, 0.4))' }} />
                        ) : renderSimple()}
                    </div>
                )}
            </div>

            {/* Layer 2: Interactive Tooltip */}
            {(showTooltip || mode === 'tooltip') && renderTooltip()}

            <style jsx>{`
                .hybrid-math-container {
                    transition: all 0.3s ease;
                }
                .hybrid-math-container:hover {
                    transform: scale(1.02);
                }
            `}</style>
        </div>
    );
};
