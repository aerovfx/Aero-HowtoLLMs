import React, { useState } from 'react';

interface BlockComponentProps {
    label: string;
    details: string;
    color: string;
    delay: number;
}

const TransformerLayer: React.FC<{ index: number, isActive: boolean }> = ({ index, isActive }) => {
    return (
        <div className={`transformer-layer ${isActive ? 'active' : ''}`} style={{
            transform: `translateZ(${index * 60}px)`,
            transitionDelay: `${index * 0.1}s`
        }}>
            <div className="layer-frame">
                <span className="layer-label">L\u1edbp {index}</span>
                <div className="sub-components">
                    <div className="comp attention">Attn</div>
                    <div className="comp ffn">FFN</div>
                </div>
            </div>
            <style jsx>{`
                .transformer-layer {
                    position: absolute;
                    width: 300px;
                    height: 100px;
                    border: 1px solid rgba(198, 216, 230, 0.3);
                    background: rgba(42, 114, 163, 0.1);
                    backdrop-filter: blur(4px);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.6s cubic-bezier(0.23, 1, 0.32, 1);
                    transform-style: preserve-3d;
                }
                .transformer-layer.active {
                    background: rgba(249, 114, 121, 0.2);
                    border-color: #f97279;
                    box-shadow: 0 0 20px rgba(249, 114, 121, 0.3);
                }
                .layer-frame {
                    padding: 15px;
                    text-align: center;
                    width: 100%;
                }
                .layer-label {
                    color: #c6d8e6;
                    font-size: 12px;
                    font-family: 'Inter', sans-serif;
                    opacity: 0.7;
                }
                .sub-components {
                    display: flex;
                    gap: 10px;
                    margin-top: 10px;
                    justify-content: center;
                }
                .comp {
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 10px;
                    font-weight: bold;
                    color: #fff;
                }
                .attention { background: #2a72a3; }
                .ffn { background: #f97279; }
            `}</style>
        </div>
    );
};

export const ArchitectureConstruction: React.FC = () => {
    const [layers, setLayers] = useState(6);
    const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);

    return (
        <div className="arch-construction" style={{
            padding: '40px',
            background: 'rgba(255, 255, 255, 0.02)',
            borderRadius: '16px',
            border: '1px solid rgba(198, 216, 230, 0.1)',
            overflow: 'hidden'
        }}>
            <div className="header" style={{ marginBottom: '30px' }}>
                <h3 style={{ color: '#c6d8e6', marginBottom: '10px' }}>🏗️ Thiết Kế Kiến Trúc: Skyscraper Mode</h3>
                <p style={{ color: 'rgba(198, 216, 230, 0.6)', fontSize: '14px' }}>
                    Transformer được xây dựng bằng cách chồng các lớp xử lý lên nhau. GPT-4 có khoảng 120 lớp như thế này.
                </p>
                <div style={{ marginTop: '15px' }}>
                    <label style={{ color: '#f97279', fontSize: '12px', marginRight: '10px' }}>Số lượng lớp: {layers}</label>
                    <input
                        type="range" min="1" max="12" value={layers}
                        onChange={(e) => setLayers(parseInt(e.target.value))}
                        style={{ accentColor: '#f97279', verticalAlign: 'middle' }}
                    />
                </div>
            </div>

            <div className="scene-container">
                <div className="building-scene">
                    {/* Skyscraper tiers */}
                    {[...Array(layers)].map((_, i) => (
                        <div
                            key={i}
                            onMouseEnter={() => setHoveredLayer(i)}
                            onMouseLeave={() => setHoveredLayer(null)}
                            style={{ position: 'absolute', width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}
                        >
                            <TransformerLayer index={i} isActive={hoveredLayer === i} />
                        </div>
                    ))}

                    {/* Ground Layer (Embedding) */}
                    <div className="base-layer" style={{ transform: 'translateZ(-40px) translateY(80px) rotateX(90px)' }}>
                        Nh\u00fang \u0111\u1ea7u v\u00e0o (Input Embedding)
                    </div>
                </div>
            </div>

            <style jsx>{`
                .scene-container {
                    height: 500px;
                    perspective: 1200px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: radial-gradient(circle at center, rgba(42, 114, 163, 0.1) 0%, transparent 70%);
                }
                .building-scene {
                    position: relative;
                    width: 300px;
                    height: 100px;
                    transform-style: preserve-3d;
                    transform: rotateX(65deg) rotateZ(-25deg);
                    animation: subtleRotate 20s linear infinite;
                }
                .base-layer {
                    position: absolute;
                    width: 320px;
                    height: 320px;
                    background: repeating-linear-gradient(45deg, rgba(198, 216, 230, 0.05), rgba(198, 216, 230, 0.05) 10px, transparent 10px, transparent 20px);
                    border: 2px solid #2a72a3;
                    color: #2a72a3;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 800;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    font-size: 10px;
                }
                @keyframes subtleRotate {
                    0% { transform: rotateX(65deg) rotateZ(0deg); }
                    100% { transform: rotateX(65deg) rotateZ(360deg); }
                }
            `}</style>
        </div>
    );
};
