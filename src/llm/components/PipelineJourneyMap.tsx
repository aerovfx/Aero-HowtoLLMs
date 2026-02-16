import React, { useState } from 'react';
import { PipelineStage } from '../VisualEngine';

interface StageNode {
    id: PipelineStage;
    label: string;
    description: string;
    posX: number;
    posY: number;
}

const stages: StageNode[] = [
    { id: PipelineStage.DataCollection, label: 'Thu Thập Dữ Liệu', description: 'Crawl Internet, Sách, Code...', posX: 100, posY: 100 },
    { id: PipelineStage.PreProcessing, label: 'Tiền Xử Lý', description: 'Lọc nhiễu, Tokenization, Đóng gói', posX: 250, posY: 150 },
    { id: PipelineStage.ArchitectureDesign, label: 'Thiết Kế Kiến Trúc', description: 'Xây dựng Transformer, MoE, Attention', posX: 400, posY: 100 },
    { id: PipelineStage.PreTraining, label: 'Tiền Huấn Luyện', description: 'Học ngôn ngữ trên 10,000 GPU', posX: 550, posY: 150 },
    { id: PipelineStage.FineTuning, label: 'Tinh Chỉnh (SFT)', description: 'Dạy mô hình làm theo chỉ dẫn', posX: 700, posY: 100 },
    { id: PipelineStage.RLHF, label: 'Căn Chỉnh (RLHF)', description: 'Tối ưu hóa theo phản hồi con người', posX: 850, posY: 150 },
    { id: PipelineStage.Evaluation, label: 'Đánh Giá', description: 'Kiểm tra độ thông minh và an toàn', posX: 1000, posY: 100 },
];

export const PipelineJourneyMap: React.FC<{ onStageSelect: (stage: PipelineStage) => void }> = ({ onStageSelect }) => {
    const [hoveredStage, setHoveredStage] = useState<PipelineStage | null>(null);

    return (
        <div className="pipeline-journey-map" style={{ width: '100%', height: '300px', background: 'rgba(42, 114, 163, 0.1)', backdropFilter: 'blur(12px)', padding: '20px', borderRadius: '16px', border: '1px solid rgba(198, 216, 230, 0.2)', overflowX: 'auto' }}>
            <svg width="1100" height="250" viewBox="0 0 1100 250">
                {/* Đường nối giữa các nodes */}
                <path
                    d={`M 100 100 L 250 150 L 400 100 L 550 150 L 700 100 L 850 150 L 1000 100`}
                    fill="none"
                    stroke="rgba(198, 216, 230, 0.3)"
                    strokeWidth="3"
                    strokeDasharray="10 6"
                    className="flowing-path"
                />

                {/* Các Nodes Stages */}
                {stages.map((stage) => (
                    <g
                        key={stage.id}
                        style={{ cursor: 'pointer' }}
                        onMouseEnter={() => setHoveredStage(stage.id)}
                        onMouseLeave={() => setHoveredStage(null)}
                        onClick={() => onStageSelect(stage.id)}
                    >
                        <circle
                            cx={stage.posX}
                            cy={stage.posY}
                            r={hoveredStage === stage.id ? 14 : 9}
                            fill={hoveredStage === stage.id ? '#f97279' : '#2a72a3'}
                            stroke="#c6d8e6"
                            strokeWidth={hoveredStage === stage.id ? 2 : 0}
                            style={{ transition: 'all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)' }}
                        />
                        <text
                            x={stage.posX}
                            y={stage.posY - 30}
                            textAnchor="middle"
                            fill="#c6d8e6"
                            style={{ fontSize: '13px', fontWeight: '600', fontFamily: 'Inter, sans-serif', letterSpacing: '0.5px' }}
                        >
                            {stage.label}
                        </text>
                        {hoveredStage === stage.id && (
                            <text
                                x={stage.posX}
                                y={stage.posY + 45}
                                textAnchor="middle"
                                fill="rgba(198, 216, 230, 0.7)"
                                style={{ fontSize: '11px', fontFamily: 'Inter, sans-serif' }}
                            >
                                {stage.description}
                            </text>
                        )}
                    </g>
                ))}
            </svg>

            <style jsx>{`
                .flowing-path {
                    stroke-dashoffset: 1000;
                    animation: dash 60s linear infinite;
                }
                @keyframes dash {
                    to {
                        stroke-dashoffset: 0;
                    }
                }
            `}</style>
        </div>
    );
};
