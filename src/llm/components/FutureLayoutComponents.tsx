import React from 'react';
import s from '../FutureLayout.module.css';
import { useProgramState } from '../Sidebar';
import { faExpand, faMagnifyingGlass } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { Vec3 } from '@/src/utils/vector';
import { Mat4f } from '@/src/utils/matrix';

import clsx from 'clsx';

export const TopStatsBar: React.FC = () => {
    let progState = useProgramState();
    let example = progState.examples[progState.currExampleId] ?? progState.mainExample;
    let currShape = example.shape;

    const models = [
        { id: -1, name: 'NANO-GPT', color: 'cyan' },
        { id: 0, name: 'MICRO-GPT', color: 'cyan' },
        { id: 1, name: 'GPT-2 (S)', color: 'cyan' },
        { id: 3, name: 'GPT-3', color: 'amber' },
        { id: 4, name: 'GPT-4', color: 'magenta' },
        { id: 5, name: 'GPT-5 (THINK)', color: 'magenta' },
    ];

    function handleModelClick(id: number) {
        let example = progState.examples[id] ?? progState.mainExample;
        if (!example.enabled) {
            example.enabled = true;
        }
        progState.currExampleId = id;
        if (example.camera) {
            progState.camera.desiredCamera = example.camera;
        }
        progState.markDirty();
    }

    function onExpandClick() {
        let example = progState.examples[progState.currExampleId] ?? progState.mainExample;
        progState.camera.desiredCamera = example.camera;
        progState.markDirty();
    }

    function onMagnifyClick() {
        let example = progState.examples[progState.currExampleId] ?? progState.mainExample;
        let layout = example.layout ?? progState.layout;
        let obj = layout.residual0;
        let modelTarget = new Vec3(obj.x, obj.y, obj.z);
        let modelMtx = progState.camera.modelMtx.mul(Mat4f.fromTranslation(example.offset))

        let center = modelMtx.mulVec3Proj(modelTarget);
        let zoom = progState.currExampleId === -1 ? 0.7 : 4;
        progState.camera.desiredCamera = {
            center, angle: new Vec3(270, 4.5, zoom),
        };
        progState.markDirty();
    }

    return (
        <div className={`${s.topNav} ${s.glassPanel}`}>
            <div className="flex items-center gap-6">
                <div className={s.techTitle}>LLM VISUALIZER v5.0</div>
                <div className="flex gap-2">
                    {models.map((m) => {
                        const isActive = progState.currExampleId === m.id;
                        return (
                            <button
                                key={m.id}
                                onClick={() => handleModelClick(m.id)}
                                className={clsx(
                                    s.modelButtonHUD,
                                    isActive && s.active
                                )}
                            >
                                {m.name}
                            </button>
                        );
                    })}
                    <div className="w-[1px] h-6 bg-white/10 mx-2 self-center" />
                    <button className={s.iconButtonHUD} onClick={onExpandClick} title="Về khung nhìn chuẩn">
                        <FontAwesomeIcon icon={faExpand} />
                    </button>
                    <button className={s.iconButtonHUD} onClick={onMagnifyClick} title="Phóng to chi tiết">
                        <FontAwesomeIcon icon={faMagnifyingGlass} />
                    </button>
                </div>
            </div>
            <div className="flex gap-8">
                <div className="flex flex-col">
                    <span className={s.metricLabel}>CONTEXT WINDOW</span>
                    <span className={s.metricValue}>{currShape.T >= 1000 ? (currShape.T / 1000).toFixed(0) + 'K' : currShape.T} TOKENS</span>
                </div>
                <div className="flex flex-col">
                    <span className={s.metricLabel}>PARAMS</span>
                    <span className={s.metricValue}>
                        {progState.currExampleId === -1 ? '85K' :
                            progState.currExampleId === 0 ? '4K' :
                                progState.currExampleId === 1 ? '124M' :
                                    progState.currExampleId === 3 ? '175B' :
                                        progState.currExampleId === 4 ? '1.8T' : '10T+'}
                    </span>
                </div>
            </div>
        </div>
    );
};

