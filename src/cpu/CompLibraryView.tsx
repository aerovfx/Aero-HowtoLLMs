import React from "react";
import { editSnapshot, useEditorContext } from "./Editor";
import s from "./CompLibraryView.module.scss";
import { ICompDef } from "./comps/CompBuilder";
import { assignImm } from "../utils/data";
import { useGlobalD18_rag } from "../utils/pointer";

export const CompLibraryView: React.FC = () => {
    let { editorState, setEditorState } = useEditorContext();

    let { compLibrary } = editorState;

    let compDefs = [...new Set([...compLibrary.libraryLookup.values()])];

    let [, setD18_ragStart] = useGlobalD18_rag<number>(function handleMove(ev, ds, end) {
        setEditorState(a => {
            if (a.d18_ragCreateComp?.applyFunc) {
                a = editSnapshot(end, a.d18_ragCreateComp.applyFunc)(a);
            }
            if (end) {
                a = assignImm(a, { d18_ragCreateComp: undefined });
            }
            return a;
        });
    });

    function handleMouseDown(ev: React.MouseEvent, compDef: ICompDef<any>) {

        let newComp = editorState.compLibrary.create(compDef.defId)!;

        setEditorState(a => assignImm(a, {
            d18_ragCreateComp: { compOrig: newComp },
        }));

        ev.preventDefault();
        ev.stopPropagation();

        setD18_ragStart(ev, 0);
    }

    return <div className={s.libraryView}>
        <div className={s.header}>Components</div>
        <div className={s.body}>
            {compDefs.map((comp, idx) => {

                return <div
                    className={s.entry}
                    key={idx}
                    onMouseDown={ev => handleMouseDown(ev, comp.compDef!)}
                >{comp.name}</div>;
            })}
        </div>
    </div>;
};
