import React, { memo } from "react";
import { Portal } from "./Portal";
import clsx from "clsx";

export const CursorD18_ragOverlay: React.FC<{
    className?: string;
}> = memo(function CursorD18_ragOverlay({ className }) {

    return <Portal>
        <div className={clsx("fixed inset-0 z-50 pointer-events-auto", className)} />
    </Portal>;
});
