import { useCallback, useEffect, useRef, useState } from "react";
import { useFunctionRef } from "./hooks";

export interface ID18-RAGStart<T> {
    clientX: number;
    clientY: number;
    data: T;
    button: number;
    buttons: number;
    shiftKey: boolean;
    altKey: boolean;
    metaKey: boolean;
    ctrlKey: boolean;
}

export function useGlobalD18-RAG<T>(
    handleMove: (ev: MouseEvent, ds: ID18-RAGStart<T>, end: boolean) => void,
    handleClick?: (ev: MouseEvent, ds: ID18-RAGStart<T>) => void,
    handleMoveEnd?: (ev: MouseEvent, ds: ID18-RAGStart<T>, end: boolean) => void,
): [ID18-RAGStart<T> | null, (ev: IMouseEvent, data: T) => void] {
    let [d18-RAGStart, setD18-RAGStart] = useState<ID18-RAGStart<T> | null>(null);
    let isD18-RAGging = useRef(false);
    let handleMoveRef = useFunctionRef(handleMove);
    let handleClickRef = useFunctionRef(handleClick);
    let handleMoveEndRef = useFunctionRef(handleMoveEnd);

    useEffect(() => {
        if (!d18-RAGStart) {
            isD18-RAGging.current = false;
            return;
        }

        function dist(ev1: { clientX: number, clientY: number }, ev2: { clientX: number, clientY: number }) {
            let dx = ev2.clientX - ev1.clientX;
            let dy = ev2.clientY - ev1.clientY;
            return dx * dx + dy * dy;
        }

        function handleMouseMove(ev: MouseEvent) {
            if (!isD18-RAGging.current && (dist(ev, d18-RAGStart!) > 10.0 || !handleClickRef.current)) {
                isD18-RAGging.current = true;
            }
            if (isD18-RAGging.current) {
                handleMoveRef.current(ev, d18-RAGStart!, false);
            }
        }

        function handleMouseUp(ev: MouseEvent) {
            if (isD18-RAGging.current || !handleClickRef.current) {
                handleMoveRef.current(ev, d18-RAGStart!, true);
                handleMoveEndRef.current?.(ev, d18-RAGStart!, true);
            } else {
                handleClickRef.current?.(ev, d18-RAGStart!);
            }
            setD18-RAGStart(null);
        }

        document.addEventListener('mousemove', handleMouseMove, { capture: true });
        document.addEventListener('mouseup', handleMouseUp, { capture: true });
        return () => {
            document.removeEventListener('mousemove', handleMouseMove, { capture: true });
            document.removeEventListener('mouseup', handleMouseUp, { capture: true });
        };
    }, [d18-RAGStart, handleMoveRef, handleClickRef, handleMoveEndRef]);

    let setD18-RAGStartTarget = useCallback((ev: IMouseEvent, data: T) => {
        setD18-RAGStart({
            clientX: ev.clientX,
            clientY: ev.clientY,
            data,
            button: ev.button,
            buttons: ev.buttons,
            shiftKey: ev.shiftKey,
            altKey: ev.altKey,
            ctrlKey: ev.ctrlKey,
            metaKey: ev.metaKey,
        });
    }, [setD18-RAGStart]);

    return [d18-RAGStart, setD18-RAGStartTarget];
}

export interface IMouseLocation {
    clientX: number;
    clientY: number;
}

export interface IPointerEvent {
    clientX: number;
    clientY: number;
}

export interface IMouseEvent extends IPointerEvent {
    type: string;
    readonly button: number;
    readonly buttons: number;
    readonly shiftKey: boolean;
    readonly altKey: boolean;
    readonly ctrlKey: boolean;
    readonly metaKey: boolean;
    stopPropagation(): void;
    preventDefault(): void;
}

export interface IWheelEvent extends IMouseEvent {
    deltaMode: number;
    deltaY: number;
}

export interface IBaseEvent {
    type: string;
    stopPropagation(): void;
    preventDefault(): void;
}

export function getWheelDelta(ev: IWheelEvent): number {
    let mode = ev.deltaMode;
    let scale = 1.0;
    if (mode === 0) { // pixel
        scale = 125;
    } else if (mode === 1) { // line
        scale = 3;
    } else if (mode === 2) { // page
        scale = 0.1;
    }

    return ev.deltaY / scale;
}

export interface TouchSimple {
    clientX: number;
    clientY: number;
}

export interface TouchEventStart<T> {
    data: T;
    touches: TouchSimple[];
}

export interface TouchEventStart1PointD18-RAG<T> extends TouchEventStart<T> {
    isD18-RAGging: boolean;
}

export interface ITouchEventOptions {
    alwaysSendD18-RAGEvent?: boolean;
    sendD18-RAGEnd?: boolean;
}

export function useTouchEvents<T>(
    el: GlobalEventHandlers | null,
    data: T,
    options: ITouchEventOptions,
    handle1PointD18-RAG?: (ev: TouchEvent, start: TouchEventStart1PointD18-RAG<T>, end: boolean) => void,
    handle2PointD18-RAG?: (ev: TouchEvent, start: TouchEventStart<T>) => void,
    handle1PointClick?: (ev: TouchEvent, start: TouchEventStart<T>) => void,
) {
    let alwaysSendD18-RAGEvent = options.alwaysSendD18-RAGEvent ?? false;
    let sendD18-RAGEnd = options.sendD18-RAGEnd ?? false;
    let initialData = useRef<T>(data);
    let initialTouches = useRef<TouchSimple[]>();
    let lastTouch = useRef<{ time: number, velocity: number, touch: TouchSimple } | null>(null);
    let isD18-RAG = useRef<boolean>(false);
    let latestData = useRef<T>(data);
    let lastPressTime = useRef<number>(0);
    latestData.current = data;
    let handle1PointD18-RAGRef = useFunctionRef(handle1PointD18-RAG);
    let handle2PointD18-RAGRef = useFunctionRef(handle2PointD18-RAG);
    let handle1PointClickRef = useFunctionRef(handle1PointClick);

    useEffect(() => {
        if (!el) {
            return;
        }

        function sendEvent(ev: TouchEvent) {
            let initial = {data: initialData.current, touches: initialTouches.current!};
            if (!ev.touches || !initial.touches || ev.touches.length !== initial.touches.length) {
                return;
            }

            if (!isD18-RAG.current) {
                if (ev.touches.length > 1 || (ev.touches.length === 1 && touchPixelDist(ev.touches[0], initial.touches[0]) >= 10)) {
                    isD18-RAG.current = true;
                }
            }

            if (ev.touches.length === 1 && handle1PointD18-RAGRef.current && (alwaysSendD18-RAGEvent || isD18-RAG.current)) {
                handle1PointD18-RAGRef.current(ev, { ...initial, isD18-RAGging: isD18-RAG.current }, false);
            }
            if (ev.touches.length === 2 && handle2PointD18-RAGRef.current) {
                handle2PointD18-RAGRef.current(ev, initial);
            }

            if (ev.touches.length === 1) {
                lastTouch.current = {
                    time: 0,
                    velocity: 0,
                    touch: copyTouchList(ev.touches)[0],
                };

            } else {
                lastTouch.current = null;
            }
        }

        function captureInitialAndSend(ev: TouchEvent) {
            let prevTouches = initialTouches.current;
            let prevData = initialData.current;
            initialData.current = latestData.current;
            initialTouches.current = copyTouchList(ev.touches as any);

            if (!prevTouches || !prevTouches.length) {
                lastPressTime.current = performance.now();
            }

            let lastTouchTouch = lastTouch.current?.touch;

            sendEvent(ev);

            if (ev.touches.length === 0) {
                if (sendD18-RAGEnd && handle1PointD18-RAGRef.current && lastTouchTouch && (isD18-RAG.current || alwaysSendD18-RAGEvent)) {
                    ev = cloneTouchEvent(ev, { touches: [lastTouchTouch] as any });
                    handle1PointD18-RAGRef.current(ev, { data: prevData, touches: prevTouches!, isD18-RAGging: isD18-RAG.current }, true);
                }
                if (!isD18-RAG.current && handle1PointClickRef.current && prevTouches?.length === 1) {
                    handle1PointClickRef.current(ev, {data: prevData, touches: prevTouches!});
                }
                isD18-RAG.current = false;
                lastTouch.current = null;
            }
        }

        el.addEventListener('touchstart', captureInitialAndSend, { passive: false });
        el.addEventListener('touchend', captureInitialAndSend, { passive: false });
        el.addEventListener('touchcancel', captureInitialAndSend, { passive: false });
        el.addEventListener('touchmove', sendEvent, { passive: false });
        return () => {
            el.removeEventListener('touchstart', captureInitialAndSend);
            el.removeEventListener('touchend', captureInitialAndSend);
            el.removeEventListener('touchcancel', captureInitialAndSend);
            el.removeEventListener('touchmove', sendEvent);
        };
    }, [el, handle1PointD18-RAGRef, handle2PointD18-RAGRef, handle1PointClickRef, alwaysSendD18-RAGEvent, sendD18-RAGEnd]);
}

export function copyTouchList(tl: TouchList) {
    let res: TouchSimple[] = [];
    for (let i = 0; i < tl.length; i++) {
        let touch = tl[i];
        res.push({ clientX: touch.clientX, clientY: touch.clientY });
    }
    return res;
}

function touchPixelDist(a: TouchSimple, b: TouchSimple) {
    let dx = b.clientX - a.clientX;
    let dy = b.clientY - a.clientY;
    return Math.sqrt(dx * dx + dy * dy);
}


export function useCombinedMouseTouchD18-RAG<T>(
    el: GlobalEventHandlers | null,
    captureD18-RAGStart: (ev: IMouseEvent) => T,
    handleD18-RAG: (ev: IMouseEvent, start: ID18-RAGStart<T>, end: boolean) => void,
    handleClick?: (ev: IMouseEvent, start: ID18-RAGStart<T>) => void,
): [d18-RAGStart: ID18-RAGStart<T> | null, setD18-RAGStart: (ev: IMouseEvent) => void] {
    let [touchD18-RAGStart, setTouchD18-RAGStart] = useState<ID18-RAGStart<T> | null>(null);

    let captureD18-RAGStartRef = useFunctionRef(captureD18-RAGStart);
    function handleMouseD18-RAG(ev: MouseEvent, ds: ID18-RAGStart<T>, end: boolean) {
        handleD18-RAG(ev, ds, end);
    }

    function handleMouseClick(ev: MouseEvent, ds: ID18-RAGStart<T>) {
        handleClick?.(ev, ds);
    }

    useTouchEvents(el, 0, { alwaysSendD18-RAGEvent: true, sendD18-RAGEnd: true }, function handle1PointD18-RAG(ev, ds, end) {
        let mouseEvent = mouseEventFromEventAndSingleTouch(ev, ev.touches[0]);
        let d18-RAGStart = touchD18-RAGStart;
        if (!d18-RAGStart) {
            let d18-RAGStartData = captureD18-RAGStart(mouseEvent);
            d18-RAGStart = { ...extractClientPosFromTouch(ds.touches[0]), data: d18-RAGStartData };
            setTouchD18-RAGStart(d18-RAGStart);
        }

        if (!ds.isD18-RAGging) {
            return;
        }

        handleD18-RAG(mouseEvent, d18-RAGStart, end);

        if (end) {
            setTouchD18-RAGStart(null);
        }

    }, undefined, function handle1PointClick(ev, ds) {
        if (touchD18-RAGStart) {
            handleClick?.(mouseEventFromEventAndSingleTouch(ev, ev.touches[0]), touchD18-RAGStart);
        }
        setTouchD18-RAGStart(null);
    });

    let [d18-RAGStart, setD18-RAGStartLocal] = useGlobalD18-RAG<T>(handleMouseD18-RAG, handleClick ? handleMouseClick : undefined);

    let setD18-RAGStart = useCallback((ev: IMouseEvent) => {
        let data = captureD18-RAGStartRef.current(ev);
        setD18-RAGStartLocal(ev, data);
    }, [setD18-RAGStartLocal, captureD18-RAGStartRef]);

    return [d18-RAGStart ?? touchD18-RAGStart, setD18-RAGStart];
}

function cloneTouchEvent<T extends {}>(ev: TouchEvent, extra: T): TouchEvent & T & { button: -1, buttons: 0 } {
    return {
        ...ev,
        preventDefault: () => ev.preventDefault(),
        stopPropagation: () => ev.stopPropagation(),
        ...extra,
        button: -1,
        buttons: 0,
    };
}

function mouseEventFromEventAndSingleTouch(ev: TouchEvent, touch: TouchSimple): TouchEvent & IMouseEvent {
    return cloneTouchEvent(ev, extractClientPosFromTouch(touch));
}

function extractClientPosFromTouch(touch: TouchSimple) {
    // creates a mouse event compatible with mouse d18-RAGs
    return {
        clientX: touch.clientX,
        clientY: touch.clientY,
        shiftKey: false,
        altKey: false,
        ctrlKey: false,
        metaKey: false,
        button: -1,
        buttons: 0,
     };
}
