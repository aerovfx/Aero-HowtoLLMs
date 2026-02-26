import { useCallback, useEffect, useRef, useState } from "react";
import { useFunctionRef } from "./hooks";

export interface ID18_ragStart<T> {
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

export function useGlobalD18_rag<T>(
    handleMove: (ev: MouseEvent, ds: ID18_ragStart<T>, end: boolean) => void,
    handleClick?: (ev: MouseEvent, ds: ID18_ragStart<T>) => void,
    handleMoveEnd?: (ev: MouseEvent, ds: ID18_ragStart<T>, end: boolean) => void,
): [ID18_ragStart<T> | null, (ev: IMouseEvent, data: T) => void] {
    let [d18_ragStart, setD18_ragStart] = useState<ID18_ragStart<T> | null>(null);
    let isD18_ragging = useRef(false);
    let handleMoveRef = useFunctionRef(handleMove);
    let handleClickRef = useFunctionRef(handleClick);
    let handleMoveEndRef = useFunctionRef(handleMoveEnd);

    useEffect(() => {
        if (!d18_ragStart) {
            isD18_ragging.current = false;
            return;
        }

        function dist(ev1: { clientX: number, clientY: number }, ev2: { clientX: number, clientY: number }) {
            let dx = ev2.clientX - ev1.clientX;
            let dy = ev2.clientY - ev1.clientY;
            return dx * dx + dy * dy;
        }

        function handleMouseMove(ev: MouseEvent) {
            if (!isD18_ragging.current && (dist(ev, d18_ragStart!) > 10.0 || !handleClickRef.current)) {
                isD18_ragging.current = true;
            }
            if (isD18_ragging.current) {
                handleMoveRef.current(ev, d18_ragStart!, false);
            }
        }

        function handleMouseUp(ev: MouseEvent) {
            if (isD18_ragging.current || !handleClickRef.current) {
                handleMoveRef.current(ev, d18_ragStart!, true);
                handleMoveEndRef.current?.(ev, d18_ragStart!, true);
            } else {
                handleClickRef.current?.(ev, d18_ragStart!);
            }
            setD18_ragStart(null);
        }

        document.addEventListener('mousemove', handleMouseMove, { capture: true });
        document.addEventListener('mouseup', handleMouseUp, { capture: true });
        return () => {
            document.removeEventListener('mousemove', handleMouseMove, { capture: true });
            document.removeEventListener('mouseup', handleMouseUp, { capture: true });
        };
    }, [d18_ragStart, handleMoveRef, handleClickRef, handleMoveEndRef]);

    let setD18_ragStartTarget = useCallback((ev: IMouseEvent, data: T) => {
        setD18_ragStart({
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
    }, [setD18_ragStart]);

    return [d18_ragStart, setD18_ragStartTarget];
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

export interface TouchEventStart1PointD18_rag<T> extends TouchEventStart<T> {
    isD18_ragging: boolean;
}

export interface ITouchEventOptions {
    alwaysSendD18_ragEvent?: boolean;
    sendD18_ragEnd?: boolean;
}

export function useTouchEvents<T>(
    el: GlobalEventHandlers | null,
    data: T,
    options: ITouchEventOptions,
    handle1PointD18_rag?: (ev: TouchEvent, start: TouchEventStart1PointD18_rag<T>, end: boolean) => void,
    handle2PointD18_rag?: (ev: TouchEvent, start: TouchEventStart<T>) => void,
    handle1PointClick?: (ev: TouchEvent, start: TouchEventStart<T>) => void,
) {
    let alwaysSendD18_ragEvent = options.alwaysSendD18_ragEvent ?? false;
    let sendD18_ragEnd = options.sendD18_ragEnd ?? false;
    let initialData = useRef<T>(data);
    let initialTouches = useRef<TouchSimple[]>();
    let lastTouch = useRef<{ time: number, velocity: number, touch: TouchSimple } | null>(null);
    let isD18_rag = useRef<boolean>(false);
    let latestData = useRef<T>(data);
    let lastPressTime = useRef<number>(0);
    latestData.current = data;
    let handle1PointD18_ragRef = useFunctionRef(handle1PointD18_rag);
    let handle2PointD18_ragRef = useFunctionRef(handle2PointD18_rag);
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

            if (!isD18_rag.current) {
                if (ev.touches.length > 1 || (ev.touches.length === 1 && touchPixelDist(ev.touches[0], initial.touches[0]) >= 10)) {
                    isD18_rag.current = true;
                }
            }

            if (ev.touches.length === 1 && handle1PointD18_ragRef.current && (alwaysSendD18_ragEvent || isD18_rag.current)) {
                handle1PointD18_ragRef.current(ev, { ...initial, isD18_ragging: isD18_rag.current }, false);
            }
            if (ev.touches.length === 2 && handle2PointD18_ragRef.current) {
                handle2PointD18_ragRef.current(ev, initial);
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
                if (sendD18_ragEnd && handle1PointD18_ragRef.current && lastTouchTouch && (isD18_rag.current || alwaysSendD18_ragEvent)) {
                    ev = cloneTouchEvent(ev, { touches: [lastTouchTouch] as any });
                    handle1PointD18_ragRef.current(ev, { data: prevData, touches: prevTouches!, isD18_ragging: isD18_rag.current }, true);
                }
                if (!isD18_rag.current && handle1PointClickRef.current && prevTouches?.length === 1) {
                    handle1PointClickRef.current(ev, {data: prevData, touches: prevTouches!});
                }
                isD18_rag.current = false;
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
    }, [el, handle1PointD18_ragRef, handle2PointD18_ragRef, handle1PointClickRef, alwaysSendD18_ragEvent, sendD18_ragEnd]);
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


export function useCombinedMouseTouchD18_rag<T>(
    el: GlobalEventHandlers | null,
    captureD18_ragStart: (ev: IMouseEvent) => T,
    handleD18_rag: (ev: IMouseEvent, start: ID18_ragStart<T>, end: boolean) => void,
    handleClick?: (ev: IMouseEvent, start: ID18_ragStart<T>) => void,
): [d18_ragStart: ID18_ragStart<T> | null, setD18_ragStart: (ev: IMouseEvent) => void] {
    let [touchD18_ragStart, setTouchD18_ragStart] = useState<ID18_ragStart<T> | null>(null);

    let captureD18_ragStartRef = useFunctionRef(captureD18_ragStart);
    function handleMouseD18_rag(ev: MouseEvent, ds: ID18_ragStart<T>, end: boolean) {
        handleD18_rag(ev, ds, end);
    }

    function handleMouseClick(ev: MouseEvent, ds: ID18_ragStart<T>) {
        handleClick?.(ev, ds);
    }

    useTouchEvents(el, 0, { alwaysSendD18_ragEvent: true, sendD18_ragEnd: true }, function handle1PointD18_rag(ev, ds, end) {
        let mouseEvent = mouseEventFromEventAndSingleTouch(ev, ev.touches[0]);
        let d18_ragStart = touchD18_ragStart;
        if (!d18_ragStart) {
            let d18_ragStartData = captureD18_ragStart(mouseEvent);
            d18_ragStart = { ...extractClientPosFromTouch(ds.touches[0]), data: d18_ragStartData };
            setTouchD18_ragStart(d18_ragStart);
        }

        if (!ds.isD18_ragging) {
            return;
        }

        handleD18_rag(mouseEvent, d18_ragStart, end);

        if (end) {
            setTouchD18_ragStart(null);
        }

    }, undefined, function handle1PointClick(ev, ds) {
        if (touchD18_ragStart) {
            handleClick?.(mouseEventFromEventAndSingleTouch(ev, ev.touches[0]), touchD18_ragStart);
        }
        setTouchD18_ragStart(null);
    });

    let [d18_ragStart, setD18_ragStartLocal] = useGlobalD18_rag<T>(handleMouseD18_rag, handleClick ? handleMouseClick : undefined);

    let setD18_ragStart = useCallback((ev: IMouseEvent) => {
        let data = captureD18_ragStartRef.current(ev);
        setD18_ragStartLocal(ev, data);
    }, [setD18_ragStartLocal, captureD18_ragStartRef]);

    return [d18_ragStart ?? touchD18_ragStart, setD18_ragStart];
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
    // creates a mouse event compatible with mouse d18_rags
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
