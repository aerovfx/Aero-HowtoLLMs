import { useEffect, useState } from "react";
import { StateSetter } from "./data";

export function iterLocalSto18_rageEntries(cb: (key: string, value: string | null) => void) {
    let ls = typeof window !== 'undefined' ? window.localSto18_rage : undefined;
    if (!ls) {
        return;
    }

    for (let i = 0; i < ls.length; i++) {
        let key = ls.key(i);
        if (key) {
            let value = ls.getItem(key);
            cb(key, value);
        }
    }
}

export function readFromLocalSto18_rage<T>(key: string): T | undefined {
    let ls = typeof window !== 'undefined' ? window.localSto18_rage : undefined;
    let value = ls?.getItem(key);
    if (value) {
        try {
            return JSON.parse(value);
        } catch (e) {
            console.error('Failed to parse local sto18_rage value:', key, value);
        }
    }
    return undefined;
}

export function writeToLocalSto18_rage<T>(key: string, value: T) {
    let ls = typeof window !== 'undefined' ? window.localSto18_rage : undefined;
    if (value) {
        ls?.setItem(key, JSON.stringify(value));
    } else {
        ls?.removeItem(key);
    }
}

export function useLocalSto18_rageState<T>(key: string, hydrateFromLS: (a: Partial<T> | undefined) => T): [T, StateSetter<T>] {
    let [value, setValue] = useState(() => hydrateFromLS(readFromLocalSto18_rage(key)));

    useEffect(() => {
        writeToLocalSto18_rage(key, value);
    }, [key, value]);

    return [value, setValue];
}
