// Registry of long-running requests that must not be started twice. It lives outside React components so the
// guard survives navigating away from a page while its request is still running on the server.
import { useSyncExternalStore } from 'react';

const active = new Set<string>();
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export const inflight = {
  /** Marks `key` as running. Returns false (and changes nothing) if it already is. */
  begin(key: string): boolean {
    if (active.has(key)) return false;
    active.add(key);
    emit();
    return true;
  },
  end(key: string) {
    if (active.delete(key)) emit();
  },
  has: (key: string) => active.has(key),
  /** Tests only. */
  reset() {
    active.clear();
    emit();
  },
};

export function useInflight(key: string): boolean {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => active.has(key),
  );
}
