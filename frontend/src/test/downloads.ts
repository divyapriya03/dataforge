// Test-only: captures what the page hands to the browser as a download.
import { vi } from 'vitest';

export const readBlob = (b: Blob) =>
  new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = reject;
    r.readAsText(b);
  });

export function captureDownloads() {
  const saved: { filename: string; blob: Blob }[] = [];
  const blobs = new Map<string, Blob>();
  let n = 0;
  (URL as unknown as { createObjectURL: (b: Blob) => string }).createObjectURL = (b: Blob) => {
    const url = `blob:test/${n++}`;
    blobs.set(url, b);
    return url;
  };
  (URL as unknown as { revokeObjectURL: (u: string) => void }).revokeObjectURL = () => {};
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    saved.push({ filename: this.download, blob: blobs.get(this.href)! });
  });
  return saved;
}
