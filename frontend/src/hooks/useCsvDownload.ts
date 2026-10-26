import { useRef, useState } from 'react';
import { downloadExperimentCsv } from '../services/api';
import { saveBlob } from '../services/download';

export interface DownloadNotice {
  kind: 'success' | 'error';
  text: string;
}

/**
 * One CSV download per experiment at a time (repeated clicks are ignored). The bytes come from
 * GET /api/export/{experiment_id}; nothing is generated or cached in the browser.
 */
export function useCsvDownload(onNotFound?: () => void) {
  const inFlight = useRef<Set<string>>(new Set());
  const [downloading, setDownloading] = useState<string[]>([]);
  const [notice, setNotice] = useState<DownloadNotice | null>(null);

  const download = async (experimentId: string) => {
    if (inFlight.current.has(experimentId)) return;
    inFlight.current.add(experimentId);
    setDownloading(Array.from(inFlight.current));
    setNotice(null);
    try {
      const res = await downloadExperimentCsv(experimentId);
      if (res.success && res.data) {
        saveBlob(res.data.blob, res.data.filename);
        setNotice({ kind: 'success', text: `Downloaded ${res.data.filename}.` });
      } else {
        setNotice({ kind: 'error', text: `Download of ${experimentId} failed: ${res.error ?? 'unknown error'}` });
        if (res.status === 404) onNotFound?.();
      }
    } finally {
      inFlight.current.delete(experimentId);
      setDownloading(Array.from(inFlight.current));
    }
  };

  return { download, downloading, notice, clearNotice: () => setNotice(null) };
}
