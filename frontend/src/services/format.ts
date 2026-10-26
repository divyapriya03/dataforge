// Display helpers. A missing value (null/undefined/NaN) is never shown as 0.

export const UNAVAILABLE = 'not available';

export const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

export const fmtNum = (v: number | null | undefined, digits = 4): string => (isNum(v) ? v.toFixed(digits) : UNAVAILABLE);

export const fmtPct = (v: number | null | undefined, digits = 1): string => (isNum(v) ? `${v.toFixed(digits)}%` : UNAVAILABLE);

export const fmtText = (v: string | number | null | undefined): string => (v === null || v === undefined || v === '' ? UNAVAILABLE : String(v));

export const STATUS_LABELS: Record<string, string> = {
  completed: 'Completed',
  output_missing: 'Output file missing',
};

export const statusLabel = (status: string): string => STATUS_LABELS[status] ?? status;
