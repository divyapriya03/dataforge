/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_APP_TITLE: string;
  /** Milliseconds the UI waits for the synchronous training request (default 600000). */
  readonly VITE_TRAIN_TIMEOUT_MS?: string;
  /** Milliseconds the UI waits for a generation request (default 300000). */
  readonly VITE_GENERATE_TIMEOUT_MS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
