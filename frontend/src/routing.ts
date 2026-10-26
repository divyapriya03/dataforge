// Hash-based routing (the app has no router library). A route is `#/<page>` plus optional query
// parameters that carry identifiers between pages, e.g. `#/quality?experiment=gen_1a2b3c4d`.

export type TabId = 'dashboard' | 'studio' | 'model' | 'generator' | 'quality' | 'history';

export const TAB_PATHS: Record<TabId, string> = {
  dashboard: '/dashboard',
  studio: '/dataset-studio',
  model: '/model-lab',
  generator: '/generator',
  quality: '/quality',
  history: '/history',
};

/** Which query parameter each page understands. */
export const TAB_PARAMS: Record<TabId, string | null> = {
  dashboard: null,
  studio: 'dataset',
  model: 'dataset',
  generator: 'model',
  quality: 'experiment',
  history: null,
};

export interface Route {
  /** 'notfound' for a hash that matches no page. */
  tab: TabId | 'notfound';
  params: Record<string, string>;
}

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#/, '');
  if (raw === '' || raw === '/') return { tab: 'dashboard', params: {} };
  const [path, query = ''] = raw.split('?');
  const tab = (Object.keys(TAB_PATHS) as TabId[]).find((t) => TAB_PATHS[t] === path.replace(/\/+$/, ''));
  if (!tab) return { tab: 'notfound', params: {} };
  const params: Record<string, string> = {};
  new URLSearchParams(query).forEach((v, k) => {
    if (v) params[k] = v;
  });
  return { tab, params };
}

export function buildHash(tab: TabId, params: Record<string, string | undefined> = {}): string {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v) query.set(k, v);
  });
  const q = query.toString();
  return `#${TAB_PATHS[tab]}${q ? `?${q}` : ''}`;
}
