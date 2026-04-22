import api from '../api';

export interface SnapshotRequest {
  symbols?: string[];
  intervals?: string[];
}

export interface PathRequest {
  symbol_a: string;
  symbol_b: string;
  interval: string;
}

export const marketRadarService = {
  getSnapshot: async (req: SnapshotRequest): Promise<{ metrics: any[] }> => {
    const response = await api.post('/market-radar/snapshot', req);
    return response.data;
  },

  getPathAnalysis: async (req: PathRequest): Promise<{ 
    metrics: any, 
    path_data: any[], 
    stats_data: any[],
    volatility: any 
  }> => {
    const response = await api.post('/market-radar/path', req);
    return response.data;
  }
};
