import api from '../api';

export interface DiagnosticsRequest {
  symbol: string;
  interval?: string;
  metric_window?: number;
  diag_window?: number;
}

export interface DiagnosticsResponse {
  performance: {
    sharpe: number;
    sortino: number;
    maxdd: number;
    avgdd: number;
    cvar: number;
    volatility: number;
    omega: number;
    beta: number;
    alpha: number;
    impact_spread: number;
    imbalance: number;
  };
  charts: {
    metrics: { Metric: string, Value: number }[];
    mn_cum_ret: number[];
    prices: {
      hist: number[];
      forecast: number[];
      ci_lower: number[];
      ci_upper: number[];
    };
    volatility: {
      hist: number[];
      forecast: number[];
    };
    ohlcv?: {
      time: string;
      open: number;
      high: number;
      low: number;
      close: number;
      volume: number;
      bb_up: number;
      bb_dn: number;
      bb_mid: number;
    }[];
  };
}

export const diagnosticsService = {
  runDiagnostics: async (req: DiagnosticsRequest): Promise<DiagnosticsResponse> => {
    const response = await api.post('/diagnostics/run', req);
    return response.data;
  }
};
