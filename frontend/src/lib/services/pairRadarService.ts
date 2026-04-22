import api from '../api';

export interface PairRequest {
  symbol_a: string;
  symbol_b: string;
  interval: string;
  mode: string;
  rolling_window: number;
  pair_window: number;
  copula_mode?: string;
  copula_type?: string;
  copula_param?: number;
  r_window?: number;
  copula_stationarize?: boolean;
  copula_ema_window?: number;
}

export interface PairResponse {
  metrics: {
    Coefficient: number | null;
    VolRatio: number | null;
    ADF_P: number | null;
    R2: number | null;
  };
  chart_data: any[];
  copula?: {
    u: number[]; v: number[];
    u_curr: number; v_curr: number;
    x: number[]; y: number[];
    p_uv: number; p_vu: number;
    dens_a: number[]; dens_b: number[];
    title_suffix: string;
  };
  comp?: {
    cum_ret_a: number[];
    cum_ret_b: number[];
    ts: string[];
  };
}

export const pairRadarService = {
  generatePairRadar: async (req: PairRequest): Promise<PairResponse> => {
    const response = await api.post('/pair-radar/generate', req);
    return response.data;
  }
};
