import api from '../api';

export interface PredictiveRequest {
  ticker?: string;
  interval?: string;
  trade_direction?: string;
  features?: string[];
  reg_type?: string;
  test_ratio?: number;
  rf_max_depth?: number;
  eng_lookback?: number;
  pred_lookback?: number;
  bar_type?: string;
  labeler_type?: string;
  labeler_params?: any;
  standardize?: boolean;
  vif_th?: number;
}

export interface PredictiveResponse {
  features_used: string[];
  train_size: number;
  meta_size: number;
  test_size: number;
  accuracy: number;
  classification_report: any;
}

export const predictiveService = {
  runAnalysis: async (req: PredictiveRequest): Promise<PredictiveResponse> => {
    const response = await api.post('/predictive/analyze', req);
    return response.data;
  }
};
