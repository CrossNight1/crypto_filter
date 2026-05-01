import api from '../api';

export interface MatrixRequest {
  symbols: string[];
  interval?: string;
  structure?: string;
  data_source?: string;
  data_structure?: string;
  method?: string;
  window?: number;
  mean_reversion?: boolean;
}

export interface DecompRequest {
  symbols: string[];
  interval?: string;
  method?: string;
  data_source?: string;
  data_structure?: string;
  window?: number;
  n_components?: number;
  linkage_method?: string;
  matrix_structure?: string;
  matrix_method?: string;
  mean_reversion?: boolean;
}

export const multivariateService = {
  generateMatrix: async (req: MatrixRequest): Promise<{ columns: string[], index: string[], data: number[][] }> => {
    const response = await api.post('/multivariate/matrix', req);
    return response.data;
  },

  runDecomposition: async (req: DecompRequest): Promise<any> => {
    const response = await api.post('/multivariate/decomposition', req);
    return response.data;
  }
};
