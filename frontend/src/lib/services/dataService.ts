import api from '../api';

export interface FileMetadata {
  file: string;
  size_mb: number;
  last_modified: number;
  ticker: string;
  interval: string;
}

export interface FetchRequest {
  symbols?: string[];
  intervals?: string[];
  mode?: string;
}

export const dataService = {
  getUniverse: async (): Promise<string[]> => {
    const response = await api.get('/data/universe');
    return response.data.symbols;
  },

  getMetadata: async (): Promise<FileMetadata[]> => {
    const response = await api.get('/data/metadata');
    return response.data.metadata;
  },

  fetchData: async (req: FetchRequest): Promise<{ status: string, task_id: string, message: string }> => {
    const response = await api.post('/data/fetch', req);
    return response.data;
  },

  deleteCache: async (interval: string): Promise<{ status: string, message: string }> => {
    const response = await api.delete(`/data/cache/${interval}`);
    return response.data;
  }
};
