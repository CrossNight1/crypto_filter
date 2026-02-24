
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.sparse.csgraph import minimum_spanning_tree as scipy_mst
from sklearn.decomposition import FastICA

class CorrelationEngine:
    """
    Modular engine for calculating correlation matrices between assets.
    """

    @staticmethod
    def calculate_coint_matrix(data_map, window_size=50):
        def _ols_residual(y, x):
            x = np.vstack([x, np.ones(len(x))]).T
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            y_hat = x @ beta
            return y - y_hat

        def _adf_tstat(residual):

            r = np.asarray(residual, dtype=float)

            if len(r) < 5 or np.any(~np.isfinite(r)):
                print("Invalid residual")
                return np.nan

            dr = np.diff(r)
            r_lag = r[:-1]

            X = np.column_stack([r_lag, np.ones(len(r_lag))])

            beta = np.linalg.lstsq(X, dr, rcond=None)[0]

            pred = X @ beta
            err = dr - pred

            dof = len(dr) - 2
            if dof <= 0:
                print("Invalid dof")
                return np.nan

            s2 = np.sum(err**2) / dof
            var_beta = s2 * np.linalg.pinv(X.T @ X)

            if var_beta[0,0] <= 0:
                return np.nan

            se = np.sqrt(var_beta[0,0])
            if se == 0:
                return np.nan

            return beta[0] / se

        wide_long = data_map[window_size:]
        wide_short = data_map[window_size // 10:]

        wide_long = wide_long.dropna(axis=0, how="any").dropna(axis=1, how="any")
        wide_short = wide_short.dropna(axis=0, how="any").dropna(axis=1, how="any")

        common_cols = wide_long.columns.intersection(wide_short.columns)
        cols = common_cols

        coint_long = pd.DataFrame(np.nan, index=cols, columns=cols)
        coint_short = pd.DataFrame(np.nan, index=cols, columns=cols)

        for i in cols:
            for j in cols:
                if i == j:
                    coint_long.loc[i, j] = 0
                    coint_short.loc[i, j] = 0
                    continue

                # ---------- LONG ----------
                df_pair_long = pd.concat([wide_long[i], wide_long[j]], axis=1).dropna()

                y = df_pair_long.iloc[:, 0].values
                x = df_pair_long.iloc[:, 1].values
                try:
                    r = _ols_residual(y, x)
                    coint_long.loc[i, j] = _adf_tstat(r)
                except:
                    pass

                # ---------- SHORT ----------
                df_pair_short = pd.concat([wide_short[i], wide_short[j]], axis=1).dropna()

                y = df_pair_short.iloc[:, 0].values
                x = df_pair_short.iloc[:, 1].values
                try:
                    r = _ols_residual(y, x)
                    coint_short.loc[i, j] = _adf_tstat(r)
                except:
                    pass

        long_weight = 0.9
        short_weight = 0.1

        coint_long = coint_long.fillna(0)
        coint_short = coint_short.fillna(0)

        return (long_weight * coint_long) + (short_weight * coint_short)

    @staticmethod
    def calculate_matrix(data_map, method="pearson", window_size=50):

        wide_long = pd.DataFrame(data_map)[window_size:]
        wide_short = pd.DataFrame(data_map)[window_size // 10:]

        corr_long = wide_long.corr(method=method, min_periods=window_size)
        corr_short = wide_short.corr(method=method, min_periods=window_size)

        long_weight = 0.9
        short_weight = 0.1

        corr_shrink = (long_weight * corr_long) + (short_weight * corr_short)

        return corr_shrink

    @staticmethod
    def calculate_covariance(data_map, window_size=50):

        wide_long = pd.DataFrame(data_map)[window_size:]
        wide_short = pd.DataFrame(data_map)[window_size // 10:]

        cov_long = wide_long.cov(min_periods=window_size)
        cov_short = wide_short.cov(min_periods=window_size)

        long_weight = 0.9
        short_weight = 0.1

        cov_shrink = (long_weight * cov_long) + (short_weight * cov_short)
        cov_shrink *= 100

        return cov_shrink

    @staticmethod
    def stabilize_matrix(matrix, eps=1e-6):

        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        matrix = matrix.dropna(axis=0, how="any")
        matrix = matrix.dropna(axis=1, how="any")

        # Force symmetry
        matrix = (matrix + matrix.T) / 2

        # Add diagonal regularization
        matrix = matrix + np.eye(matrix.shape[0]) * eps

        return matrix

    @staticmethod
    def safe_pinv(matrix):
        for eps in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            try:
                m = matrix + np.eye(matrix.shape[0]) * eps if eps > 0 else matrix
                result = np.linalg.pinv(m)
                if np.isfinite(result).all():
                    return result
            except np.linalg.LinAlgError:
                continue
        # Last resort: return zeros
        return np.zeros_like(matrix)

    @staticmethod
    def calculate_partial_correlation(data_map, window_size=50):

        wide_df = pd.DataFrame(data_map)[window_size:]
        cov = wide_df.cov(min_periods=window_size)

        cov = CorrelationEngine.stabilize_matrix(cov)

        precision = CorrelationEngine.safe_pinv(cov.values)

        d = np.sqrt(np.diag(precision))
        partial = -precision / np.outer(d, d)

        np.fill_diagonal(partial, 1)

        return pd.DataFrame(partial, index=cov.index, columns=cov.columns)

    @staticmethod
    def calculate_precision_matrix(data_map, window_size=50):

        wide_df = pd.DataFrame(data_map)[window_size:]
        cov = wide_df.cov(min_periods=window_size)

        cov = CorrelationEngine.stabilize_matrix(cov)

        precision = CorrelationEngine.safe_pinv(cov.values)

        return pd.DataFrame(precision, index=cov.index, columns=cov.columns)

    @staticmethod
    def filter_blanks(corr_matrix):
        """
        Iteratively drops symbols with the most NaNs in the correlation matrix.
        
        Returns:
            (filtered_matrix, dropped_symbols)
        """
        dropped_symbols = []
        matrix = corr_matrix.copy()
        
        while matrix.isna().any().any():
            nan_counts = matrix.isna().sum()
            if nan_counts.max() == 0:
                break
            
            worst_symbol = nan_counts.idxmax()
            matrix = matrix.drop(index=worst_symbol, columns=worst_symbol)
            dropped_symbols.append(worst_symbol)
            
            if len(matrix) < 2:
                break
                
        return matrix, dropped_symbols


class DecompositionEngine:
    """Spectral, statistical, and structural decomposition of matrices."""

    @staticmethod
    def eigen_decompose(matrix):
        """Eigenvalue decomposition Σ = QΛQ^T."""
        vals = np.linalg.eigvalsh(matrix.values)
        eigenvalues = np.sort(vals)[::-1]
        total = eigenvalues.sum()
        explained = eigenvalues / total if total > 0 else eigenvalues * 0
        cumulative = np.cumsum(explained)
        return {
            'eigenvalues': eigenvalues,
            'explained_ratio': explained,
            'cumulative': cumulative,
            'labels': matrix.columns.tolist()
        }

    @staticmethod
    def spectral_filter_rmt(corr_matrix, T, N):
        """Random Matrix Theory denoising.
        Removes eigenvalues within the Marchenko-Pastur bulk.
        λ± = (1 ± √(N/T))²
        """
        q = N / T
        lambda_plus = (1 + np.sqrt(q)) ** 2
        lambda_minus = (1 - np.sqrt(q)) ** 2

        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix.values)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Replace bulk eigenvalues with average of bulk
        bulk_mask = (eigenvalues <= lambda_plus) & (eigenvalues >= lambda_minus)
        if bulk_mask.any():
            bulk_avg = eigenvalues[bulk_mask].mean()
            eigenvalues[bulk_mask] = bulk_avg

        # Reconstruct
        denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Re-normalize diagonal to 1
        d = np.sqrt(np.diag(denoised))
        d[d == 0] = 1
        denoised = denoised / np.outer(d, d)

        return {
            'denoised_matrix': pd.DataFrame(denoised, index=corr_matrix.index, columns=corr_matrix.columns),
            'eigenvalues_raw': np.sort(np.linalg.eigvalsh(corr_matrix.values))[::-1],
            'eigenvalues_filtered': eigenvalues,
            'lambda_plus': lambda_plus,
            'lambda_minus': lambda_minus,
            'n_signal': int((~bulk_mask).sum()),
            'n_noise': int(bulk_mask.sum())
        }

    @staticmethod
    def k_factor_decompose(returns_df, k, mode='top'):
        
        cov_matrix = returns_df.cov()
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix.values)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        N = len(eigenvalues)
        k = min(k, N)

        if mode == 'top':
            selected_evals = eigenvalues[:k]
            V_k = eigenvectors[:, :k]
            col_names = [f"F{i+1}" for i in range(k)]
        else:
            selected_evals = eigenvalues[-k:]
            V_k = eigenvectors[:, -k:]
            col_names = [f"F{N-k+i+1}" for i in range(k)]

        factor_returns = returns_df.values @ V_k
        reconstructed_returns = factor_returns @ V_k.T

        factor_returns_df = pd.DataFrame(
            factor_returns,
            index=returns_df.index,
            columns=col_names
        )

        reconstructed_df = pd.DataFrame(
            reconstructed_returns,
            index=returns_df.index,
            columns=returns_df.columns
        )

        loadings = V_k * np.sqrt(np.maximum(selected_evals, 0))

        total = eigenvalues.sum()
        variance_captured = selected_evals.sum() / total if total > 0 else 0

        return {
            'factor_returns': factor_returns_df,
            'reconstructed_returns': reconstructed_df,
            'reconstructed': reconstructed_df.corr(),
            'loadings': pd.DataFrame(loadings, index=returns_df.columns, columns=col_names),
            'eigenvalues': eigenvalues,
            'variance_captured': variance_captured,
            'k': k,
            'mode': mode
        }

    @staticmethod
    def distance_matrix(corr_matrix):
        """D_ij = √(2(1 - ρ_ij))"""
        dist = np.sqrt(2 * (1 - corr_matrix.values))
        np.fill_diagonal(dist, 0)
        return pd.DataFrame(dist, index=corr_matrix.index, columns=corr_matrix.columns)

    @staticmethod
    def hierarchical_cluster(dist_matrix, method='ward'):
        """Hierarchical clustering on distance matrix."""
        from scipy.spatial.distance import squareform
        condensed = squareform(dist_matrix.values, checks=False)
        Z = linkage(condensed, method=method)
        labels = dist_matrix.index.tolist()

        # Determine optimal clusters via max gap in merge distances
        merge_dists = Z[:, 2]
        gaps = np.diff(merge_dists)
        n_clusters = len(labels) - np.argmax(gaps) - 1 if len(gaps) > 0 else 2
        n_clusters = max(2, min(n_clusters, len(labels) // 2))

        cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

        return {
            'linkage_matrix': Z,
            'labels': labels,
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters
        }

    @staticmethod
    def mst_spillover(dist_matrix):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree as scipy_mst
        import networkx as nx

        sparse = csr_matrix(dist_matrix.values)
        mst = scipy_mst(sparse).toarray()
        labels = dist_matrix.index.tolist()

        G = nx.Graph()
        for i, row in enumerate(mst):
            for j, w in enumerate(row):
                if w > 0:
                    G.add_edge(labels[i], labels[j], weight=w)

        # compute spillover: sum of inverse path lengths from node to all others
        spillover = {}
        for node in G.nodes:
            lengths = nx.single_source_dijkstra_path_length(G, node, weight='weight')
            # inverse distances as spillover measure
            spillover[node] = sum(1 / l if l > 0 else 0 for target, l in lengths.items())

        edges = [{'source': u, 'target': v, 'weight': d['weight']} for u, v, d in G.edges(data=True)]

        return {
            'edges': edges,
            'labels': labels,
            'n_edges': len(edges),
            'spillover': spillover
        }


    @staticmethod
    def ica_decompose(data_wide, n_components=5):
        """FastICA decomposition on return matrix."""
        data_clean = data_wide.dropna(axis=1, how='any').dropna(axis=0, how='any')
        n_components = min(n_components, data_clean.shape[1], data_clean.shape[0])

        ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
        sources = ica.fit_transform(data_clean.values)
        mixing = ica.mixing_

        return {
            'sources': pd.DataFrame(sources, index=data_clean.index, columns=[f"IC{i+1}" for i in range(n_components)]),
            'mixing': pd.DataFrame(mixing, index=data_clean.columns, columns=[f"IC{i+1}" for i in range(n_components)]),
            'n_components': n_components
        }
