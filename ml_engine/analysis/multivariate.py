
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.sparse.csgraph import minimum_spanning_tree as scipy_mst
from sklearn.decomposition import FastICA, PCA

class MatrixEngine:
    """
    Modular engine for calculating correlation matrices between assets.
    """
    _residual_cache = {}

    @staticmethod
    def clear_residual_cache():
        MatrixEngine._residual_cache = {}

    @staticmethod
    def _prepare_data(df, data_source="price", data_structure="raw"):
        """
        Transforms the input DataFrame based on source and structure options.
        """
        # 1. Handle data source
        if data_source == "return":
            # Calculate log returns if not already returns
            df = np.log(df.replace(0, np.nan)).diff().dropna(axis=0, how="all")
        
        # 2. Handle data structure
        if data_structure == "ranking":
            df = df.rank(axis=0, pct=True)
        elif data_structure == "sign":
            if data_source == "price":
                # Sign of the change
                df = np.sign(df.diff().fillna(0))
            else:
                # Sign of the returns
                df = np.sign(df.fillna(0))
        
        return df

    @staticmethod
    def _get_residual(i, j, wide):

        key = (i, j)

        if key in MatrixEngine._residual_cache:
            return MatrixEngine._residual_cache[key]

        df_pair = pd.concat([wide[i], wide[j]], axis=1).dropna()
        if df_pair.shape[0] < 5:
            return None

        y = df_pair.iloc[:, 0].values
        x = df_pair.iloc[:, 1].values

        x_ = np.vstack([x, np.ones(len(x))]).T
        beta = np.linalg.lstsq(x_, y, rcond=None)[0]
        r = y - x_ @ beta

        MatrixEngine._residual_cache[key] = r
        return r

    @staticmethod
    def build_residual_cache(wide):

        cols = wide.columns

        for i in cols:
            for j in cols:

                if i == j:
                    continue

                key = (i, j)
                if key in MatrixEngine._residual_cache:
                    continue

                df = pd.concat([wide[i], wide[j]], axis=1).dropna()
                if len(df) < 5:
                    continue

                y = df.iloc[:,0].values
                x = df.iloc[:,1].values

                X = np.vstack([x, np.ones(len(x))]).T
                beta = np.linalg.lstsq(X, y, rcond=None)[0]

                MatrixEngine._residual_cache[key] = y - X @ beta

    @staticmethod
    def calculate_coint_matrix(data_map, window_size=50):

        def _adf_tstat(r):

            r = np.asarray(r, dtype=float)
            if len(r) < 5 or np.any(~np.isfinite(r)):
                return np.nan

            dr = np.diff(r)
            r_lag = r[:-1]

            X = np.column_stack([r_lag, np.ones(len(r_lag))])
            beta = np.linalg.lstsq(X, dr, rcond=None)[0]

            pred = X @ beta
            err = dr - pred

            dof = len(dr) - 2
            if dof <= 0:
                return np.nan

            s2 = np.sum(err**2) / dof
            var_beta = s2 * np.linalg.pinv(X.T @ X)

            if var_beta[0, 0] <= 0:
                return np.nan

            se = np.sqrt(var_beta[0, 0])
            if se == 0:
                return np.nan

            return beta[0] / se

        MatrixEngine.clear_residual_cache()
        wide = pd.DataFrame(data_map)[-window_size:]
        wide = np.log(wide)

        cols = wide.columns
        coint_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

        for i in cols:
            for j in cols:

                if i == j:
                    coint_matrix.loc[i, j] = 0
                    continue

                r = MatrixEngine._get_residual(i, j, wide)
                if r is None:
                    continue

                coint_matrix.loc[i, j] = _adf_tstat(r)

        return coint_matrix.fillna(0)

    @staticmethod
    def calculate_matrix(data_map, method="pearson", window_size=50, data_source="price", data_structure="raw"):
        # 1. Efficiently merge only the necessary window
        trimmed = {}
        for s, v in data_map.items():
            if len(v) > 0:
                trimmed[s] = v.iloc[-window_size:]
        
        if not trimmed:
            return pd.DataFrame()
            
        df = pd.DataFrame(trimmed)
        
        # 2. Prepare data based on source and structure
        df = MatrixEngine._prepare_data(df, data_source=data_source, data_structure=data_structure)
        
        if df.empty:
            return pd.DataFrame()

        # 3. Slice windows
        wide_long = df
        wide_short = df.iloc[-(window_size // 10):] if len(df) >= (window_size // 10) else df

        # 4. Calculate correlations with relaxed min_periods
        corr_long = wide_long.corr(method=method, min_periods=window_size // 2)
        corr_short = wide_short.corr(method=method, min_periods=max(2, window_size // 20))

        # 5. Robust shrinkage: fallback to long-term for missing short-term pairs
        corr_shrink = (0.9 * corr_long) + (0.1 * corr_short.fillna(corr_long))

        return corr_shrink

    @staticmethod
    def calculate_covariance(data_map, window_size=50, data_source="price", data_structure="raw"):
        # 1. Efficiently merge only the necessary window
        trimmed = {}
        for s, v in data_map.items():
            if len(v) > 0:
                trimmed[s] = v.iloc[-window_size:]
        
        if not trimmed:
            return pd.DataFrame()
            
        df = pd.DataFrame(trimmed)

        # 2. Prepare data
        df = MatrixEngine._prepare_data(df, data_source=data_source, data_structure=data_structure)
        
        if df.empty:
            return pd.DataFrame()

        wide_long = df
        wide_short = df.iloc[-(window_size // 10):] if len(df) >= (window_size // 10) else df

        cov_long = wide_long.cov(min_periods=window_size // 2)
        cov_short = wide_short.cov(min_periods=max(2, window_size // 20))

        # Robust shrinkage
        cov_shrink = (0.9 * cov_long) + (0.1 * cov_short.fillna(cov_long))
        cov_shrink *= 100

        return cov_shrink

    @staticmethod
    def stabilize_matrix(matrix, eps=1e-6):

        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        # Relaxed filtering: only drop if entire row/column is NaN
        matrix = matrix.dropna(axis=1, how="all")
        matrix = matrix.dropna(axis=0, how="all")
        
        # Then use iterative blank filtering for remaining NaNs
        matrix, _ = MatrixEngine.filter_blanks(matrix)

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
    def calculate_partial_correlation(data_map, window_size=50, data_source="price", data_structure="raw"):

        wide_df = pd.DataFrame(data_map)[-window_size:]
        # Prepare data
        wide_df = MatrixEngine._prepare_data(wide_df, data_source=data_source, data_structure=data_structure)
        
        if wide_df.empty:
            return pd.DataFrame()

        cov = wide_df.cov(min_periods=window_size // 2)

        cov = MatrixEngine.stabilize_matrix(cov)

        precision = MatrixEngine.safe_pinv(cov.values)

        diag_vals = np.diag(precision)
        diag_vals = np.clip(diag_vals, a_min=1e-12, a_max=None)
        d = np.sqrt(diag_vals)
        partial = -precision / np.outer(d, d)

        np.fill_diagonal(partial, 1)

        return pd.DataFrame(partial, index=cov.index, columns=cov.columns)

    @staticmethod
    def calculate_precision_matrix(data_map, window_size=50, data_source="price", data_structure="raw"):

        wide_df = pd.DataFrame(data_map)[-window_size:]
        # Prepare data
        wide_df = MatrixEngine._prepare_data(wide_df, data_source=data_source, data_structure=data_structure)
        
        if wide_df.empty:
            return pd.DataFrame()

        cov = wide_df.cov(min_periods=window_size // 2)

        cov = MatrixEngine.stabilize_matrix(cov)

        precision = MatrixEngine.safe_pinv(cov.values)

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

    @staticmethod
    def calculate_zscore_matrix(data_map, window_size=50):

        wide = pd.DataFrame(data_map)[-window_size:]
        wide = wide.dropna(axis=1, how="any").dropna(axis=0, how="any")
        wide = np.log(wide)

        cols = wide.columns
        z_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

        for i in cols:
            for j in cols:

                if i == j:
                    z_matrix.loc[i, j] = 0
                    continue

                r = MatrixEngine._get_residual(i, j, wide)
                if r is None or len(r) < 5:
                    continue

                mu = np.mean(r)
                sigma = np.std(r) + 1e-6

                z_matrix.loc[i, j] = (r[-1] - mu) / sigma

        return z_matrix

    @staticmethod
    def calculate_halflife_matrix(data_map, window_size=50):

        wide = pd.DataFrame(data_map)[-window_size:]
        wide = wide.dropna(axis=1, how="any").dropna(axis=0, how="any")
        wide = np.log(wide)

        cols = wide.columns
        hl_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

        for i in cols:
            for j in cols:

                if i == j:
                    hl_matrix.loc[i, j] = 0
                    continue

                r = MatrixEngine._get_residual(i, j, wide)
                if r is None or len(r) < 10:
                    continue

                dr = np.diff(r)
                r_lag = r[:-1]

                X = np.column_stack([r_lag, np.ones(len(r_lag))])
                beta = np.linalg.lstsq(X, dr, rcond=None)[0][0]

                if beta >= 0:
                    hl_matrix.loc[i, j] = 0
                    continue

                halflife = -np.log(2) / beta
                halflife = min(halflife, 100)

                hl_matrix.loc[i, j] = halflife

        return hl_matrix

    @staticmethod
    def calculate_vol_ratio_matrix(data_map, window_size=50):

        wide = pd.DataFrame(data_map)[-window_size:]

        returns = np.log(wide).diff().dropna(axis=0, how="all")
        vol = returns.std()

        cols = vol.index
        vr_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

        for i in cols:
            for j in cols:

                if i == j:
                    vr_matrix.loc[i, j] = 1
                    continue

                if vol[j] == 0:
                    continue

                ratio = vol[i] / vol[j]
                vr_matrix.loc[i, j] = min(ratio, 1 / ratio)

        return vr_matrix

    @staticmethod
    def calculate_arbitrage_score_matrix(data_map, window_size=50, mean_reversion=True):
        """
        Compute an arbitrage score matrix combining correlation, partial correlation,
        cointegration, z-score, and half-life.

        Returns:
            pd.DataFrame: matrix of arbitrage scores (0–1 normalized)
        """
        if mean_reversion:
            weights = {
                "corr": 0.2,
                "partial": 0.2,
                "halflife": 0.2,
                "coint": 0.2,
                "zscore": 0.2,
            }
        else:
            weights = {
                "corr": 0.4,
                "partial": -0.1,
                "halflife": -0.2,
                "coint": 0.2,
                "zscore": 0.7,
            }
        
        wide = np.log(pd.DataFrame(data_map)[-window_size:])
        MatrixEngine.clear_residual_cache()
        MatrixEngine.build_residual_cache(wide)

        # Compute base matrices
        corr = MatrixEngine.calculate_matrix(data_map, method="pearson", window_size=window_size)
        partial = MatrixEngine.calculate_partial_correlation(data_map, window_size=window_size)
        coint = MatrixEngine.calculate_coint_matrix(data_map, window_size=window_size)
        zscore = MatrixEngine.calculate_zscore_matrix(data_map, window_size=window_size)
        halflife = MatrixEngine.calculate_halflife_matrix(data_map, window_size=window_size)

        # Transform directions: higher = better
        corr_s = corr.abs()
        partial_s = partial.abs()
        coint_s = -coint           # more negative ADF = stronger
        z_s = zscore.abs()
        
        # Safe inverse: avoid division by zero or very small values
        hl_s = pd.DataFrame(0.0, index=halflife.index, columns=halflife.columns)
        mask = (halflife > 0)
        hl_s[mask] = 1.0 / halflife[mask]

        # Normalize helper
        def normalize(mat, v_min=None, v_max=None):
            # Replace inf with nan for min/max calculation
            vals = mat.values
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                return mat * 0
            
            if not v_min:
                v_min = np.min(vals)
            if not v_max:
                v_max = np.max(vals)
            
            if v_max - v_min == 0:
                return mat * 0
            
            # Clip and then normalize to ensure [0, 1] and no infs
            normed = (mat.clip(v_min, v_max) - v_min) / (v_max - v_min)
            return normed.fillna(0)

        corr_s = normalize(corr_s, -1, 1)
        partial_s = normalize(partial_s, -0.5, 0.5)
        coint_s = normalize(coint_s, 0, 5)
        z_s = normalize(z_s, -2, 2)
        hl_s = normalize(hl_s, 5, 100)

        # Weighted aggregation
        score = (
            weights["corr"] * corr_s +
            weights["partial"] * partial_s +
            weights["coint"] * coint_s +
            weights["zscore"] * z_s +
            weights["halflife"] * hl_s
        )

        score = score
        return score        
        
class DecompositionEngine:
    """Spectral, statistical, and structural decomposition of matrices."""

    @staticmethod
    def pca_decompose(returns_df, n_components=5):
        """Principal Component Analysis omitting the 1st component (market factor)."""
        data_clean = returns_df.dropna(axis=1, how='any').dropna(axis=0, how='any')
        n_assets = data_clean.shape[1]
        n_samples = data_clean.shape[0]
        
        # Fit K + 1 components to account for removing the market factor
        # K is the user-requested dimensions AFTER removing market
        k_user = min(n_components, 5, n_assets - 1, n_samples - 1)
        k_user = max(k_user, 1) # Ensure at least one
        k_fit = k_user + 1
        
        pca = PCA(n_components=k_fit, random_state=42)
        pca.fit(data_clean.values)
        
        # explained_ratio: how much variance each component captures
        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        
        # factor_returns: the principal component time-series
        factor_returns = pca.transform(data_clean.values)
        
        # Create a "Market-Neutral" version of factor returns by zeroing out the 1st component
        neutral_factors = factor_returns.copy()
        neutral_factors[:, 0] = 0
        
        # Reconstruction using only components 2..K+1
        reconstructed_returns = pca.inverse_transform(neutral_factors)
        reconstructed_df = pd.DataFrame(
            reconstructed_returns,
            index=data_clean.index,
            columns=data_clean.columns
        )
        
        eigenvalues = pca.explained_variance_
        col_names = [f"PC{i+1}" for i in range(k_fit)]
        loadings = pca.components_.T * np.sqrt(np.maximum(eigenvalues, 0))

        return {
            'eigenvalues': eigenvalues,
            'explained_ratio': explained,
            'cumulative': cumulative,
            'loadings': pd.DataFrame(loadings, index=data_clean.columns, columns=col_names),
            'factor_returns': pd.DataFrame(factor_returns, index=data_clean.index, columns=col_names),
            'reconstructed_returns': reconstructed_df,
            'reconstructed': reconstructed_df.corr(),
            'k': k_user,
            'k_fit': k_fit,
            'is_market_neutral': True,
            'labels': data_clean.columns.tolist()
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
        
        # If Systematic (top), use ONLY PC1
        # If Idiosyncratic (bottom), remove ONLY PC1 (use PC2..PCN)
        if mode == 'top':
            k = 1
            selected_evals = eigenvalues[:k]
            V_k = eigenvectors[:, :k]
            col_names = [f"PC{i+1}" for i in range(k)]
        else:
            # Idiosyncratic: Use all components EXCEPT the first one
            k = max(1, N - 1)
            selected_evals = eigenvalues[1:1+k]
            V_k = eigenvectors[:, 1:1+k]
            col_names = [f"PC{i+2}" for i in range(k)]

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
        explained = eigenvalues / total if total > 0 else eigenvalues * 0
        cumulative = np.cumsum(explained)

        return {
            'factor_returns': factor_returns_df,
            'reconstructed_returns': reconstructed_df,
            'reconstructed': reconstructed_df.corr(),
            'loadings': pd.DataFrame(loadings, index=returns_df.columns, columns=col_names),
            'eigenvalues': eigenvalues,
            'explained_ratio': explained,
            'cumulative': cumulative,
            'variance_captured': variance_captured,
            'k': k,
            'mode': mode,
            'labels': returns_df.columns.tolist()
        }

    @staticmethod
    def distance_matrix(matrix, matrix_type=None):
        """Calculate distance matrix depending on the matrix type."""
        vals = matrix.values.copy()
        vals = np.where(np.isfinite(vals), vals, 0)
        
        if matrix_type == "Correlation" or matrix_type == "Partial Correlation":
            vals = np.clip(vals, -1.0, 1.0)
            dist = np.sqrt(np.maximum(2 * (1 - vals), 0))
        elif matrix_type == "Covariance":
            d = np.sqrt(np.maximum(np.diag(vals), 1e-8))
            corr = vals / np.outer(d, d)
            corr = np.clip(corr, -1.0, 1.0)
            dist = np.sqrt(np.maximum(2 * (1 - corr), 0))
        elif matrix_type == "Arbitrage - cointegration":
            dist = np.maximum(vals, -10)
            dist = (dist - np.min(dist))
        elif matrix_type == "Arbitrage - halflife":
            dist = np.clip(vals, 0, 100)
        elif matrix_type == "Arbitrage - zscore":
            max_z = np.max(np.abs(vals)) + 1e-8
            sim = np.abs(vals) / max_z
            dist = np.sqrt(np.maximum(2 * (1 - sim), 0))
        elif matrix_type == "Arbitrage - vol_ratio":
            dist = np.sqrt(np.maximum(2 * (1 - vals), 0))
        elif matrix_type == "Arbitrage - arbitrage_score":
            dist = np.sqrt(np.maximum(2 * (1 - vals), 0))
        else:
            # Fallback for unknown or backward compatibility
            v_abs_max = np.max(np.abs(vals)) + 1e-8
            if v_abs_max > 1.001:
                vals = vals / v_abs_max
            vals = np.clip(vals, -1.0, 1.0)
            dist = np.sqrt(np.maximum(2 * (1 - vals), 0))
            
        np.fill_diagonal(dist, 0)
        return pd.DataFrame(dist, index=matrix.index, columns=matrix.columns)

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
        # Max 5 clusters as requested
        n_clusters = max(2, min(n_clusters, 5))

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
    def vol_spillover(vol_matrix, H=10, alpha=0.001):
        import numpy as np
        import networkx as nx
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        # Drop zero-variance columns
        vol_matrix = vol_matrix.loc[:, vol_matrix.std(axis=0) > 1e-10]
        if vol_matrix.empty or vol_matrix.shape[1] < 2:
            return {'edges': [], 'labels': [], 'n_edges': 0, 'spillover': {}, 
                    'in_spillover': {}, 'out_spillover': {}, 'total_spillover_index': 0}

        X = np.log(vol_matrix.values + 1e-8)
        Z = StandardScaler().fit_transform(X)

        T, N = Z.shape
        p = 1  # lag 1

        if T <= p:
            return {'edges': [], 'labels': [], 'n_edges': 0, 'spillover': {}, 
                    'in_spillover': {}, 'out_spillover': {}, 'total_spillover_index': 0}

        Z_curr = Z[p:]
        Z_lag = Z[:-p]

        A = np.zeros((N, N))
        residuals = np.zeros((T-p, N))

        for i in range(N):
            try:
                model = LassoCV(cv=5, fit_intercept=False, max_iter=2000, tol=1e-3, selection='random').fit(Z_lag, Z_curr[:, i])
                A[i, :] = model.coef_
                residuals[:, i] = Z_curr[:, i] - Z_lag @ model.coef_
            except:
                try:
                    coef, _, _, _ = np.linalg.lstsq(Z_lag, Z_curr[:, i], rcond=None)
                    A[i, :] = coef
                    residuals[:, i] = Z_curr[:, i] - Z_lag @ coef
                except:
                    residuals[:, i] = Z_curr[:, i]

        # Companion matrix for lag 1 is just A itself
        companion = A
        Sigma = np.cov(residuals, rowvar=False)
        Sigma += np.eye(N) * 1e-8

        Phi = [np.eye(N)]
        for h in range(1, H):
            Phi.append(Phi[-1] @ companion)

        fevd = np.zeros((N, N))
        for i in range(N):
            denom = 0
            for h in range(H):
                term = Phi[h] @ Sigma @ Phi[h].T
                denom += term[i, i]
            denom = max(denom, 1e-10)

            for j in range(N):
                numer = 0
                for h in range(H):
                    e_i = np.zeros(N); e_i[i] = 1
                    e_j = np.zeros(N); e_j[j] = 1
                    numer += (e_i @ Phi[h] @ Sigma @ e_j) ** 2 / Sigma[j, j]
                fevd[i, j] = numer / denom

        # Normalize
        fevd = np.maximum(fevd, 0)
        row_sums = fevd.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10
        fevd = fevd / row_sums

        labels = vol_matrix.columns.tolist()
        G = nx.DiGraph()
        for i in range(N):
            for j in range(N):
                if i != j and fevd[i, j] > alpha:
                    G.add_edge(labels[j], labels[i], weight=fevd[i, j])

        spillover = {}
        in_spillover = {}
        out_spillover = {}
        for i in range(N):
            to_others = fevd[:, i].sum() - fevd[i, i]
            from_others = fevd[i, :].sum() - fevd[i, i]
            out_spillover[labels[i]] = to_others
            in_spillover[labels[i]] = from_others
            spillover[labels[i]] = to_others - from_others

        edges = [{'source': u, 'target': v, 'weight': d['weight']} for u, v, d in G.edges(data=True)]
        total_spill = 100 * (fevd.sum() - np.trace(fevd)) / N

        return {
            'edges': edges,
            'labels': labels,
            'n_edges': len(edges),
            'spillover': spillover,
            'in_spillover': in_spillover,
            'out_spillover': out_spillover,
            'total_spillover_index': total_spill
        }
    
    @staticmethod
    def return_spillover(return_matrix, H=10, alpha=0.001):
        import numpy as np
        import networkx as nx
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        # Drop zero-variance columns
        return_matrix = return_matrix.loc[:, return_matrix.std(axis=0) > 1e-10]
        if return_matrix.empty or return_matrix.shape[1] < 2:
            return {'edges': [], 'labels': [], 'n_edges': 0, 'spillover': {}, 
                    'in_spillover': {}, 'out_spillover': {}, 'total_spillover_index': 0}

        # Standardize returns
        Z = StandardScaler().fit_transform(return_matrix.values)

        T, N = Z.shape
        p = 1  # lag 1

        if T <= p:
            return {'edges': [], 'labels': [], 'n_edges': 0, 'spillover': {}, 
                    'in_spillover': {}, 'out_spillover': {}, 'total_spillover_index': 0}

        Z_curr = Z[p:]
        Z_lag = Z[:-p]

        A = np.zeros((N, N))
        residuals = np.zeros((T-p, N))

        for i in range(N):
            try:
                model = LassoCV(cv=5, fit_intercept=False, max_iter=2000, tol=1e-3, selection='random').fit(Z_lag, Z_curr[:, i])
                A[i, :] = model.coef_
                residuals[:, i] = Z_curr[:, i] - Z_lag @ model.coef_
            except:
                try:
                    coef, _, _, _ = np.linalg.lstsq(Z_lag, Z_curr[:, i], rcond=None)
                    A[i, :] = coef
                    residuals[:, i] = Z_curr[:, i] - Z_lag @ coef
                except:
                    residuals[:, i] = Z_curr[:, i]

        companion = A
        Sigma = np.cov(residuals, rowvar=False)
        Sigma += np.eye(N) * 1e-8

        Phi = [np.eye(N)]
        for h in range(1, H):
            Phi.append(Phi[-1] @ companion)

        fevd = np.zeros((N, N))
        for i in range(N):
            denom = 0
            for h in range(H):
                term = Phi[h] @ Sigma @ Phi[h].T
                denom += term[i, i]
            denom = max(denom, 1e-10)

            for j in range(N):
                numer = 0
                for h in range(H):
                    e_i = np.zeros(N); e_i[i] = 1
                    e_j = np.zeros(N); e_j[j] = 1
                    numer += (e_i @ Phi[h] @ Sigma @ e_j) ** 2 / Sigma[j, j]
                fevd[i, j] = numer / denom

        # Normalize rows
        row_sums = fevd.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10
        fevd = fevd / row_sums

        labels = return_matrix.columns.tolist()
        G = nx.DiGraph()
        for i in range(N):
            for j in range(N):
                if i != j and fevd[i, j] > alpha:
                    G.add_edge(labels[j], labels[i], weight=fevd[i, j])

        spillover = {}
        in_spillover = {}
        out_spillover = {}
        for i in range(N):
            to_others = fevd[:, i].sum() - fevd[i, i]
            from_others = fevd[i, :].sum() - fevd[i, i]
            out_spillover[labels[i]] = to_others
            in_spillover[labels[i]] = from_others
            spillover[labels[i]] = to_others - from_others

        edges = [{'source': u, 'target': v, 'weight': d['weight']} for u, v, d in G.edges(data=True)]
        total_spill = 100 * (fevd.sum() - np.trace(fevd)) / N

        return {
            'edges': edges,
            'labels': labels,
            'n_edges': len(edges),
            'spillover': spillover,
            'in_spillover': in_spillover,
            'out_spillover': out_spillover,
            'total_spillover_index': total_spill
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
