
import pandas as pd
import numpy as np
import statsmodels.api as sm

class Predictor:
    """
    Standardized interface for making predictions from trained models.
    """
    
    def __init__(self, model, model_type, scaler=None):
        self.model = model
        self.model_type = model_type
        self.scaler = scaler
        self.is_sklearn = hasattr(model, 'predict')
        self.is_classification = model_type in ['Random Forest Classifier', 'XGB Classifier', 'Logit']

    def predict(self, X_input):
        """
        Make a prediction.
        """
        # 1. Preprocess (Scale)
        X_processed = X_input.copy()
        if self.scaler:
            # If input is DataFrame, keep it as DataFrame with columns
            if isinstance(X_processed, pd.DataFrame):
                scaled_vals = self.scaler.transform(X_processed)
                X_processed = pd.DataFrame(scaled_vals, columns=X_processed.columns, index=X_processed.index)
            else:
                X_processed = self.scaler.transform(X_processed)

        # 2. Add Constant for statsmodels if needed
        if self.model_type in ['OLS', 'Logit', 'Linear (OLS)']:
            X_processed = sm.add_constant(X_processed, has_constant='add')

        # 3. Predict
        result = {}
        
        if self.is_classification:
            if self.model_type in ['Random Forest Classifier', 'XGB Classifier']:
                # Sklearn Classifiers
                prob = self.model.predict_proba(X_processed)
                prediction = self.model.predict(X_processed)[0]
                result['predicted_class'] = int(prediction)
                
                # Confidence handling
                if prob.shape[1] == 2:
                    # Binary
                    p_up = prob[:, 1][0]
                    result['probability'] = p_up
                    result['confidence'] = max(p_up, 1 - p_up)
                    result['signal'] = "UP" if prediction == 1 else "DOWN"
                else:
                    # Multiclass (3 classes expected: Up, Down, Sideways)
                    # We assume labels map to [-1, 0, 1] or similar
                    result['signal'] = str(prediction)
                    # Confidence is max prob
                    result['confidence'] = float(np.max(prob))
                    # If we can map classes to names:
                    mapping = {1: "UP", -1: "DOWN", 0: "SIDEWAYS"}
                    result['signal_name'] = mapping.get(prediction, str(prediction))
                    
            elif self.model_type == 'Logit':
                # Statsmodels Logit (returns probability of class 1 directly)
                p_up = self.model.predict(X_processed)[0]
                prediction = 1 if p_up > 0.5 else 0
                result['probability'] = float(p_up)
                result['predicted_class'] = int(prediction)
                result['confidence'] = max(p_up, 1 - p_up)
                result['signal'] = "UP" if prediction == 1 else "DOWN"

        else:
            # Regression
            pred_val = self.model.predict(X_processed)[0]
            result['predicted_value'] = float(pred_val)
            result['signal'] = "UP" if pred_val > 0 else "DOWN"
            
        return result


class IsotonicCalibrator:
    def fit(self, p, y):
        p = np.asarray(p)
        y = np.asarray(y)

        order = np.argsort(p)
        self.p_sorted = p[order]
        y_sorted = y[order]

        n = len(y_sorted)
        solution = y_sorted.astype(float)
        weight = np.ones(n)

        i = 0
        while i < n - 1:
            if solution[i] > solution[i + 1]:
                total_weight = weight[i] + weight[i + 1]
                avg = (solution[i] * weight[i] + solution[i + 1] * weight[i + 1]) / total_weight
                solution[i] = avg
                solution[i + 1] = avg
                weight[i] = total_weight
                weight[i + 1] = total_weight

                j = i
                while j > 0 and solution[j - 1] > solution[j]:
                    total_weight = weight[j - 1] + weight[j]
                    avg = (solution[j - 1] * weight[j - 1] + solution[j] * weight[j]) / total_weight
                    solution[j - 1] = avg
                    solution[j] = avg
                    weight[j - 1] = total_weight
                    weight[j] = total_weight
                    j -= 1
            i += 1

        self.solution = solution
        return self

    def predict(self, p):
        return np.interp(np.asarray(p), self.p_sorted, self.solution)

class CalibratedModelWrapper:
    def __init__(self, model, calibrators, class_to_index):
        self.model = model
        self.calibrators = calibrators
        self.classes_ = model.classes_
        self.class_to_index = class_to_index
        for attr in ['feature_names_in_', 'feature_importances_', 'coef_']:
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))
            
    def predict_proba(self, X):
        proba = self.model.predict_proba(X).copy()
        for c, cal in self.calibrators.items():
            ci = self.class_to_index[c]
            proba[:, ci] = cal.predict(proba[:, ci])
        
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
