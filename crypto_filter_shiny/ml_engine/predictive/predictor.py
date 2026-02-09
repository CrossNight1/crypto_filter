
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

