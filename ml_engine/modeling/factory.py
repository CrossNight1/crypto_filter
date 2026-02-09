from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
import statsmodels.api as sm
import numpy as np

class XGBClassifierWrapper:
    """
    Wrapper for XGBClassifier to handle label encoding automatically.
    XGBoost requires labels in range [0, num_class - 1].
    """
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, X, y, **fit_params):
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.model.fit(X, y_encoded, **fit_params)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __getattr__(self, name):
        # Delegate other attributes to the underlying XGBoost model
        return getattr(self.model, name)

class StatsModelWrapper:
    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs
        self.model = None
        self.params = None

    def fit(self, X, y):
        # Add constant for statsmodels
        X_const = sm.add_constant(X, has_constant='add')
        self.model = self.model_class(y, X_const, **self.kwargs).fit(disp=0)
        return self

    def predict(self, X):
        X_const = sm.add_constant(X, has_constant='add')
        return self.model.predict(X_const)

class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Factory method to create ML models.
        """
        if model_type in ['Random Forest Regressor', 'Random Forest Classifier']:
            is_reg = 'Regressor' in model_type
            cls = RandomForestRegressor if is_reg else RandomForestClassifier
            return cls(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'XGB Regressor':
            return XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42),
                enable_categorical=True
            )
        elif model_type == 'XGB Classifier':
            return XGBClassifierWrapper(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42),
                enable_categorical=True
            )
        elif model_type in ['OLS', 'Linear (OLS)']:
            return StatsModelWrapper(sm.OLS)
        elif model_type == 'Logit':
            return StatsModelWrapper(sm.Logit)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

