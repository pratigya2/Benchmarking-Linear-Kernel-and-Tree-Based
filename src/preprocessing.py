from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.decomposition import PCA

from .config import CAT_COL, NUM_FEATURES, SEED


def build_column_transformer():
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", sparse_output=False), [CAT_COL]),
        ("num", StandardScaler(), NUM_FEATURES),
    ])


def build_preprocessor(with_interactions=True, with_pca=False, n_components=None):
    steps = [("encode_scale", build_column_transformer())]
    if with_interactions:
        steps.append((
            "interactions",
            PolynomialFeatures(degree=2, interaction_only=True,
                               include_bias=False),
        ))
    if with_pca:
        steps.append((
            "pca",
            PCA(n_components=n_components or 0.95, random_state=SEED),
        ))
    return Pipeline(steps)


def get_feature_names(fitted_pipeline):
    ct = fitted_pipeline.named_steps["encode_scale"]
    cat_names = list(ct.named_transformers_["cat"].get_feature_names_out([CAT_COL]))
    base_names = cat_names + NUM_FEATURES
    if "interactions" in fitted_pipeline.named_steps:
        poly = fitted_pipeline.named_steps["interactions"]
        return list(poly.get_feature_names_out(base_names))
    return base_names
