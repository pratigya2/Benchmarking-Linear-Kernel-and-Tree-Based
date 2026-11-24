from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from .config import CAT_COL, NUM_FEATURES, SEED


def build_column_transformer():
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", sparse_output=False), [CAT_COL]),
        ("num", StandardScaler(), NUM_FEATURES),
    ], verbose_feature_names_out=False)


def interaction_steps():
    """Pairwise interactions, then drop constant columns.

    Type_L x Type_M is always 0 (one-hot dummies are mutually exclusive),
    so VarianceThreshold(0.0) removes it.
    """
    return [
        ("interactions",
         PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("drop_constant", VarianceThreshold(0.0)),
    ]


def build_preprocessor(with_interactions=True, with_pca=False, n_components=None):
    steps = [("encode_scale", build_column_transformer())]
    if with_interactions:
        steps.extend(interaction_steps())
    if with_pca:
        steps.append((
            "pca",
            PCA(n_components=n_components or 0.95, random_state=SEED),
        ))
    return Pipeline(steps)


def get_feature_names(fitted_pipeline):
    """Output feature names of a fitted preprocessor (after all steps)."""
    return list(fitted_pipeline.get_feature_names_out())
