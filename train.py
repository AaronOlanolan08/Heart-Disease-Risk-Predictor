from pathlib import Path

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from model_utils import make_safe_feature_name
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, RobustScaler

RANDOM_STATE = 42
TEST_SIZES = (0.2, 0.3)
CV_FOLDS = 3
IMPORTANCE_COVERAGE = 0.95

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "dataset" / "heart_2020_uncleaned.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
PREPROCESSING_PATH = MODEL_DIR / "preprocessing.joblib"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
COMPARISON_PATH = MODEL_DIR / "test_size_comparison.csv"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.joblib"

NUMERIC_FEATURES = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
CATEGORICAL_FEATURES = ["AgeCategory", "Race", "Diabetic", "GenHealth"]
BINARY_FEATURES = [
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "PhysicalActivity",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]
BINARY_MAP = {"yes": 1, "no": 0, "male": 1, "female": 0}


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)

    for column in ("BMI", "SleepTime", "PhysicalHealth", "MentalHealth"):
        df[column] = pd.to_numeric(df[column], errors="raise")

    for column in BINARY_FEATURES:
        values = df[column].astype(str).str.lower().map(BINARY_MAP)
        invalid_values = df[column].notna() & values.isna()
        if invalid_values.any():
            raise ValueError(f"Unexpected values in binary feature {column!r}")
        df[column] = values

    y = df.pop("HeartDisease").map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("HeartDisease contains values other than 'Yes' or 'No'")
    return df, y.astype(int)


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    numeric_features = [name for name in NUMERIC_FEATURES if name in features]
    categorical_features = [name for name in CATEGORICAL_FEATURES if name in features]
    binary_features = [name for name in BINARY_FEATURES if name in features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                feature_name_combiner=make_safe_feature_name,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    if binary_features:
        transformers.append(
            (
                "binary",
                SimpleImputer(strategy="most_frequent"),
                binary_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.set_output(transform="pandas")
    return preprocessor


def map_transformed_features(
    transformed_names: list[str], raw_features: list[str]
) -> list[str]:
    mapped_features = []
    for transformed_name in transformed_names:
        _, _, encoded_name = transformed_name.partition("__")
        matches = [
            feature
            for feature in raw_features
            if encoded_name == feature or encoded_name.startswith(f"{feature}_")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Unable to map transformed feature {transformed_name!r} "
                "to exactly one input feature"
            )
        mapped_features.append(matches[0])
    return mapped_features


def evaluate_test_size(
    X: pd.DataFrame, y: pd.Series, test_size: float
) -> dict[str, object]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    training_pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(list(X.columns))),
            ("undersampler", RandomUnderSampler(random_state=RANDOM_STATE)),
            (
                "classifier",
                LGBMClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=-1,
                    importance_type="gain",
                ),
            ),
        ]
    )
    parameter_grid = {
        "undersampler__sampling_strategy": [1.0],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [-1, 10],
        "classifier__num_leaves": [15, 31],
    }
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    search = GridSearchCV(
        estimator=training_pipeline,
        param_grid=parameter_grid,
        scoring="average_precision",
        cv=cross_validation,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]
    transformed_names = fitted_preprocessor.get_feature_names_out().tolist()
    source_features = map_transformed_features(transformed_names, list(X.columns))

    importance_by_feature = pd.DataFrame(
        {"feature": source_features, "importance": classifier.feature_importances_}
    ).groupby("feature", as_index=False)["importance"].sum()
    importance_by_feature = importance_by_feature.sort_values(
        "importance", ascending=False, kind="stable"
    ).reset_index(drop=True)

    total_importance = importance_by_feature["importance"].sum()
    if total_importance <= 0:
        raise ValueError("The tuned model produced no feature importance")
    importance_by_feature["importance_share"] = (
        importance_by_feature["importance"] / total_importance
    )
    importance_by_feature["cumulative_importance"] = (
        importance_by_feature["importance_share"].cumsum()
    )
    importance_by_feature["selected"] = (
        importance_by_feature["cumulative_importance"].shift(fill_value=0)
        < IMPORTANCE_COVERAGE
    )
    selected_features = importance_by_feature.loc[
        importance_by_feature["selected"], "feature"
    ].tolist()

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    decision_threshold = 0.5

    final_preprocessor = make_preprocessor(selected_features)
    X_train_processed = final_preprocessor.fit_transform(X_train_selected)
    X_test_processed = final_preprocessor.transform(X_test_selected)
    X_train_resampled, y_train_resampled = RandomUnderSampler(
        random_state=RANDOM_STATE
    ).fit_resample(X_train_processed, y_train)
    final_model = LGBMClassifier(**classifier.get_params())
    final_model.fit(X_train_resampled, y_train_resampled)

    probabilities = final_model.predict_proba(X_test_processed)[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)
    metrics = {
        "test_size": test_size,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "decision_threshold": decision_threshold,
        "cv_pr_auc": search.best_score_,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
    }
    return {
        "metrics": metrics,
        "best_params": search.best_params_,
        "selected_features": selected_features,
        "feature_importance": importance_by_feature,
        "model": final_model,
        "preprocessor": final_preprocessor,
        "y_test": y_test,
        "predictions": predictions,
        "threshold": decision_threshold,
    }


def main() -> None:
    X, y = load_dataset()
    results = [evaluate_test_size(X, y, size) for size in TEST_SIZES]
    comparison = pd.DataFrame([result["metrics"] for result in results])
    comparison = comparison.sort_values(
        ["recall", "precision", "balanced_accuracy"], ascending=False
    ).reset_index(drop=True)
    best_test_size = float(comparison.loc[0, "test_size"])
    best_result = next(
        result for result in results if result["metrics"]["test_size"] == best_test_size
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)
    best_result["feature_importance"].to_csv(IMPORTANCE_PATH, index=False)
    joblib.dump(best_result["model"], MODEL_PATH)
    joblib.dump(best_result["preprocessor"], PREPROCESSING_PATH)
    joblib.dump(best_result["threshold"], THRESHOLD_PATH)

    print("Majority-class undersampling is enabled; SMOTE and class weights are not used.")
    print("Holdout comparison (ranked by recall, then precision and balanced accuracy):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Best hyperparameters: {best_result['best_params']}")
    print(
        f"Selected {len(best_result['selected_features'])} of {len(X.columns)} "
        f"original features to cover {IMPORTANCE_COVERAGE:.0%} of gain importance:"
    )
    print(", ".join(best_result["selected_features"]))
    print(f"Feature importance saved to {IMPORTANCE_PATH}")
    print(f"Split comparison saved to {COMPARISON_PATH}")
    print(f"Retaining the {best_test_size:.0%} test split by held-out recall.")
    print(f"Decision threshold: {best_result['threshold']:.2f}")
    print("Holdout classification report:")
    print(
        classification_report(
            best_result["y_test"],
            best_result["predictions"],
            target_names=["No heart disease", "Heart disease"],
            zero_division=0,
        )
    )
    print(f"Model saved to {MODEL_PATH}")
    print(f"Preprocessor saved to {PREPROCESSING_PATH}")
    print(f"Decision threshold saved to {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
