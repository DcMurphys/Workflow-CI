# ===== IMPORT LIBRARY YANG DIPERLUKAN ===== 
import inspect 
import logging
import json
import os
import tempfile
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature 

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    make_scorer, accuracy_score, roc_auc_score, 
    precision_score, recall_score, f1_score,
    log_loss, precision_recall_curve, average_precision_score, 
    roc_curve, ConfusionMatrixDisplay
) 
from sklearn.utils import estimator_html_repr

# Menghilangkan pesan warning pada output terminal 
import warnings
warnings.filterwarnings("ignore")

# ===== KONFIGURASI LOGGING =====
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ===== Konfigurasi awal MLFlow tracking URI =====
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("customer-churn-prediction-system")

# ===== FUNGSI HELPER TERKAIT UTILITY =====
def get_input_example(X, n_rows=5):
    """
    Memperoleh (GET) input example untuk tujuan MLflow signature logging
    """
    if hasattr(X, "head"):
        return X.head(n_rows)
    
    return X[:n_rows]


def get_feature_names(X):
    """
    Memperoleh (GET) nama fitur/attribute 
    """
    if hasattr(X, "columns"):
        return list(X.columns)
    
    return [f"feature_{i}" for i in range(X.shape[1])]


def sanitize_params(params):
    """
    Mengubah object kompleks menjadi String agar tidak gagal ketika logging
    """
    sanitized = []

    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    
    return sanitized


# ===== FUNGSI HELPER TERKAIT DATA =====
def load_data_csv(file_path):
    """
    Memuat dataset dari file .CSV 
    """
    try:
        data = pd.read_csv(file_path)
        print(f"File {file_path} berhasil dimuat!")
        return data
    except Exception as error: 
        print(f"Terjadi error saat memuat data: {error}")


def get_x_and_y(df, target_col):
    """
    Memperoleh (GET) variabel X dan y dari dataset yang dipilih.

    Hasil kembalian (return): 
    * Variabel X: DataFrame tanpa kolom Target (target_col)
    * Variabel y: DataFrame yang hanya berisikan kolom Target (target_col)
    """
    X = df.drop(columns=target_col)
    y = df[target_col]
    return X, y 


# ===== FUNGSI HELPER TERKAIT MLFLOW MANUAL LOGGING =====
def fit_search(search, X_train, y_train):
    """
    Pencarian model dengan parameter terbaik berdasarkan hasil cross validation scoring
    """
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_score = search.best_score_

    return best_model, best_params, best_cv_score 


def evaluate_model(model, X, y, prefix="training", average="binary"):
    """ 
    Evaluasi model memanfaatkan variabel X dan y, menyesuaikan prefix dataset yang digunakan.
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    metrics = {
        f"{prefix}_score": model.score(X, y),
        f"{prefix}_accuracy_score": accuracy_score(y, y_pred),
        f"{prefix}_precision_score": precision_score(
            y, y_pred,
            average=average,
            zero_division=0
        ),
        f"{prefix}_recall_score": recall_score(
            y, y_pred,
            average=average,
            zero_division=0
        ),           
        f"{prefix}_f1_score": f1_score(
            y, y_pred,
            average=average,
            zero_division=0
        ),
        f"{prefix}_log_loss": log_loss(y, y_pred_proba),
        f"{prefix}_roc_auc": roc_auc_score(
            y, y_pred_proba[:, 1]
        )
    }

    return y_pred, metrics


def log_params(best_params, cv_best_score):
    """
    Membuat logging parameter terbaik dari hasil Cross Validation secara manual
    """
    logger.info("Logging best parameters manually...")
    mlflow.log_params(best_params)
    mlflow.log_metric("cv_best_score", cv_best_score)


def log_metrics(metrics):
    """ 
    Membuat logging metric yang ada secara manual
    """
    logger.info("Logging metrics manually...")
    mlflow.log_metrics({key: value for key, value in metrics.items()})


def log_best_model(model, input_example, model_name):
    """
    Menyimpan hasil logging model terbaik ke MLflow tracking server
    """
    logger.info("Logging best models...")
    prediction_example = model.predict(input_example)
    signature = infer_signature(input_example, prediction_example)
    log_model_signature = inspect.signature(mlflow.sklearn.log_model)
    model_path_argument = (
        "name" if "name" in log_model_signature.parameters else "artifact_path"
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        input_example=input_example,
        signature=signature,
        **{model_path_argument: model_name}
    )


def log_estimator(model, artifact_dir, artifact_name="estimator.html"):
    """
    Membuat logging struktur estimator model sklearn ke MLflow dalam bentuk visualisasi HTML
    """
    logger.info("Logging estimator.html...")
    html_content = estimator_html_repr(model)
    output_path = artifact_dir / artifact_name
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    mlflow.log_artifact(local_path=output_path)


def log_precision_recall_curve(
        y_true, y_scores, 
        prefix="testing", 
        model=None, 
        artifact_name="precision_recall_curve.png"
    ):
    """
    Membuat logging precision-recall ke MLflow dalam bentuk visualisasi kurva
    """
    logger.info(f"Logging {prefix} Precision-Recall curve plot...")
    model_name = type(model).__name__ if model is not None else "Model"
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"{model_name} (AP = {average_precision:.2f})", 
    )

    ax.set_xlim([0.0, 1.0])  
    ax.set_ylim([0.0, 1.05])  
    ax.set_xlabel("Recall (Positive label: 1)", fontsize=12)  
    ax.set_ylabel("Precision (Positive label: 1)", fontsize=12)  
    ax.set_title(f"Precision-recall curve ({prefix})", fontsize=13)  
    ax.legend(loc="lower left", fontsize=10)  
    ax.grid(alpha=0.3)  

    mlflow.log_figure(fig, f"{prefix}_{artifact_name}")
    plt.close(fig)


def log_roc_curve(
        y_true, y_scores, 
        model=None,
        prefix="testing", 
        artifact_name="roc_curve.png"
    ): 
    """
    Membuat logging ROC curve ke MLflow dalam bentuk visualisasi kurva
    """
    logger.info(f"Logging {prefix} ROC curve plot...")
    model_name = type(model).__name__ if model is not None else "Model"

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(  
        fpr,  
        tpr,  
        color="blue",  
        lw=2,  
        label=f"{model_name} (AUC = {auc_score:.2f})",  
    )  

    ax.set_xlim([0.0, 1.0])  
    ax.set_ylim([0.0, 1.05])  
    ax.set_xlabel("False Positive Rate (Positive label: 1)", fontsize=12)  
    ax.set_ylabel("True Positive Rate (Positive label: 1)", fontsize=12)  
    ax.set_title(f"ROC Curve ({prefix})", fontsize=13)  
    ax.legend(loc="lower right", fontsize=10)  
    ax.grid(alpha=0.3)

    mlflow.log_figure(fig, f"{prefix}_{artifact_name}")
    plt.close(fig)


# ===== FUNGSI HELPER UNTUK ARTIFACT MLFLOW TAMBAHAN DI LUAR AUTOLOG ===== 
def log_confusion_matrix(
        y_true, y_scores, 
        prefix="testing", 
        artifact_name="confusion_matrix.png"
    ):
    """
    Menghasilkan logging artifact tambahan berupa Confusion Matrix 
    """
    logger.info(f"Logging {prefix} confusion matrix artifact...")
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_scores, ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    mlflow.log_figure(fig, f"{prefix}_{artifact_name}")
    plt.close(fig)


def log_feature_importance(
        model, feature_names, 
        artifact_dir, 
        artifact_name="feature_importance.csv"
    ):
    """
    Menghasilkan logging artifact tambahan berupa Feature Importance dari model yang dilatih 
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model tidak memiliki feature_importances_. Fungsi di-skip...")
        mlflow.set_tag("feature_importance_logged", "false")
        return

    logger.info("Logging feature importance...")

    importances = model.feature_importances_

    df_feature_importance = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
    )

    output_path = artifact_dir / artifact_name
    df_feature_importance.to_csv(output_path, index=False)

    mlflow.log_artifact(str(output_path), artifact_path="reports")
    mlflow.set_tag("feature_importance_logged", "true")


def log_cv_results(search, artifact_dir, artifact_name="cv_results.csv"):
    """
    Membuat logging artifact tambahan berupa hasil Cross Validation dari model yang dilatih 
    """
    logger.info("Logging cross-validation results...")

    df_cv_results = pd.DataFrame(search.cv_results_)

    output_path = artifact_dir / artifact_name
    df_cv_results.to_csv(output_path, index=False)

    mlflow.log_artifact(str(output_path), artifact_path="reports")


def log_config_snapshot(
        model, search, 
        X_train, X_test, 
        best_params, artifact_dir, 
        artifact_name="model_config.json"
    ):
    """
    Membuat logging artifact tambahan berupa configuration snapshot dari model yang dilatih.

    Snapshot ini dapat berupa parameter, proporsi pembagian data, dan jenis model yang digunakan. 
    """
    logger.info("Logging model configuration snapshot...")

    config_snapshot = {
        "model_type": type(model).__name__,
        "model_repr": str(model),
        "best_params": best_params,
        "training_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "search_type": type(search).__name__,
        "search_scoring": getattr(search, "scoring", None),
        "search_cv": getattr(search, "cv", None),
        "search_n_iter": getattr(search, "n_iter", None),
        "random_state": getattr(search, "random_state", None)
    }

    output_path = artifact_dir / artifact_name

    with open(output_path, "w") as file:
        json.dump(config_snapshot, file, indent=4, default=str)

    mlflow.log_artifact(str(output_path), artifact_path="reports")


# ===== FUNGSI HELPER TERKAIT EKSEKUSI TRACKING EXPERIMENT =====
def run_tracked_experiment(
        search, 
        X_train, y_train, 
        X_test, y_test, 
        run_name, 
        average
    ):
    """
    Menjalankan proses eksperimen pelatihan model dengan memanfaatkan teknik K-Fold Cross Validation.
    
    Kemudian, hasil eksperimen dilacak dan direkam MLflow. 
    """
    input_example = get_input_example(X_train)
    feature_names = get_feature_names(X_train)

    with mlflow.start_run(run_name=run_name):
        try:
            mlflow.set_tags({
                "tracking_type": "manual_logging",
                "run_purpose": "kfold_randomized_search",
                "status": "running"
            })

            best_model, best_params, cv_best_score = fit_search(
                search,
                X_train,
                y_train
            )
            
            y_train_pred, train_metrics = evaluate_model(
                best_model,
                X_train, y_train,
                prefix="train", average=average
            )

            y_test_pred, test_metrics = evaluate_model(
                best_model,
                X_test, y_test,
                prefix="test", average=average
            )

            log_params(best_params=best_params, cv_best_score=cv_best_score)
            log_metrics(metrics=train_metrics)
            log_metrics(metrics=test_metrics)

            log_best_model(
                model=best_model,
                input_example=input_example,
                model_name="model"
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                artifact_dir = Path(temp_dir)
                
                # Log estimator.html
                log_estimator(
                    model=best_model,
                    artifact_dir=artifact_dir
                )

                # Log Confusion Matrix untuk train set dan test set
                log_confusion_matrix(
                    y_true=y_train,
                    y_scores=y_train_pred,
                    prefix="training"
                )

                log_confusion_matrix(
                    y_true=y_test,
                    y_scores=y_test_pred,
                    prefix="testing"
                )

                # Log Precision-Recall Curve plot untuk train set dan test set
                log_precision_recall_curve(
                    y_true=y_train, y_scores=y_train_pred,
                    prefix="training",
                    model=best_model
                )

                log_precision_recall_curve(
                    y_true=y_test, y_scores=y_test_pred,
                    prefix="testing",
                    model=best_model
                )

                # Log ROC Curve plot untuk train set dan test set
                log_roc_curve(
                    y_true=y_train, y_scores=y_train_pred,
                    prefix="training",
                    model=best_model
                )

                log_roc_curve(
                    y_true=y_test, y_scores=y_test_pred,
                    prefix="testing",
                    model=best_model
                )

                # Log feature importance.csv
                log_feature_importance(
                    model=best_model,
                    feature_names=feature_names,
                    artifact_dir=artifact_dir
                )

                # Log cv_results.csv 
                log_cv_results(
                    search=search,
                    artifact_dir=artifact_dir
                )

                # Log config_snapshot.json
                log_config_snapshot(
                    model=best_model,
                    search=search,
                    X_train=X_train,
                    X_test=X_test,
                    best_params=best_params,
                    artifact_dir=artifact_dir
                )

            mlflow.set_tag("status", "success")

            logger.info("Experiment run completed successfully.")

            return {
                "best_model": best_model,
                "best_params": best_params,
                "cv_best_score": cv_best_score,
                "train_metrics": train_metrics, 
                "test_metrics": test_metrics
            }

        except Exception as error:
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error_type", type(error).__name__)
            mlflow.set_tag("error_message", str(error))

            logger.exception("Experiment run failed.")

            raise


# ===== EKSEKUSI TRACKING EXPERIMENT TUNED MODEL DENGAN MLFLOW (GITHUB ACTIONS) =====
# Memperoleh nilai random state untuk menjaga keacakan nilai pada setiap skenario
RANDOM_STATE = 126 

# Membuat parsing argument pada variabel-variabel untuk proses pelatihan
parser = argparse.ArgumentParser()  
parser.add_argument("--train_path",   type=str)  
parser.add_argument("--test_path",    type=str)  
parser.add_argument("--target_col",   type=str)  
parser.add_argument("--n_iter",       type=int)  
parser.add_argument("--n_splits",     type=int)  
args = parser.parse_args()  

# Melakukan load data latih dan data uji 
train_path = "MLProject/telco_preprocessing/train_pca.csv"
test_path = "MLProject/telco_preprocessing/test_pca.csv"
df_train = load_data_csv(train_path)
df_test = load_data_csv(test_path)

# Mengambil variabel X dan y dari data latih dan data uji 
X_train, y_train = get_x_and_y(df_train, target_col="Churn Label")
X_test, y_test = get_x_and_y(df_test, target_col="Churn Label")

# Menentukan ruang pencarian parameter dengan randomized grid search
param_dists = {
    "n_estimators": [int(x) for x in np.linspace(start=50, stop=200, num=10)],
    "learning_rate": [0.001, 0.01, 0.05, 0.1],
    "max_depth": [int(x) for x in np.linspace(start=2, stop=16, num=4)],
    "min_samples_split": [int(x) for x in np.linspace(start=2, stop=16, num=4)],
    "min_samples_leaf": [int(x) for x in np.linspace(start=1, stop=8, num=2)],
    "subsample": [0.6, 0.8, 1.0],
    "max_features": [None, "sqrt", "log2"]
}

# Menentukan skor acuan untuk ruang pencarian parameter yang optimal
scoring = {
    "accuracy": "accuracy",
    "f1_score": make_scorer(f1_score)
}

# Instansiasi StratifiedKFold Cross Validation sebanyak 25 split
cv = StratifiedKFold(n_splits=25, shuffle=True, random_state=RANDOM_STATE)

# Membangun model GradientBoostingClassifier 
model = GradientBoostingClassifier(random_state=RANDOM_STATE)

# Menjalankan hyperparameter tuning dengan Cross Validation + RandomizedSearch 
search = RandomizedSearchCV(
     estimator=model,
     param_distributions=param_dists,
     n_iter=25,
     scoring=scoring,
     refit="f1_score", 
     cv=cv,
     verbose=1,
     n_jobs=-1,
     random_state=RANDOM_STATE
)

# Melacak eksperimen model hasil tuning dengan MLflow manual log 
mlflow.autolog(disable=True)
result = run_tracked_experiment(
    search=search,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    run_name="gradient_boost_tuned",
    average="binary"
)