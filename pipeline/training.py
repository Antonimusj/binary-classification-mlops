import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from config.settings import LOGREG_PARAMS, RF_PARAMS, NB_PARAMS

logger = logging.getLogger(__name__)


def train_logreg(X_train, y_train):
    """Instantiates and fits a Logistic Regression model using configured parameters."""
    logger.info("\033[1;34m[FIT] Fitting Logistic Regression model...\033[0m")
    model = LogisticRegression(**LOGREG_PARAMS)
    model.fit(X_train, y_train)
    logger.info("\033[1;32m[SUCCESS] Logistic Regression fitted successfully.\033[0m")
    return model


def train_rndforest(X_train, y_train):
    """Instantiates and fits a Random Forest Classifier model using configured parameters."""
    logger.info("\033[1;34m[FIT] Fitting Random Forest model...\033[0m")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    logger.info("\033[1;32m[SUCCESS] Random Forest fitted successfully.\033[0m")
    return model


def train_bayes(X_train, y_train):
    """Instantiates and fits a Gaussian Naive Bayes model using configured parameters."""
    logger.info("\033[1;34m[FIT] Fitting Naive Bayes model...\033[0m")
    model = GaussianNB(**NB_PARAMS)
    model.fit(X_train, y_train)
    logger.info("\033[1;32m[SUCCESS] Naive Bayes fitted successfully.\033[0m")
    return model


def predict_model(x_test, model):
    """Generates class predictions using the provided trained model instance."""
    logger.info("\033[1;34m[PREDICT] Generating predictions with model: %s...\033[0m", type(model).__name__)
    predictions = model.predict(x_test)
    logger.info("\033[1;32m[SUCCESS] Predictions generated successfully.\033[0m")
    return predictions