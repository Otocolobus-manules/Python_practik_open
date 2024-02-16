import pandas as pd
from Scripts.Preprocessing import RegressionDiamondPreprocessing, ClassificationTransactionPreprocessing


def format_reg(data, model_pr=RegressionDiamondPreprocessing):
    regression_pr = model_pr(data)
    regression_pr.all_preprocess()
    return regression_pr()


def format_class(data, model_pr=ClassificationTransactionPreprocessing):
    classification_pr = model_pr(data=data)
    classification_pr.all_preprocess(int_type=["repeat_retailer", "used_chip", "used_pin_number",
                                               "online_order", "fraud"])
    return classification_pr()
