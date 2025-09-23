import numpy as np
import pandas as pd


def positive_switch_count(y_pred: pd.Series, y_cf: pd.Series):
    """Number of original negative that switched to positive predictions."""
    return ((y_pred == 0) & (y_cf == 1)).sum()


def negative_switch_count(y_pred: pd.Series, y_cf: pd.Series):
    """Number of original positive that switched to negative predictions."""
    return ((y_pred == 1) & (y_cf == 0)).sum()


def positive_switch_rate(y_pred: pd.Series, y_cf: pd.Series):
    """Proportion of original negatives that switched to positive predictions."""
    negatives = (y_pred == 0).sum()
    if negatives == 0:
        return 0.0
    return positive_switch_count(y_pred, y_cf) / negatives


def negative_switch_rate(y_pred: pd.Series, y_cf: pd.Series):
    """Proportion of original positives that switched to negative predictions."""
    positives = (y_pred == 1).sum()
    if positives == 0:
        return 0.0
    return negative_switch_count(y_pred, y_cf) / positives


def true_positive_switch_rate(y_true: pd.Series, y_pred: pd.Series, y_cf: pd.Series):
    """Proportion of true positives that switched to negative predictions"""
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    if true_positives == 0:
        return 0.0
    return ((y_true == 1) & (y_pred == 1) & (y_cf == 0)).sum() / true_positives


def true_negative_switch_rate(y_true: pd.Series, y_pred: pd.Series, y_cf: pd.Series):
    """Proportion of true negatives that switched to positive predictions"""
    true_negatives = ((y_true == 0) & (y_pred == 0)).sum()
    if true_negatives == 0:
        return 0.0
    return ((y_true == 0) & (y_pred == 0) & (y_cf == 1)).sum() / true_negatives
