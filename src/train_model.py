"""
TeleXAI Model Training

Trains a LightGBM classifier to predict 5G tower failures within a 6-hour horizon.
Enforces strict time-based splitting to prevent temporal data leakage and uses 
class weighting to handle highly imbalanced failure events.
"""

import pandas as pd
import os
import lightgbm as lgb
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
