#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Baseline Models Module
베이스라인 모델 (로지스틱 회귀, XGBoost) 구현

Functions:
- train_logistic_regression: 로지스틱 회귀 모델 훈련
- train_xgboost: XGBoost 모델 훈련
- evaluate_model: 모델 평가
- compare_models: 모델 성과 비교
- generate_baseline_report: 베이스라인 리포트 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# XGBoost가 설치되어 있지 않을 수 있으므로 try-except로 처리
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost가 설치되지 않았습니다. pip install xgboost로 설치하세요.")

def train_logistic_regression(X_train: np.ndarray, 
                             y_train: np.ndarray,
                             X_valid: np.ndarray, 
                             y_valid: np.ndarray,
                             **kwargs) -> Dict[str, Any]:
    """
    로지스틱 회귀 모델 훈련
    
    Args:
        X_train: 훈련 피처 (3D 배열)
        y_train: 훈련 라벨
        X_valid: 검증 피처 (3D 배열)
        y_valid: 검증 라벨
        **kwargs: 추가 하이퍼파라미터
        
    Returns:
        훈련된 모델과 결과 딕셔너리
    """
    print("🔧 로지스틱 회귀 모델 훈련 시작...")
    
    # 3D 배열을 2D로 변환 (시계열 데이터를 평면화)
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_valid_2d = X_valid.reshape(X_valid.shape[0], -1)
    
    # 모델 생성
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        **kwargs
    )
    
    # 모델 훈련
    model.fit(X_train_2d, y_train)
    
    # 예측
    y_train_pred = model.predict(X_train_2d)
    y_valid_pred = model.predict(X_valid_2d)
    y_train_proba = model.predict_proba(X_train_2d)[:, 1]
    y_valid_proba = model.predict_proba(X_valid_2d)[:, 1]
    
    # 평가
    train_metrics = evaluate_model(y_train, y_train_pred, y_train_proba, "Train")
    valid_metrics = evaluate_model(y_valid, y_valid_pred, y_valid_proba, "Valid")
    
    results = {
        'model': model,
        'model_name': 'Logistic Regression',
        'train_metrics': train_metrics,
        'valid_metrics': valid_metrics,
        'feature_importance': None,  # 로지스틱 회귀는 특성 중요도가 제한적
        'predictions': {
            'train_pred': y_train_pred,
            'valid_pred': y_valid_pred,
            'train_proba': y_train_proba,
            'valid_proba': y_valid_proba
        }
    }
    
    print("✅ 로지스틱 회귀 모델 훈련 완료!")
    return results

def train_xgboost(X_train: np.ndarray, 
                  y_train: np.ndarray,
                  X_valid: np.ndarray, 
                  y_valid: np.ndarray,
                  **kwargs) -> Dict[str, Any]:
    """
    XGBoost 모델 훈련
    
    Args:
        X_train: 훈련 피처 (3D 배열)
        y_train: 훈련 라벨
        X_valid: 검증 피처 (3D 배열)
        y_valid: 검증 라벨
        **kwargs: 추가 하이퍼파라미터
        
    Returns:
        훈련된 모델과 결과 딕셔너리
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost가 설치되지 않았습니다. pip install xgboost로 설치하세요.")
    
    print("🔧 XGBoost 모델 훈련 시작...")
    
    # 3D 배열을 2D로 변환
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_valid_2d = X_valid.reshape(X_valid.shape[0], -1)
    
    # 기본 하이퍼파라미터
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # 사용자 파라미터로 업데이트
    default_params.update(kwargs)
    
    # 모델 생성
    model = xgb.XGBClassifier(**default_params)
    
    # 모델 훈련
    model.fit(
        X_train_2d, y_train,
        eval_set=[(X_valid_2d, y_valid)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # 예측
    y_train_pred = model.predict(X_train_2d)
    y_valid_pred = model.predict(X_valid_2d)
    y_train_proba = model.predict_proba(X_train_2d)[:, 1]
    y_valid_proba = model.predict_proba(X_valid_2d)[:, 1]
    
    # 평가
    train_metrics = evaluate_model(y_train, y_train_pred, y_train_proba, "Train")
    valid_metrics = evaluate_model(y_valid, y_valid_pred, y_valid_proba, "Valid")
    
    # 특성 중요도
    feature_importance = model.feature_importances_
    
    results = {
        'model': model,
        'model_name': 'XGBoost',
        'train_metrics': train_metrics,
        'valid_metrics': valid_metrics,
        'feature_importance': feature_importance,
        'predictions': {
            'train_pred': y_train_pred,
            'valid_pred': y_valid_pred,
            'train_proba': y_train_proba,
            'valid_proba': y_valid_proba
        }
    }
    
    print("✅ XGBoost 모델 훈련 완료!")
    return results

def train_random_forest(X_train: np.ndarray, 
                       y_train: np.ndarray,
                       X_valid: np.ndarray, 
                       y_valid: np.ndarray,
                       **kwargs) -> Dict[str, Any]:
    """
    랜덤 포레스트 모델 훈련 (XGBoost 대안)
    
    Args:
        X_train: 훈련 피처 (3D 배열)
        y_train: 훈련 라벨
        X_valid: 검증 피처 (3D 배열)
        y_valid: 검증 라벨
        **kwargs: 추가 하이퍼파라미터
        
    Returns:
        훈련된 모델과 결과 딕셔너리
    """
    print("🔧 랜덤 포레스트 모델 훈련 시작...")
    
    # 3D 배열을 2D로 변환
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_valid_2d = X_valid.reshape(X_valid.shape[0], -1)
    
    # 기본 하이퍼파라미터
    default_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 사용자 파라미터로 업데이트
    default_params.update(kwargs)
    
    # 모델 생성
    model = RandomForestClassifier(**default_params)
    
    # 모델 훈련
    model.fit(X_train_2d, y_train)
    
    # 예측
    y_train_pred = model.predict(X_train_2d)
    y_valid_pred = model.predict(X_valid_2d)
    y_train_proba = model.predict_proba(X_train_2d)[:, 1]
    y_valid_proba = model.predict_proba(X_valid_2d)[:, 1]
    
    # 평가
    train_metrics = evaluate_model(y_train, y_train_pred, y_train_proba, "Train")
    valid_metrics = evaluate_model(y_valid, y_valid_pred, y_valid_proba, "Valid")
    
    # 특성 중요도
    feature_importance = model.feature_importances_
    
    results = {
        'model': model,
        'model_name': 'Random Forest',
        'train_metrics': train_metrics,
        'valid_metrics': valid_metrics,
        'feature_importance': feature_importance,
        'predictions': {
            'train_pred': y_train_pred,
            'valid_pred': y_valid_pred,
            'train_proba': y_train_proba,
            'valid_proba': y_valid_proba
        }
    }
    
    print("✅ 랜덤 포레스트 모델 훈련 완료!")
    return results

def evaluate_model(y_true: np.ndarray, 
                   y_pred: np.ndarray, 
                   y_proba: np.ndarray, 
                   dataset_name: str) -> Dict[str, Any]:
    """
    모델 평가
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        y_proba: 예측 확률
        dataset_name: 데이터셋 이름
        
    Returns:
        평가 지표 딕셔너리
    """
    # 기본 지표
    accuracy = np.mean(y_true == y_pred)
    
    # 분류 리포트
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = 0.0
    
    # 정밀도, 재현율, F1
    precision = report['1']['precision'] if '1' in report else 0.0
    recall = report['1']['recall'] if '1' in report else 0.0
    f1 = report['1']['f1-score'] if '1' in report else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    print(f"📊 {dataset_name} 성과:")
    print(f"   - 정확도: {accuracy:.3f}")
    print(f"   - 정밀도: {precision:.3f}")
    print(f"   - 재현율: {recall:.3f}")
    print(f"   - F1 점수: {f1:.3f}")
    print(f"   - ROC AUC: {roc_auc:.3f}")
    
    return metrics

def compare_models(model_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    모델 성과 비교
    
    Args:
        model_results: 모델 결과 리스트
        
    Returns:
        비교 결과 데이터프레임
    """
    comparison_data = []
    
    for result in model_results:
        model_name = result['model_name']
        valid_metrics = result['valid_metrics']
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': valid_metrics['accuracy'],
            'Precision': valid_metrics['precision'],
            'Recall': valid_metrics['recall'],
            'F1 Score': valid_metrics['f1_score'],
            'ROC AUC': valid_metrics['roc_auc']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('ROC AUC', ascending=False)
    
    return df_comparison

def generate_baseline_report(model_results: List[Dict[str, Any]], 
                           comparison_df: pd.DataFrame) -> str:
    """
    베이스라인 리포트 생성
    
    Args:
        model_results: 모델 결과 리스트
        comparison_df: 모델 비교 데이터프레임
        
    Returns:
        리포트 문자열
    """
    report = f"""
📊 SectorFlow Lite - Baseline Models 리포트
{'='*60}

🏆 모델 성과 비교
{comparison_df.to_string(index=False)}

📈 상세 성과 분석
"""
    
    for result in model_results:
        model_name = result['model_name']
        train_metrics = result['train_metrics']
        valid_metrics = result['valid_metrics']
        
        report += f"""
{model_name}
{'-'*30}
훈련 데이터:
  - 정확도: {train_metrics['accuracy']:.3f}
  - 정밀도: {train_metrics['precision']:.3f}
  - 재현율: {train_metrics['recall']:.3f}
  - F1 점수: {train_metrics['f1_score']:.3f}
  - ROC AUC: {train_metrics['roc_auc']:.3f}

검증 데이터:
  - 정확도: {valid_metrics['accuracy']:.3f}
  - 정밀도: {valid_metrics['precision']:.3f}
  - 재현율: {valid_metrics['recall']:.3f}
  - F1 점수: {valid_metrics['f1_score']:.3f}
  - ROC AUC: {valid_metrics['roc_auc']:.3f}
"""
    
    # 최고 성과 모델
    best_model = comparison_df.iloc[0]
    report += f"""

🥇 최고 성과 모델: {best_model['Model']}
   - ROC AUC: {best_model['ROC AUC']:.3f}
   - F1 Score: {best_model['F1 Score']:.3f}

{'='*60}
"""
    
    return report

def train_all_baselines(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    모든 베이스라인 모델 훈련
    
    Args:
        processed_data: 처리된 데이터 딕셔너리
        
    Returns:
        모든 모델 결과 딕셔너리
    """
    print("🚀 모든 베이스라인 모델 훈련 시작...")
    
    all_results = {}
    
    for symbol, data in processed_data.items():
        print(f"\n📊 {symbol} 모델 훈련 중...")
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_valid = data['X_valid']
        y_valid = data['y_valid']
        
        symbol_results = []
        
        # 1. 로지스틱 회귀
        try:
            lr_result = train_logistic_regression(X_train, y_train, X_valid, y_valid)
            symbol_results.append(lr_result)
        except Exception as e:
            print(f"   ❌ 로지스틱 회귀 실패: {e}")
        
        # 2. 랜덤 포레스트
        try:
            rf_result = train_random_forest(X_train, y_train, X_valid, y_valid)
            symbol_results.append(rf_result)
        except Exception as e:
            print(f"   ❌ 랜덤 포레스트 실패: {e}")
        
        # 3. XGBoost (사용 가능한 경우)
        if XGBOOST_AVAILABLE:
            try:
                xgb_result = train_xgboost(X_train, y_train, X_valid, y_valid)
                symbol_results.append(xgb_result)
            except Exception as e:
                print(f"   ❌ XGBoost 실패: {e}")
        
        if symbol_results:
            # 모델 비교
            comparison_df = compare_models(symbol_results)
            
            all_results[symbol] = {
                'models': symbol_results,
                'comparison': comparison_df,
                'best_model': symbol_results[0]  # ROC AUC 기준으로 정렬된 첫 번째
            }
            
            print(f"   ✅ {symbol}: {len(symbol_results)}개 모델 훈련 완료")
        else:
            print(f"   ❌ {symbol}: 모든 모델 훈련 실패")
    
    print(f"\n✅ 총 {len(all_results)}개 종목 모델 훈련 완료!")
    return all_results

def main():
    """테스트용 메인 함수"""
    print("🚀 SectorFlow Lite - Baseline Models Module 테스트")
    print("=" * 60)
    
    # dataio.py에서 데이터 로드 (간단한 테스트용)
    from dataio import prepare_ml_data
    
    # 설정
    config = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'lookback': 30,
        'feature_cols': ['close', 'volume', 'trading_value', 'returns', 'ma_5', 'ma_20', 'volatility'],
        'scale_method': 'standard'
    }
    
    symbols = ['005930', '000660']  # 테스트용 2개 종목
    
    # 데이터 준비
    processed_data = prepare_ml_data(symbols, config)
    
    # 모든 베이스라인 모델 훈련
    all_results = train_all_baselines(processed_data)
    
    # 전체 리포트 생성
    print("\n📊 전체 모델 성과 요약:")
    for symbol, results in all_results.items():
        print(f"\n{symbol}:")
        print(results['comparison'].to_string(index=False))
    
    print("\n✅ Baseline Models Module 테스트 완료!")
    return all_results

if __name__ == "__main__":
    results = main()
