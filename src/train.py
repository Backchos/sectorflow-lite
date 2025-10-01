#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Training Module
딥러닝 모델 훈련 및 평가

Functions:
- train_model: 모델 훈련
- evaluate_model: 모델 평가
- handle_class_imbalance: 클래스 불균형 처리
- cross_validate: 교차 검증
- save_model: 모델 저장
- load_model: 모델 로드
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import os
import json
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path: str = "config.yaml") -> dict:
    """설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ 설정 파일을 찾을 수 없습니다: {config_path}")
        return {}
    except Exception as e:
        print(f"❌ 설정 파일 로드 오류: {e}")
        return {}

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    from sklearn.utils.class_weight import compute_class_weight
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow가 설치되지 않았습니다.")

# 기존 모듈들 import
from gru import create_model_factory, create_callbacks
from dataio import prepare_ml_data

def handle_class_imbalance(y_train: np.ndarray, 
                          method: str = 'class_weight') -> Dict[str, Any]:
    """
    클래스 불균형 처리
    
    Args:
        y_train: 훈련 라벨
        method: 처리 방법 ('class_weight', 'smote', 'undersample')
        
    Returns:
        처리 결과 딕셔너리
    """
    print(f"🔧 클래스 불균형 처리 중 ({method})...")
    
    # 클래스 분포 확인
    unique_classes, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique_classes, counts))
    
    print(f"   - 클래스 분포: {class_distribution}")
    
    result = {
        'method': method,
        'original_distribution': class_distribution,
        'class_weights': None,
        'y_train_processed': y_train
    }
    
    if method == 'class_weight':
        # 클래스 가중치 계산
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))
        result['class_weights'] = class_weight_dict
        
        print(f"   - 클래스 가중치: {class_weight_dict}")
    
    elif method == 'smote':
        # SMOTE 적용 (추후 구현)
        print("   - SMOTE는 추후 구현 예정")
        pass
    
    elif method == 'undersample':
        # 언더샘플링 (추후 구현)
        print("   - 언더샘플링은 추후 구현 예정")
        pass
    
    print("✅ 클래스 불균형 처리 완료!")
    return result

def train_model(model, 
                X_train: np.ndarray, 
                y_train: np.ndarray,
                X_valid: np.ndarray, 
                y_valid: np.ndarray,
                config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    모델 훈련
    
    Args:
        model: 훈련할 모델
        X_train: 훈련 피처
        y_train: 훈련 라벨
        X_valid: 검증 피처
        y_valid: 검증 라벨
        config: 훈련 설정
        
    Returns:
        훈련 결과 딕셔너리
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    print("🚀 모델 훈련 시작...")
    
    # 기본 설정
    if config is None:
        config = {
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'patience': 15,
            'learning_rate': 0.001,
            'class_imbalance_method': 'class_weight'
        }
    
    # 클래스 불균형 처리
    imbalance_result = handle_class_imbalance(y_train, config.get('class_imbalance_method', 'class_weight'))
    
    # 콜백 생성
    callbacks = create_callbacks(
        patience=config.get('patience', 15),
        min_delta=config.get('min_delta', 0.001),
        factor=config.get('factor', 0.5),
        min_lr=config.get('min_lr', 1e-7)
    )
    
    # 훈련 시작
    start_time = datetime.now()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 32),
        callbacks=callbacks,
        class_weight=imbalance_result['class_weights'],
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # 훈련 결과
    training_result = {
        'model': model,
        'history': history.history,
        'training_time': training_time,
        'epochs_trained': len(history.history['loss']),
        'best_epoch': np.argmin(history.history['val_loss']) + 1,
        'best_val_loss': min(history.history['val_loss']),
        'best_val_accuracy': max(history.history['val_accuracy']),
        'class_imbalance_result': imbalance_result,
        'config': config
    }
    
    print(f"✅ 모델 훈련 완료! (소요시간: {training_time:.1f}초)")
    print(f"   - 훈련 에포크: {training_result['epochs_trained']}")
    print(f"   - 최고 검증 정확도: {training_result['best_val_accuracy']:.4f}")
    print(f"   - 최저 검증 손실: {training_result['best_val_loss']:.4f}")
    
    return training_result

def evaluate_model(model, 
                   X_test: np.ndarray, 
                   y_test: np.ndarray,
                   threshold: float = 0.5) -> Dict[str, Any]:
    """
    모델 평가
    
    Args:
        model: 평가할 모델
        X_test: 테스트 피처
        y_test: 테스트 라벨
        threshold: 분류 임계값
        
    Returns:
        평가 결과 딕셔너리
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    print("📊 모델 평가 중...")
    
    # 예측
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > threshold).astype(int).flatten()
    
    # 기본 지표 계산
    accuracy = np.mean(y_test == y_pred)
    precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # ROC AUC 계산
    from sklearn.metrics import roc_auc_score, roc_curve
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        roc_auc = 0.0
    
    # 혼동 행렬
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 클래스별 성과
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    evaluation_result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'confusion_matrix': cm.tolist(),
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.flatten().tolist(),
        'threshold': threshold
    }
    
    print(f"📊 평가 결과:")
    print(f"   - 정확도: {accuracy:.4f}")
    print(f"   - 정밀도: {precision:.4f}")
    print(f"   - 재현율: {recall:.4f}")
    print(f"   - F1 점수: {f1_score:.4f}")
    print(f"   - ROC AUC: {roc_auc:.4f}")
    
    return evaluation_result

def cross_validate(model_factory_func,
                   X: np.ndarray, 
                   y: np.ndarray,
                   cv_folds: int = 5,
                   config: Dict[str, Any] = None,
                   run_id: str = None) -> Dict[str, Any]:
    """
    시계열 교차 검증
    
    Args:
        model_factory_func: 모델 생성 함수
        X: 전체 피처
        y: 전체 라벨
        cv_folds: 교차 검증 폴드 수
        config: 설정
        run_id: 실행 ID
        
    Returns:
        교차 검증 결과
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    print(f"🔄 {cv_folds}-Fold 시계열 교차 검증 시작...")
    
    # 시계열 교차 검증 사용
    from sklearn.model_selection import TimeSeriesSplit
    
    # 설정에서 교차 검증 옵션 가져오기
    cv_config = config.get('cv', {}) if config else {}
    use_timeseries = cv_config.get('timeseries', True)
    use_rolling = cv_config.get('rolling', True)
    
    if use_timeseries:
        cv = TimeSeriesSplit(n_splits=cv_folds)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_results = {
        'fold_scores': [],
        'mean_scores': {},
        'std_scores': {},
        'all_predictions': [],
        'all_true_labels': [],
        'cv_type': 'timeseries' if use_timeseries else 'kfold',
        'rolling': use_rolling
    }
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        print(f"   📊 Fold {fold + 1}/{cv_folds} 훈련 중...")
        
        # 데이터 분할
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 모델 생성
        model = model_factory_func()
        
        # 훈련
        training_result = train_model(
            model, X_train_fold, y_train_fold, X_val_fold, y_val_fold, config
        )
        
        # 평가
        eval_result = evaluate_model(model, X_val_fold, y_val_fold)
        
        # 결과 저장
        fold_scores.append({
            'fold': fold + 1,
            'accuracy': eval_result['accuracy'],
            'precision': eval_result['precision'],
            'recall': eval_result['recall'],
            'f1_score': eval_result['f1_score'],
            'roc_auc': eval_result['roc_auc']
        })
        
        cv_results['all_predictions'].extend(eval_result['y_pred'])
        cv_results['all_true_labels'].extend(eval_result['y_true'])
    
    # 평균 및 표준편차 계산
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    for metric in metrics:
        scores = [fold[metric] for fold in fold_scores]
        cv_results['mean_scores'][metric] = np.mean(scores)
        cv_results['std_scores'][metric] = np.std(scores)
    
    cv_results['fold_scores'] = fold_scores
    
    # CV 결과 저장
    if run_id:
        save_cv_results(cv_results, run_id)
    
    print(f"✅ 시계열 교차 검증 완료!")
    print(f"   - CV 타입: {cv_results['cv_type']}")
    print(f"   - 평균 정확도: {cv_results['mean_scores']['accuracy']:.4f} ± {cv_results['std_scores']['accuracy']:.4f}")
    print(f"   - 평균 F1 점수: {cv_results['mean_scores']['f1_score']:.4f} ± {cv_results['std_scores']['f1_score']:.4f}")
    print(f"   - 평균 ROC AUC: {cv_results['mean_scores']['roc_auc']:.4f} ± {cv_results['std_scores']['roc_auc']:.4f}")
    
    return cv_results

def save_cv_results(cv_results: Dict[str, Any], run_id: str) -> None:
    """CV 결과 저장"""
    import json
    import os
    
    # runs 디렉토리 생성
    runs_dir = f"runs/{run_id}"
    os.makedirs(runs_dir, exist_ok=True)
    
    # CV 결과 저장
    cv_file = os.path.join(runs_dir, "cv_metrics.json")
    with open(cv_file, 'w', encoding='utf-8') as f:
        json.dump(cv_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 CV 결과 저장: {cv_file}")

def save_model(training_result: Dict[str, Any], 
               model_path: str,
               save_history: bool = True) -> None:
    """
    모델 저장
    
    Args:
        training_result: 훈련 결과
        model_path: 저장 경로
        save_history: 훈련 히스토리 저장 여부
    """
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow가 설치되지 않았습니다.")
        return
    
    print(f"💾 모델 저장 중... ({model_path})")
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 모델 저장
    model = training_result['model']
    model.save(model_path)
    
    # 메타데이터 저장
    metadata = {
        'model_name': model.name,
        'epochs_trained': training_result['epochs_trained'],
        'best_epoch': training_result['best_epoch'],
        'best_val_loss': float(training_result['best_val_loss']),
        'best_val_accuracy': float(training_result['best_val_accuracy']),
        'training_time': training_result['training_time'],
        'config': training_result['config'],
        'class_imbalance_result': training_result['class_imbalance_result']
    }
    
    metadata_path = model_path.replace('.h5', '_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 훈련 히스토리 저장
    if save_history:
        history_path = model_path.replace('.h5', '_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(training_result['history'], f, indent=2, ensure_ascii=False)
    
    print("✅ 모델 저장 완료!")

def load_model(model_path: str) -> Dict[str, Any]:
    """
    모델 로드
    
    Args:
        model_path: 모델 경로
        
    Returns:
        로드된 모델과 메타데이터
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    print(f"📂 모델 로드 중... ({model_path})")
    
    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    
    # 메타데이터 로드
    metadata_path = model_path.replace('.h5', '_metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # 훈련 히스토리 로드
    history_path = model_path.replace('.h5', '_history.json')
    history = {}
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    
    result = {
        'model': model,
        'metadata': metadata,
        'history': history
    }
    
    print("✅ 모델 로드 완료!")
    return result

def train_all_models(processed_data: Dict[str, Any],
                     model_types: List[str] = ['gru', 'lstm', 'attention'],
                     config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    모든 모델 훈련
    
    Args:
        processed_data: 처리된 데이터
        model_types: 훈련할 모델 타입들
        config: 훈련 설정
        
    Returns:
        모든 모델 훈련 결과
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    print("🚀 모든 딥러닝 모델 훈련 시작...")
    
    all_results = {}
    
    for symbol, data in processed_data.items():
        print(f"\n📊 {symbol} 모델 훈련 중...")
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_valid = data['X_valid']
        y_valid = data['y_valid']
        X_test = data['X_test']
        y_test = data['y_test']
        input_shape = X_train.shape[1:]
        
        symbol_results = {}
        
        for model_type in model_types:
            try:
                print(f"   🔧 {model_type.upper()} 모델 훈련 중...")
                
                # 모델 생성
                model = create_model_factory(model_type, input_shape)
                
                # 훈련
                training_result = train_model(
                    model, X_train, y_train, X_valid, y_valid, config
                )
                
                # 평가
                eval_result = evaluate_model(model, X_test, y_test)
                
                symbol_results[model_type] = {
                    'training_result': training_result,
                    'evaluation_result': eval_result,
                    'model': model
                }
                
                print(f"   ✅ {model_type.upper()} 모델 훈련 완료")
                
            except Exception as e:
                print(f"   ❌ {model_type.upper()} 모델 훈련 실패: {e}")
                continue
        
        if symbol_results:
            all_results[symbol] = symbol_results
            print(f"   ✅ {symbol}: {len(symbol_results)}개 모델 훈련 완료")
        else:
            print(f"   ❌ {symbol}: 모든 모델 훈련 실패")
    
    print(f"\n✅ 총 {len(all_results)}개 종목 모델 훈련 완료!")
    return all_results

def main():
    """테스트용 메인 함수"""
    print("🚀 SectorFlow Lite - Training Module 테스트")
    print("=" * 60)
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow가 설치되지 않았습니다.")
        return None
    
    # 설정
    config = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'lookback': 30,
        'feature_cols': ['close', 'volume', 'trading_value', 'returns', 'ma_5', 'ma_20', 'volatility'],
        'scale_method': 'standard'
    }
    
    training_config = {
        'epochs': 50,
        'batch_size': 32,
        'patience': 10,
        'learning_rate': 0.001,
        'class_imbalance_method': 'class_weight'
    }
    
    symbols = ['005930', '000660']  # 테스트용 2개 종목
    
    # 데이터 준비
    processed_data = prepare_ml_data(symbols, config)
    
    # 모든 모델 훈련
    all_results = train_all_models(
        processed_data, 
        model_types=['gru', 'lstm'], 
        config=training_config
    )
    
    # 결과 요약
    print("\n📊 훈련 결과 요약:")
    for symbol, results in all_results.items():
        print(f"\n{symbol}:")
        for model_type, result in results.items():
            eval_result = result['evaluation_result']
            print(f"   {model_type.upper()}:")
            print(f"     - 정확도: {eval_result['accuracy']:.4f}")
            print(f"     - F1 점수: {eval_result['f1_score']:.4f}")
            print(f"     - ROC AUC: {eval_result['roc_auc']:.4f}")
    
    print("\n✅ Training Module 테스트 완료!")
    return all_results

if __name__ == "__main__":
    results = main()
