#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Inference Module
모델 예측 및 매매 신호 생성

Functions:
- load_trained_model: 훈련된 모델 로드
- predict_probability: 확률 예측
- generate_trading_signals: 매매 신호 생성
- optimize_threshold: 임계값 최적화
- create_signal_report: 신호 리포트 생성
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow가 설치되지 않았습니다.")

# 기존 모듈들 import
from train import load_model, evaluate_model
from dataio import prepare_ml_data

def load_trained_model(model_path: str) -> Dict[str, Any]:
    """
    훈련된 모델 로드
    
    Args:
        model_path: 모델 파일 경로
        
    Returns:
        로드된 모델과 메타데이터
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    print(f"📂 훈련된 모델 로드 중... ({model_path})")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    
    # 메타데이터 로드
    metadata_path = model_path.replace('.h5', '_metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    print("✅ 모델 로드 완료!")
    return {
        'model': model,
        'metadata': metadata
    }

def predict_probability(model, 
                       X: np.ndarray,
                       batch_size: int = 32) -> np.ndarray:
    """
    확률 예측
    
    Args:
        model: 훈련된 모델
        X: 입력 데이터
        batch_size: 배치 크기
        
    Returns:
        예측 확률 배열
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    print("🔮 확률 예측 중...")
    
    # 예측 실행
    probabilities = model.predict(X, batch_size=batch_size, verbose=0)
    
    # 1차원으로 변환
    if probabilities.ndim > 1:
        probabilities = probabilities.flatten()
    
    print(f"✅ {len(probabilities)}개 확률 예측 완료!")
    return probabilities

def generate_trading_signals(probabilities: np.ndarray,
                           threshold: float = 0.5,
                           min_confidence: float = 0.6,
                           signal_type: str = 'binary') -> Dict[str, Any]:
    """
    매매 신호 생성
    
    Args:
        probabilities: 예측 확률
        threshold: 분류 임계값
        min_confidence: 최소 신뢰도
        signal_type: 신호 타입 ('binary', 'confidence', 'gradient')
        
    Returns:
        신호 딕셔너리
    """
    print(f"📊 매매 신호 생성 중... (임계값: {threshold}, 신뢰도: {min_confidence})")
    
    signals = {}
    
    if signal_type == 'binary':
        # 이진 신호 (0 또는 1)
        binary_signals = (probabilities > threshold).astype(int)
        signals['binary'] = binary_signals
        signals['confidence'] = np.abs(probabilities - 0.5) * 2  # 0-1 범위로 정규화
    
    elif signal_type == 'confidence':
        # 신뢰도 기반 신호
        confidence = np.abs(probabilities - 0.5) * 2
        binary_signals = (probabilities > threshold).astype(int)
        
        # 신뢰도가 낮은 신호는 제거
        low_confidence_mask = confidence < min_confidence
        binary_signals[low_confidence_mask] = 0
        
        signals['binary'] = binary_signals
        signals['confidence'] = confidence
        signals['filtered_count'] = np.sum(low_confidence_mask)
    
    elif signal_type == 'gradient':
        # 그라디언트 기반 신호 (확률 변화율)
        if len(probabilities) > 1:
            gradient = np.gradient(probabilities)
            signals['gradient'] = gradient
            signals['binary'] = (gradient > 0).astype(int)  # 상승 추세
        else:
            signals['gradient'] = np.array([0])
            signals['binary'] = np.array([0])
    
    # 신호 통계
    signal_counts = np.bincount(signals['binary'])
    signals['stats'] = {
        'total_signals': len(probabilities),
        'buy_signals': signal_counts[1] if len(signal_counts) > 1 else 0,
        'hold_signals': signal_counts[0] if len(signal_counts) > 0 else 0,
        'buy_ratio': signal_counts[1] / len(probabilities) if len(signal_counts) > 1 else 0,
        'avg_confidence': np.mean(signals.get('confidence', [0])),
        'max_confidence': np.max(signals.get('confidence', [0])),
        'min_confidence': np.min(signals.get('confidence', [0]))
    }
    
    print(f"✅ 신호 생성 완료!")
    print(f"   - 총 신호: {signals['stats']['total_signals']}개")
    print(f"   - 매수 신호: {signals['stats']['buy_signals']}개 ({signals['stats']['buy_ratio']:.1%})")
    print(f"   - 평균 신뢰도: {signals['stats']['avg_confidence']:.3f}")
    
    return signals

def optimize_threshold(y_true: np.ndarray,
                      y_prob: np.ndarray,
                      metric: str = 'f1',
                      thresholds: np.ndarray = None) -> Dict[str, Any]:
    """
    임계값 최적화
    
    Args:
        y_true: 실제 라벨
        y_prob: 예측 확률
        metric: 최적화할 지표 ('f1', 'precision', 'recall', 'accuracy', 'roc_auc')
        thresholds: 테스트할 임계값들
        
    Returns:
        최적화 결과
    """
    print(f"🎯 임계값 최적화 중... (지표: {metric})")
    
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    best_threshold = 0.5
    best_score = 0
    threshold_scores = []
    
    for threshold in thresholds:
        # 임계값으로 예측
        y_pred = (y_prob > threshold).astype(int)
        
        # 지표 계산
        if metric == 'f1':
            from sklearn.metrics import f1_score
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            from sklearn.metrics import precision_score
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            from sklearn.metrics import recall_score
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = np.mean(y_true == y_pred)
        elif metric == 'roc_auc':
            from sklearn.metrics import roc_auc_score
            try:
                score = roc_auc_score(y_true, y_prob)
            except ValueError:
                score = 0
        else:
            raise ValueError(f"지원하지 않는 지표: {metric}")
        
        threshold_scores.append({
            'threshold': threshold,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # 최적 임계값으로 최종 평가
    y_pred_best = (y_prob > best_threshold).astype(int)
    
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_true, y_pred_best, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_best)
    
    optimization_result = {
        'best_threshold': best_threshold,
        'best_score': best_score,
        'threshold_scores': threshold_scores,
        'final_metrics': {
            'accuracy': np.mean(y_true == y_pred_best),
            'precision': report['1']['precision'] if '1' in report else 0,
            'recall': report['1']['recall'] if '1' in report else 0,
            'f1_score': report['1']['f1-score'] if '1' in report else 0,
            'confusion_matrix': cm.tolist()
        }
    }
    
    print(f"✅ 임계값 최적화 완료!")
    print(f"   - 최적 임계값: {best_threshold:.3f}")
    print(f"   - 최고 {metric} 점수: {best_score:.3f}")
    
    return optimization_result

def create_signal_report(signals: Dict[str, Any],
                        dates: List[str] = None,
                        prices: List[float] = None) -> str:
    """
    신호 리포트 생성
    
    Args:
        signals: 신호 딕셔너리
        dates: 날짜 리스트
        prices: 가격 리스트
        
    Returns:
        리포트 문자열
    """
    stats = signals['stats']
    
    report = f"""
📊 SectorFlow Lite - 매매 신호 리포트
{'='*50}

📈 신호 통계
- 총 신호 수: {stats['total_signals']}개
- 매수 신호: {stats['buy_signals']}개 ({stats['buy_ratio']:.1%})
- 보유 신호: {stats['hold_signals']}개
- 평균 신뢰도: {stats['avg_confidence']:.3f}
- 최대 신뢰도: {stats['max_confidence']:.3f}
- 최소 신뢰도: {stats['min_confidence']:.3f}

📊 신호 분포
"""
    
    # 신호 분포 시각화 (텍스트)
    if 'binary' in signals:
        binary_signals = signals['binary']
        signal_changes = np.diff(binary_signals)
        buy_entries = np.where(signal_changes == 1)[0]
        sell_entries = np.where(signal_changes == -1)[0]
        
        report += f"- 매수 진입: {len(buy_entries)}회\n"
        report += f"- 매도 진입: {len(sell_entries)}회\n"
    
    # 신뢰도 분포
    if 'confidence' in signals:
        confidence = signals['confidence']
        high_conf = np.sum(confidence > 0.8)
        med_conf = np.sum((confidence > 0.5) & (confidence <= 0.8))
        low_conf = np.sum(confidence <= 0.5)
        
        report += f"\n🎯 신뢰도 분포\n"
        report += f"- 높음 (>0.8): {high_conf}개 ({high_conf/len(confidence):.1%})\n"
        report += f"- 중간 (0.5-0.8): {med_conf}개 ({med_conf/len(confidence):.1%})\n"
        report += f"- 낮음 (≤0.5): {low_conf}개 ({low_conf/len(confidence):.1%})\n"
    
    # 최근 신호 (날짜와 가격이 제공된 경우)
    if dates is not None and prices is not None and 'binary' in signals:
        report += f"\n📅 최근 신호 (최근 10개)\n"
        recent_signals = list(zip(dates[-10:], prices[-10:], signals['binary'][-10:]))
        
        for date, price, signal in recent_signals:
            signal_text = "🟢 매수" if signal == 1 else "🔴 보유"
            report += f"- {date}: {price:,.0f}원 {signal_text}\n"
    
    report += f"\n{'='*50}\n"
    
    return report

def predict_and_generate_signals(model_path: str,
                                X: np.ndarray,
                                threshold: float = 0.5,
                                min_confidence: float = 0.6,
                                optimize_thresh: bool = False,
                                y_true: np.ndarray = None,
                                run_id: str = None,
                                ticker: str = None) -> Dict[str, Any]:
    """
    예측 및 신호 생성 (통합 함수)
    
    Args:
        model_path: 모델 경로
        X: 입력 데이터
        threshold: 분류 임계값
        min_confidence: 최소 신뢰도
        optimize_thresh: 임계값 최적화 여부
        y_true: 실제 라벨 (최적화용)
        
    Returns:
        예측 및 신호 결과
    """
    print("🚀 예측 및 신호 생성 시작...")
    
    # 모델 로드
    model_data = load_trained_model(model_path)
    model = model_data['model']
    metadata = model_data['metadata']
    
    # 확률 예측
    probabilities = predict_probability(model, X)
    
    # 임계값 최적화 (실제 라벨이 있는 경우)
    if optimize_thresh and y_true is not None:
        print("🎯 임계값 최적화 실행...")
        optimization_result = optimize_threshold(y_true, probabilities)
        threshold = optimization_result['best_threshold']
        print(f"   - 최적 임계값: {threshold:.3f}")
    
    # 신호 생성
    signals = generate_trading_signals(
        probabilities, 
        threshold=threshold, 
        min_confidence=min_confidence
    )
    
    # 결과 통합
    result = {
        'probabilities': probabilities,
        'signals': signals,
        'threshold': threshold,
        'model_metadata': metadata,
        'optimization_result': optimization_result if optimize_thresh and y_true is not None else None
    }
    
    # 신호 CSV 저장
    if run_id and ticker:
        save_signals_csv(probabilities, signals, threshold, run_id, ticker, metadata)
    
    print("✅ 예측 및 신호 생성 완료!")
    return result

def save_signals_csv(probabilities: np.ndarray,
                    signals: Dict[str, Any],
                    threshold: float,
                    run_id: str,
                    ticker: str,
                    model_metadata: Dict[str, Any]) -> str:
    """모델 신호를 CSV로 저장"""
    import pandas as pd
    import os
    from datetime import datetime
    
    # runs 디렉토리 생성
    runs_dir = f"runs/{run_id}"
    os.makedirs(runs_dir, exist_ok=True)
    
    # 신호 데이터 생성
    dates = pd.date_range(start='2024-01-01', periods=len(probabilities), freq='D')
    
    signals_data = {
        'date': dates,
        'ticker': ticker,
        'prob': probabilities,
        'signal': signals['binary'],
        'model_name': model_metadata.get('model_name', 'unknown'),
        'threshold': threshold
    }
    
    # DataFrame 생성
    df_signals = pd.DataFrame(signals_data)
    
    # CSV 저장
    csv_path = os.path.join(runs_dir, "signals_model.csv")
    df_signals.to_csv(csv_path, index=False)
    
    print(f"💾 모델 신호 저장: {csv_path}")
    return csv_path

def batch_predict(symbols: List[str],
                  model_paths: Dict[str, str],
                  processed_data: Dict[str, Any],
                  threshold: float = 0.5,
                  min_confidence: float = 0.6) -> Dict[str, Any]:
    """
    배치 예측 (여러 종목)
    
    Args:
        symbols: 종목 리스트
        model_paths: 종목별 모델 경로
        processed_data: 처리된 데이터
        threshold: 분류 임계값
        min_confidence: 최소 신뢰도
        
    Returns:
        모든 종목의 예측 결과
    """
    print("🚀 배치 예측 시작...")
    
    all_predictions = {}
    
    for symbol in symbols:
        if symbol not in model_paths:
            print(f"   ⚠️ {symbol}: 모델 경로가 없습니다.")
            continue
        
        if symbol not in processed_data:
            print(f"   ⚠️ {symbol}: 데이터가 없습니다.")
            continue
        
        try:
            print(f"   📊 {symbol} 예측 중...")
            
            # 테스트 데이터 가져오기
            X_test = processed_data[symbol]['X_test']
            y_test = processed_data[symbol]['y_test']
            
            # 예측 및 신호 생성
            result = predict_and_generate_signals(
                model_paths[symbol],
                X_test,
                threshold=threshold,
                min_confidence=min_confidence,
                optimize_thresh=True,
                y_true=y_test
            )
            
            all_predictions[symbol] = result
            print(f"   ✅ {symbol} 예측 완료")
            
        except Exception as e:
            print(f"   ❌ {symbol} 예측 실패: {e}")
            continue
    
    print(f"✅ 배치 예측 완료! ({len(all_predictions)}개 종목)")
    return all_predictions

def main():
    """테스트용 메인 함수"""
    print("🚀 SectorFlow Lite - Inference Module 테스트")
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
    
    symbols = ['005930', '000660']  # 테스트용 2개 종목
    
    # 데이터 준비
    processed_data = prepare_ml_data(symbols, config)
    
    # 간단한 모델 생성 및 훈련 (테스트용)
    from train import train_all_models
    training_config = {
        'epochs': 10,  # 빠른 테스트를 위해 적은 에포크
        'batch_size': 32,
        'patience': 5
    }
    
    print("🔧 테스트용 모델 훈련 중...")
    training_results = train_all_models(
        processed_data, 
        model_types=['gru'], 
        config=training_config
    )
    
    # 모델 경로 생성 (실제로는 저장된 모델을 사용)
    model_paths = {}
    for symbol in symbols:
        if symbol in training_results:
            # 실제 환경에서는 저장된 모델 경로를 사용
            model_paths[symbol] = f"models/{symbol}_gru_model.h5"
    
    # 배치 예측
    if model_paths:
        predictions = batch_predict(
            symbols, 
            model_paths, 
            processed_data,
            threshold=0.5,
            min_confidence=0.6
        )
        
        # 결과 요약
        print("\n📊 예측 결과 요약:")
        for symbol, result in predictions.items():
            signals = result['signals']
            stats = signals['stats']
            print(f"\n{symbol}:")
            print(f"   - 매수 신호: {stats['buy_signals']}개 ({stats['buy_ratio']:.1%})")
            print(f"   - 평균 신뢰도: {stats['avg_confidence']:.3f}")
            print(f"   - 사용된 임계값: {result['threshold']:.3f}")
    
    print("\n✅ Inference Module 테스트 완료!")
    return predictions if 'predictions' in locals() else None

if __name__ == "__main__":
    results = main()
