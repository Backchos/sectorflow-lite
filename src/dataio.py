#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Data I/O Module
데이터 파이프라인 및 전처리 모듈

Functions:
- load_data: 데이터 로드 및 기본 전처리
- create_labels: 라벨 생성 (익일 종가 상승 여부)
- create_windows: 시계열 윈도우 생성
- split_data: 데이터 분할 (train/valid/test)
- scale_data: 데이터 스케일링
- prepare_ml_data: 머신러닝용 데이터 준비
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_data(symbols: List[str], 
              start_date: str, 
              end_date: str,
              data_path: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    데이터 로드 및 기본 전처리
    
    Args:
        symbols: 종목 코드 리스트
        start_date: 시작 날짜
        end_date: 종료 날짜
        data_path: 데이터 경로
        
    Returns:
        종목별 데이터프레임 딕셔너리
    """
    print("📊 데이터 로드 시작...")
    
    data_dict = {}
    
    for symbol in symbols:
        try:
            # 실제 데이터 로드 (현재는 샘플 데이터 생성)
            df = create_sample_data(symbol, start_date, end_date)
            
            # 기본 전처리
            df = preprocess_data(df)
            
            data_dict[symbol] = df
            print(f"   ✅ {symbol}: {len(df)}일 데이터 로드 완료")
            
        except Exception as e:
            print(f"   ❌ {symbol}: 데이터 로드 실패 - {e}")
            continue
    
    print(f"✅ 총 {len(data_dict)}개 종목 데이터 로드 완료")
    return data_dict

def create_sample_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    샘플 데이터 생성 (실제 데이터가 없을 때 사용)
    
    Args:
        symbol: 종목 코드
        start_date: 시작 날짜
        end_date: 종료 날짜
        
    Returns:
        OHLCV 데이터프레임
    """
    # 날짜 범위 생성
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # 주말 제외
    
    # 가격 데이터 생성 (랜덤 워크)
    np.random.seed(hash(symbol) % 2**32)  # 종목별로 다른 시드
    
    initial_price = 50000 + (hash(symbol) % 100000)  # 종목별 초기 가격
    prices = [initial_price]
    
    for _ in range(len(dates) - 1):
        daily_return = np.random.normal(0, 0.02)  # 2% 일일 변동성
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 1000))  # 최소 가격 1000원
    
    # OHLCV 데이터 생성
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 10000000, len(dates))
    })
    
    # High, Low 조정
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    기본 데이터 전처리
    
    Args:
        df: 원본 데이터프레임
        
    Returns:
        전처리된 데이터프레임
    """
    df = df.copy()
    
    # 날짜 정렬
    df = df.sort_values('date').reset_index(drop=True)
    
    # 거래대금 계산
    df['trading_value'] = df['close'] * df['volume']
    
    # 수익률 계산
    df['returns'] = df['close'].pct_change()
    
    # 이동평균 계산
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_60'] = df['close'].rolling(window=60).mean()
    
    # 변동성 계산
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # 거래량 이동평균
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    
    return df

def create_labels(df: pd.DataFrame, 
                  target_col: str = 'close',
                  horizon: int = 1,
                  threshold: float = 0.0) -> pd.DataFrame:
    """
    라벨 생성 (익일 종가 상승 여부)
    
    Args:
        df: 데이터프레임
        target_col: 타겟 컬럼 (기본: close)
        horizon: 예측 기간 (기본: 1일)
        threshold: 상승 임계값 (기본: 0.0 = 0% 이상)
        
    Returns:
        라벨이 추가된 데이터프레임
    """
    df = df.copy()
    
    # 미래 가격 계산
    future_price = df[target_col].shift(-horizon)
    current_price = df[target_col]
    
    # 수익률 계산
    returns = (future_price - current_price) / current_price
    
    # 라벨 생성 (1: 상승, 0: 하락)
    df['label'] = (returns > threshold).astype(int)
    df['future_return'] = returns
    df['future_price'] = future_price
    
    # 마지막 horizon일은 라벨이 없음
    df.loc[df.index[-horizon:], 'label'] = np.nan
    df.loc[df.index[-horizon:], 'future_return'] = np.nan
    df.loc[df.index[-horizon:], 'future_price'] = np.nan
    
    return df

def create_windows(df: pd.DataFrame, 
                   lookback: int = 30,
                   feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    시계열 윈도우 생성
    
    Args:
        df: 데이터프레임
        lookback: 윈도우 크기 (기본: 30일)
        feature_cols: 사용할 피처 컬럼들
        
    Returns:
        (X, y) - 피처와 라벨 배열
    """
    if feature_cols is None:
        feature_cols = ['close', 'volume', 'trading_value', 'returns', 'ma_5', 'ma_20', 'volatility']
    
    # 유효한 피처 컬럼만 선택
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if not available_cols:
        raise ValueError("사용 가능한 피처 컬럼이 없습니다.")
    
    # 데이터 정리
    df_clean = df[available_cols + ['label']].dropna()
    
    if len(df_clean) < lookback + 1:
        raise ValueError(f"데이터가 부족합니다. 최소 {lookback + 1}개 행이 필요합니다.")
    
    X, y = [], []
    
    for i in range(lookback, len(df_clean)):
        # 피처 윈도우
        window = df_clean[available_cols].iloc[i-lookback:i].values
        X.append(window)
        
        # 라벨
        label = df_clean['label'].iloc[i]
        y.append(label)
    
    return np.array(X), np.array(y)

def split_data(X: np.ndarray, 
               y: np.ndarray,
               train_ratio: float = 0.7,
               valid_ratio: float = 0.15,
               test_ratio: float = 0.15,
               random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    데이터 분할 (train/valid/test)
    
    Args:
        X: 피처 배열
        y: 라벨 배열
        train_ratio: 훈련 데이터 비율
        valid_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        random_state: 랜덤 시드
        
    Returns:
        (X_train, X_valid, X_test, y_train, y_valid, y_test)
    """
    # 비율 검증
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("비율의 합이 1.0이어야 합니다.")
    
    # 시계열 데이터이므로 순차적으로 분할
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    valid_end = int(n_samples * (train_ratio + valid_ratio))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_valid = X[train_end:valid_end]
    y_valid = y[train_end:valid_end]
    
    X_test = X[valid_end:]
    y_test = y[valid_end:]
    
    print(f"📊 데이터 분할 완료:")
    print(f"   - 훈련: {len(X_train)}개 ({len(X_train)/n_samples:.1%})")
    print(f"   - 검증: {len(X_valid)}개 ({len(X_valid)/n_samples:.1%})")
    print(f"   - 테스트: {len(X_test)}개 ({len(X_test)/n_samples:.1%})")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def scale_data(X_train: np.ndarray, 
               X_valid: np.ndarray, 
               X_test: np.ndarray,
               method: str = 'standard') -> Tuple[np.ndarray, ...]:
    """
    데이터 스케일링
    
    Args:
        X_train: 훈련 피처
        X_valid: 검증 피처
        X_test: 테스트 피처
        method: 스케일링 방법 ('standard' 또는 'minmax')
        
    Returns:
        스케일링된 피처들
    """
    print(f"🔧 데이터 스케일링 시작 ({method})...")
    
    # 스케일러 선택
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method는 'standard' 또는 'minmax'여야 합니다.")
    
    # 3D 배열을 2D로 변환하여 스케일링
    original_shape = X_train.shape
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_valid_2d = X_valid.reshape(-1, X_valid.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    
    # 훈련 데이터로 스케일러 피팅
    scaler.fit(X_train_2d)
    
    # 모든 데이터 변환
    X_train_scaled = scaler.transform(X_train_2d).reshape(original_shape)
    X_valid_scaled = scaler.transform(X_valid_2d).reshape(X_valid.shape)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)
    
    print("✅ 데이터 스케일링 완료!")
    
    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler

def prepare_ml_data(symbols: List[str],
                    config: Dict[str, Any]) -> Dict[str, Any]:
    """
    머신러닝용 데이터 준비 (전체 파이프라인)
    
    Args:
        symbols: 종목 코드 리스트
        config: 설정 딕셔너리
        
    Returns:
        준비된 데이터 딕셔너리
    """
    print("🚀 머신러닝용 데이터 준비 시작...")
    
    # 설정값 추출
    start_date = config.get('start_date', '2024-01-01')
    end_date = config.get('end_date', '2024-12-31')
    lookback = config.get('lookback', 30)
    feature_cols = config.get('feature_cols', ['close', 'volume', 'trading_value', 'returns', 'ma_5', 'ma_20', 'volatility'])
    scale_method = config.get('scale_method', 'standard')
    
    # 1. 데이터 로드
    data_dict = load_data(symbols, start_date, end_date)
    
    if not data_dict:
        raise ValueError("로드된 데이터가 없습니다.")
    
    # 2. 각 종목별로 데이터 처리
    processed_data = {}
    
    for symbol, df in data_dict.items():
        print(f"\n📊 {symbol} 데이터 처리 중...")
        
        # 라벨 생성
        df_labeled = create_labels(df, threshold=0.0)
        
        # 윈도우 생성
        try:
            X, y = create_windows(df_labeled, lookback=lookback, feature_cols=feature_cols)
            
            # 데이터 분할
            X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
                X, y, 
                train_ratio=0.7, 
                valid_ratio=0.15, 
                test_ratio=0.15
            )
            
            # 데이터 스케일링
            X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_data(
                X_train, X_valid, X_test, method=scale_method
            )
            
            processed_data[symbol] = {
                'X_train': X_train_scaled,
                'X_valid': X_valid_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_valid': y_valid,
                'y_test': y_test,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'lookback': lookback,
                'original_df': df_labeled
            }
            
            print(f"   ✅ {symbol}: {len(X)}개 윈도우 생성 완료")
            
        except Exception as e:
            print(f"   ❌ {symbol}: 데이터 처리 실패 - {e}")
            continue
    
    print(f"\n✅ 총 {len(processed_data)}개 종목 데이터 준비 완료!")
    
    return processed_data

def get_data_summary(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    데이터 요약 정보 생성
    
    Args:
        processed_data: 처리된 데이터 딕셔너리
        
    Returns:
        데이터 요약 딕셔너리
    """
    summary = {
        'total_symbols': len(processed_data),
        'symbols': list(processed_data.keys()),
        'data_info': {}
    }
    
    for symbol, data in processed_data.items():
        summary['data_info'][symbol] = {
            'train_samples': len(data['X_train']),
            'valid_samples': len(data['X_valid']),
            'test_samples': len(data['X_test']),
            'feature_shape': data['X_train'].shape[1:],
            'positive_ratio': np.mean(data['y_train']),
            'feature_cols': data['feature_cols']
        }
    
    return summary

def main():
    """테스트용 메인 함수"""
    print("🚀 SectorFlow Lite - Data I/O Module 테스트")
    print("=" * 50)
    
    # 설정
    config = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'lookback': 30,
        'feature_cols': ['close', 'volume', 'trading_value', 'returns', 'ma_5', 'ma_20', 'volatility'],
        'scale_method': 'standard'
    }
    
    symbols = ['005930', '000660', '035420']  # 삼성전자, SK하이닉스, 네이버
    
    # 데이터 준비
    processed_data = prepare_ml_data(symbols, config)
    
    # 요약 정보
    summary = get_data_summary(processed_data)
    
    print("\n📋 데이터 요약:")
    for symbol, info in summary['data_info'].items():
        print(f"\n{symbol}:")
        print(f"   - 훈련: {info['train_samples']}개")
        print(f"   - 검증: {info['valid_samples']}개")
        print(f"   - 테스트: {info['test_samples']}개")
        print(f"   - 피처 형태: {info['feature_shape']}")
        print(f"   - 양성 비율: {info['positive_ratio']:.2%}")
    
    print("\n✅ Data I/O Module 테스트 완료!")
    return processed_data

if __name__ == "__main__":
    data = main()
