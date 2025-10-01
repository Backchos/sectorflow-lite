#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Features Module
거래대금 Z-score와 RS 지표 계산 모듈

Functions:
- calculate_trading_value_zscore: 거래대금 Z-score 계산 (z20)
- calculate_rs_indicator: RS 지표 계산 (4주 기준)
- process_features: 전체 피처 처리 파이프라인
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

def calculate_trading_value_zscore(df: pd.DataFrame, 
                                 window: int = 20,
                                 trading_value_col: str = 'trading_value') -> pd.DataFrame:
    """
    거래대금 Z-score 계산 (z20)
    
    Args:
        df: OHLCV 데이터가 포함된 DataFrame
        window: Z-score 계산 윈도우 (기본 20일)
        trading_value_col: 거래대금 컬럼명
        
    Returns:
        z20 컬럼이 추가된 DataFrame
    """
    df = df.copy()
    
    # 거래대금 계산 (종가 * 거래량)
    if 'trading_value' not in df.columns:
        if 'close' in df.columns and 'volume' in df.columns:
            df['trading_value'] = df['close'] * df['volume']
        else:
            raise ValueError("거래대금 계산을 위해 'close'와 'volume' 컬럼이 필요합니다.")
    
    # Z-score 계산 (20일 이동평균과 표준편차 기준)
    df['trading_value_mean'] = df[trading_value_col].rolling(window=window).mean()
    df['trading_value_std'] = df[trading_value_col].rolling(window=window).std()
    df['z20'] = (df[trading_value_col] - df['trading_value_mean']) / df['trading_value_std']
    
    # 무한대 값 처리
    df['z20'] = df['z20'].replace([np.inf, -np.inf], np.nan)
    
    return df

def calculate_rs_indicator(df: pd.DataFrame, 
                          period: int = 20,
                          close_col: str = 'close') -> pd.DataFrame:
    """
    RS (Relative Strength) 지표 계산 (4주 기준)
    
    Args:
        df: OHLCV 데이터가 포함된 DataFrame
        period: RS 계산 기간 (기본 20일 = 4주)
        close_col: 종가 컬럼명
        
    Returns:
        rs_4w 컬럼이 추가된 DataFrame
    """
    df = df.copy()
    
    # 수익률 계산
    df['returns'] = df[close_col].pct_change()
    
    # 양수 수익률과 음수 수익률 분리
    positive_returns = df['returns'].where(df['returns'] > 0, 0)
    negative_returns = df['returns'].where(df['returns'] < 0, 0).abs()
    
    # 이동평균 계산
    df['avg_gain'] = positive_returns.rolling(window=period).mean()
    df['avg_loss'] = negative_returns.rolling(window=period).mean()
    
    # RS 계산 (평균 이익 / 평균 손실)
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['rs'] = df['rs'].replace([np.inf, -np.inf], np.nan)
    
    # RS를 rs_4w로 컬럼명 변경
    df['rs_4w'] = df['rs']
    
    # 불필요한 중간 컬럼 제거
    df = df.drop(['returns', 'avg_gain', 'avg_loss', 'rs'], axis=1, errors='ignore')
    
    return df

def process_features(df: pd.DataFrame, 
                    config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    전체 피처 처리 파이프라인
    
    Args:
        df: OHLCV 데이터가 포함된 DataFrame
        config: 설정 딕셔너리
        
    Returns:
        피처가 추가된 DataFrame
    """
    if config is None:
        config = {
            'zscore_window': 20,
            'rs_period': 20,
            'trading_value_col': 'trading_value',
            'close_col': 'close'
        }
    
    df = df.copy()
    
    print("🔧 피처 계산 시작...")
    
    # 1. 거래대금 Z-score 계산
    df = calculate_trading_value_zscore(
        df, 
        window=config['zscore_window'],
        trading_value_col=config['trading_value_col']
    )
    
    # 2. RS 지표 계산
    df = calculate_rs_indicator(
        df,
        period=config['rs_period'],
        close_col=config['close_col']
    )
    
    # 3. 추가 기술적 지표 (선택사항)
    df = add_technical_indicators(df)
    
    print("✅ 피처 계산 완료!")
    print(f"   - z20 (거래대금 Z-score): {df['z20'].notna().sum()}개 값")
    print(f"   - rs_4w (RS 지표): {df['rs_4w'].notna().sum()}개 값")
    
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    추가 기술적 지표 계산
    
    Args:
        df: OHLCV 데이터가 포함된 DataFrame
        
    Returns:
        기술적 지표가 추가된 DataFrame
    """
    df = df.copy()
    
    # 이동평균
    if 'close' in df.columns:
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_60'] = df['close'].rolling(window=60).mean()
    
    # 변동성 (ATR)
    if all(col in df.columns for col in ['high', 'low', 'close']):
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
    
    # 거래량 이동평균
    if 'volume' in df.columns:
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    
    return df

def validate_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    피처 데이터 검증
    
    Args:
        df: 피처가 포함된 DataFrame
        
    Returns:
        검증 결과 딕셔너리
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # 필수 컬럼 확인
    required_cols = ['z20', 'rs_4w']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"필수 컬럼 누락: {missing_cols}")
    
    # NaN 값 확인
    for col in required_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            total_count = len(df)
            nan_ratio = nan_count / total_count
            
            validation_results['stats'][f'{col}_nan_ratio'] = nan_ratio
            
            if nan_ratio > 0.5:
                validation_results['warnings'].append(f"{col}: NaN 비율이 높음 ({nan_ratio:.2%})")
    
    return validation_results

def main():
    """테스트용 메인 함수"""
    print("🚀 SectorFlow Lite - Features Module 테스트")
    print("=" * 50)
    
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # 가격 데이터 생성 (랜덤 워크)
    price = 100
    prices = [price]
    for _ in range(99):
        price *= (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    # 샘플 데이터프레임 생성
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # High, Low 조정
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    
    print("📊 샘플 데이터 생성 완료")
    print(f"   - 기간: {df['date'].min()} ~ {df['date'].max()}")
    print(f"   - 데이터 수: {len(df)}개")
    
    # 피처 계산
    df_with_features = process_features(df)
    
    # 검증
    validation = validate_features(df_with_features)
    
    print("\n📋 피처 계산 결과:")
    print(f"   - z20 최대값: {df_with_features['z20'].max():.2f}")
    print(f"   - z20 최소값: {df_with_features['z20'].min():.2f}")
    print(f"   - rs_4w 최대값: {df_with_features['rs_4w'].max():.2f}")
    print(f"   - rs_4w 최소값: {df_with_features['rs_4w'].min():.2f}")
    
    print("\n🔍 검증 결과:")
    print(f"   - 유효성: {'✅ 통과' if validation['is_valid'] else '❌ 실패'}")
    if validation['warnings']:
        print(f"   - 경고: {len(validation['warnings'])}개")
        for warning in validation['warnings']:
            print(f"     • {warning}")
    
    print("\n📈 최근 5일 데이터:")
    recent_cols = ['date', 'close', 'volume', 'z20', 'rs_4w']
    print(df_with_features[recent_cols].tail())
    
    print("\n✅ Features Module 테스트 완료!")
    return df_with_features

if __name__ == "__main__":
    df_result = main()

