#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Rules Module
매매 룰 신호 생성 모듈

Functions:
- generate_trading_signals: 매매 신호 생성
- apply_trading_rules: 거래 룰 적용
- validate_signals: 신호 검증
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def generate_trading_signals(df: pd.DataFrame, 
                           config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    매매 신호 생성
    
    BUY 조건: z20 >= 1.0 and rs_4w > 1.0
    그 외는 HOLD
    
    Args:
        df: 피처가 포함된 DataFrame (z20, rs_4w 컬럼 필요)
        config: 설정 딕셔너리
        
    Returns:
        trading_signal 컬럼이 추가된 DataFrame
    """
    if config is None:
        config = {
            'z20_threshold': 1.0,
            'rs_threshold': 1.0,
            'z20_col': 'z20',
            'rs_col': 'rs_4w'
        }
    
    df = df.copy()
    
    # 필수 컬럼 확인
    required_cols = [config['z20_col'], config['rs_col']]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")
    
    print("📊 매매 신호 생성 중...")
    
    # 매매 조건 적용
    z20_condition = df[config['z20_col']] >= config['z20_threshold']
    rs_condition = df[config['rs_col']] > config['rs_threshold']
    
    # BUY 조건: z20 >= 1.0 AND rs_4w > 1.0
    buy_condition = z20_condition & rs_condition
    
    # 신호 생성
    df['trading_signal'] = 'HOLD'
    df.loc[buy_condition, 'trading_signal'] = 'BUY'
    
    # 신호 강도 계산 (선택사항)
    df['signal_strength'] = calculate_signal_strength(df, config)
    
    # 신호 지속성 계산
    df['signal_duration'] = calculate_signal_duration(df)
    
    print(f"✅ 신호 생성 완료!")
    print(f"   - BUY 신호: {df['trading_signal'].value_counts().get('BUY', 0)}개")
    print(f"   - HOLD 신호: {df['trading_signal'].value_counts().get('HOLD', 0)}개")
    
    return df

def calculate_signal_strength(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """
    신호 강도 계산
    
    Args:
        df: 피처가 포함된 DataFrame
        config: 설정 딕셔너리
        
    Returns:
        신호 강도 Series
    """
    z20_col = config['z20_col']
    rs_col = config['rs_col']
    
    # 정규화된 점수 계산 (0-100)
    z20_score = np.clip(df[z20_col] * 20 + 50, 0, 100)  # z20을 0-100으로 변환
    rs_score = np.clip(df[rs_col] * 25, 0, 100)  # rs를 0-100으로 변환
    
    # 가중 평균 (z20: 60%, rs: 40%)
    signal_strength = (z20_score * 0.6 + rs_score * 0.4)
    
    return signal_strength

def calculate_signal_duration(df: pd.DataFrame) -> pd.Series:
    """
    신호 지속 기간 계산
    
    Args:
        df: 신호가 포함된 DataFrame
        
    Returns:
        신호 지속 기간 Series
    """
    signal_duration = pd.Series(0, index=df.index)
    
    current_signal = None
    duration = 0
    
    for i, signal in enumerate(df['trading_signal']):
        if signal == current_signal:
            duration += 1
        else:
            duration = 1
            current_signal = signal
        
        signal_duration.iloc[i] = duration
    
    return signal_duration

def apply_trading_rules(df: pd.DataFrame, 
                       rules_config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    거래 룰 적용
    
    Args:
        df: 피처가 포함된 DataFrame
        rules_config: 룰 설정 딕셔너리
        
    Returns:
        룰이 적용된 DataFrame
    """
    if rules_config is None:
        rules_config = {
            'min_volume_ratio': 1.0,  # 최소 거래량 비율
            'max_position_days': 5,   # 최대 보유 일수
            'stop_loss_ratio': 0.05,  # 손절 비율
            'take_profit_ratio': 0.10  # 익절 비율
        }
    
    df = df.copy()
    
    print("🔧 거래 룰 적용 중...")
    
    # 1. 거래량 필터링
    if 'volume_ma_20' in df.columns:
        volume_condition = df['volume'] >= df['volume_ma_20'] * rules_config['min_volume_ratio']
        df.loc[~volume_condition, 'trading_signal'] = 'HOLD'
    
    # 2. 최대 보유 일수 제한
    df = apply_position_duration_limit(df, rules_config['max_position_days'])
    
    # 3. 손절/익절 룰 적용
    df = apply_stop_loss_take_profit(df, rules_config)
    
    # 4. 신호 정리
    df = clean_trading_signals(df)
    
    print("✅ 거래 룰 적용 완료!")
    
    return df

def apply_position_duration_limit(df: pd.DataFrame, max_days: int) -> pd.DataFrame:
    """
    최대 보유 일수 제한 적용
    
    Args:
        df: 신호가 포함된 DataFrame
        max_days: 최대 보유 일수
        
    Returns:
        보유 일수 제한이 적용된 DataFrame
    """
    df = df.copy()
    
    # BUY 신호가 max_days 이상 지속되면 HOLD로 변경
    buy_mask = df['trading_signal'] == 'BUY'
    long_duration = df['signal_duration'] > max_days
    
    df.loc[buy_mask & long_duration, 'trading_signal'] = 'HOLD'
    
    return df

def apply_stop_loss_take_profit(df: pd.DataFrame, rules_config: Dict[str, Any]) -> pd.DataFrame:
    """
    손절/익절 룰 적용
    
    Args:
        df: 신호가 포함된 DataFrame
        rules_config: 룰 설정 딕셔너리
        
    Returns:
        손절/익절이 적용된 DataFrame
    """
    df = df.copy()
    
    if 'close' not in df.columns:
        return df
    
    stop_loss_ratio = rules_config.get('stop_loss_ratio', 0.05)
    take_profit_ratio = rules_config.get('take_profit_ratio', 0.10)
    
    # BUY 신호의 진입 가격 추적
    df['entry_price'] = np.nan
    df['position_pnl'] = 0.0
    
    entry_price = None
    for i, row in df.iterrows():
        if row['trading_signal'] == 'BUY' and pd.isna(entry_price):
            entry_price = row['close']
            df.loc[i, 'entry_price'] = entry_price
        elif row['trading_signal'] == 'HOLD' and not pd.isna(entry_price):
            # 손절/익절 체크
            current_price = row['close']
            pnl_ratio = (current_price - entry_price) / entry_price
            
            df.loc[i, 'position_pnl'] = pnl_ratio
            
            if pnl_ratio <= -stop_loss_ratio:
                df.loc[i, 'trading_signal'] = 'SELL'  # 손절
                entry_price = None
            elif pnl_ratio >= take_profit_ratio:
                df.loc[i, 'trading_signal'] = 'SELL'  # 익절
                entry_price = None
        elif row['trading_signal'] == 'HOLD' and pd.isna(entry_price):
            # 포지션이 없는 상태
            pass
    
    return df

def clean_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    신호 정리 (연속된 동일 신호 제거)
    
    Args:
        df: 신호가 포함된 DataFrame
        
    Returns:
        정리된 신호 DataFrame
    """
    df = df.copy()
    
    # 연속된 동일 신호를 하나로 합치기
    df['signal_changed'] = df['trading_signal'] != df['trading_signal'].shift(1)
    df.loc[0, 'signal_changed'] = True  # 첫 번째 행은 항상 변경으로 처리
    
    return df

def validate_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    신호 검증
    
    Args:
        df: 신호가 포함된 DataFrame
        
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
    if 'trading_signal' not in df.columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append("trading_signal 컬럼이 없습니다.")
        return validation_results
    
    # 신호 분포 확인
    signal_counts = df['trading_signal'].value_counts()
    validation_results['stats']['signal_distribution'] = signal_counts.to_dict()
    
    # BUY 신호 비율
    total_signals = len(df)
    buy_signals = signal_counts.get('BUY', 0)
    buy_ratio = buy_signals / total_signals if total_signals > 0 else 0
    
    validation_results['stats']['buy_ratio'] = buy_ratio
    
    if buy_ratio > 0.5:
        validation_results['warnings'].append(f"BUY 신호 비율이 높음: {buy_ratio:.2%}")
    
    # 연속 신호 확인
    if 'signal_duration' in df.columns:
        max_duration = df['signal_duration'].max()
        validation_results['stats']['max_signal_duration'] = max_duration
        
        if max_duration > 20:
            validation_results['warnings'].append(f"신호가 너무 오래 지속됨: {max_duration}일")
    
    return validation_results

def get_signal_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    신호 요약 정보 생성
    
    Args:
        df: 신호가 포함된 DataFrame
        
    Returns:
        신호 요약 딕셔너리
    """
    summary = {
        'total_periods': len(df),
        'signal_counts': df['trading_signal'].value_counts().to_dict(),
        'buy_ratio': 0,
        'avg_signal_strength': 0,
        'max_signal_duration': 0
    }
    
    # BUY 신호 비율
    buy_count = summary['signal_counts'].get('BUY', 0)
    summary['buy_ratio'] = buy_count / summary['total_periods']
    
    # 평균 신호 강도
    if 'signal_strength' in df.columns:
        summary['avg_signal_strength'] = df['signal_strength'].mean()
    
    # 최대 신호 지속 기간
    if 'signal_duration' in df.columns:
        summary['max_signal_duration'] = df['signal_duration'].max()
    
    return summary

def main():
    """테스트용 메인 함수"""
    print("🚀 SectorFlow Lite - Rules Module 테스트")
    print("=" * 50)
    
    # 샘플 데이터 생성 (features.py와 동일한 구조)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # 가격 데이터 생성
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
    
    # 피처 계산 (간단한 버전)
    df['trading_value'] = df['close'] * df['volume']
    df['trading_value_mean'] = df['trading_value'].rolling(window=20).mean()
    df['trading_value_std'] = df['trading_value'].rolling(window=20).std()
    df['z20'] = (df['trading_value'] - df['trading_value_mean']) / df['trading_value_std']
    df['z20'] = df['z20'].replace([np.inf, -np.inf], np.nan)
    
    # RS 계산
    df['returns'] = df['close'].pct_change()
    positive_returns = df['returns'].where(df['returns'] > 0, 0)
    negative_returns = df['returns'].where(df['returns'] < 0, 0).abs()
    df['avg_gain'] = positive_returns.rolling(window=20).mean()
    df['avg_loss'] = negative_returns.rolling(window=20).mean()
    df['rs_4w'] = df['avg_gain'] / df['avg_loss']
    df['rs_4w'] = df['rs_4w'].replace([np.inf, -np.inf], np.nan)
    
    print("📊 샘플 데이터 생성 완료")
    print(f"   - 기간: {df['date'].min()} ~ {df['date'].max()}")
    print(f"   - 데이터 수: {len(df)}개")
    
    # 매매 신호 생성
    df_with_signals = generate_trading_signals(df)
    
    # 거래 룰 적용
    df_with_rules = apply_trading_rules(df_with_signals)
    
    # 검증
    validation = validate_signals(df_with_rules)
    summary = get_signal_summary(df_with_rules)
    
    print("\n📋 신호 생성 결과:")
    print(f"   - BUY 신호: {summary['signal_counts'].get('BUY', 0)}개")
    print(f"   - HOLD 신호: {summary['signal_counts'].get('HOLD', 0)}개")
    print(f"   - BUY 비율: {summary['buy_ratio']:.2%}")
    print(f"   - 평균 신호 강도: {summary['avg_signal_strength']:.1f}")
    
    print("\n🔍 검증 결과:")
    print(f"   - 유효성: {'✅ 통과' if validation['is_valid'] else '❌ 실패'}")
    if validation['warnings']:
        print(f"   - 경고: {len(validation['warnings'])}개")
        for warning in validation['warnings']:
            print(f"     • {warning}")
    
    print("\n📈 최근 10일 신호:")
    recent_cols = ['date', 'close', 'z20', 'rs_4w', 'trading_signal', 'signal_strength']
    print(df_with_rules[recent_cols].tail(10))
    
    print("\n✅ Rules Module 테스트 완료!")
    return df_with_rules

if __name__ == "__main__":
    df_result = main()

