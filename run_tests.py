#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Simple Test Runner
pytest 없이도 실행할 수 있는 간단한 테스트 러너
"""

import sys
import os
import traceback

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_test(test_func, test_name):
    """개별 테스트 실행"""
    try:
        print(f"🧪 {test_name} 실행 중...")
        test_func()
        print(f"✅ {test_name} 통과!")
        return True
    except Exception as e:
        print(f"❌ {test_name} 실패: {str(e)}")
        print(f"   상세 오류: {traceback.format_exc()}")
        return False

def test_features():
    """피처 계산 테스트"""
    import pandas as pd
    import numpy as np
    from features import calculate_trading_value_zscore, calculate_rs_indicator, process_features
    
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    price = 100
    prices = [price]
    for _ in range(99):
        price *= (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 10000000, 100)
    })
    
    # High, Low 조정
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    
    # 테스트 1: 거래대금 Z-score 계산
    result_df = calculate_trading_value_zscore(df)
    assert 'z20' in result_df.columns
    assert 'trading_value' in result_df.columns
    assert not np.isinf(result_df['z20']).any()
    
    # 테스트 2: RS 지표 계산
    result_df = calculate_rs_indicator(df)
    assert 'rs_4w' in result_df.columns
    assert not np.isinf(result_df['rs_4w']).any()
    
    # 테스트 3: 전체 피처 처리
    result_df = process_features(df)
    assert 'z20' in result_df.columns
    assert 'rs_4w' in result_df.columns
    assert len(result_df) == len(df)

def test_dataio():
    """데이터 I/O 테스트"""
    import pandas as pd
    import numpy as np
    
    # scikit-learn이 없으면 스킵
    try:
        from dataio import create_labels, create_windows, split_data, scale_data
    except ImportError as e:
        print(f"⚠️ scikit-learn이 설치되지 않아 데이터 I/O 테스트를 스킵합니다: {e}")
        return
    
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    price = 100
    prices = [price]
    for _ in range(99):
        price *= (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(100000, 10000000, 100)
    })
    
    # 테스트 1: 라벨 생성
    result_df = create_labels(df)
    assert 'label' in result_df.columns
    assert 'future_return' in result_df.columns
    valid_labels = result_df['label'].dropna()
    assert valid_labels.isin([0, 1]).all()
    
    # 테스트 2: 윈도우 생성
    X, y = create_windows(result_df, lookback=30)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 30
    assert y.isin([0, 1]).all()
    
    # 테스트 3: 데이터 분할
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)
    total_samples = len(X)
    assert len(X_train) == int(total_samples * 0.7)
    assert len(X_valid) == int(total_samples * 0.15)
    assert len(X_test) == int(total_samples * 0.15)
    
    # 테스트 4: 데이터 스케일링
    X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_data(
        X_train, X_valid, X_test, method='standard'
    )
    assert X_train_scaled.shape == X_train.shape
    assert scaler is not None

def test_rules():
    """매매 룰 테스트"""
    import pandas as pd
    import numpy as np
    from rules import generate_trading_signals, apply_trading_rules
    
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'z20': np.random.normal(0, 1, 100),
        'rs_4w': np.random.uniform(0.5, 2.0, 100),
        'close': np.random.uniform(50000, 100000, 100),
        'volume': np.random.randint(100000, 10000000, 100)
    })
    
    # 테스트 1: 매매 신호 생성
    result_df = generate_trading_signals(df)
    assert 'trading_signal' in result_df.columns
    assert 'signal_strength' in result_df.columns
    valid_signals = result_df['trading_signal'].unique()
    assert set(valid_signals).issubset({'BUY', 'HOLD'})
    
    # 테스트 2: 거래 룰 적용
    result_df = apply_trading_rules(result_df)
    valid_signals = result_df['trading_signal'].unique()
    assert set(valid_signals).issubset({'BUY', 'HOLD', 'SELL'})

def test_backtest():
    """백테스트 테스트"""
    import pandas as pd
    import numpy as np
    from backtest import run_backtest, execute_trading_strategy, calculate_metrics
    
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    price = 100
    prices = [price]
    for _ in range(99):
        price *= (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 10000000, 100),
        'trading_signal': np.random.choice(['BUY', 'HOLD'], 100, p=[0.2, 0.8])
    })
    
    # High, Low 조정
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    
    # 테스트 1: 백테스트 실행
    result = run_backtest(df)
    required_keys = ['trades', 'portfolio_values', 'metrics', 'config']
    for key in required_keys:
        assert key in result
    
    # 테스트 2: 거래 전략 실행
    config = {
        'initial_capital': 1000000,
        'close_col': 'close',
        'open_col': 'open',
        'signal_col': 'trading_signal'
    }
    
    result = execute_trading_strategy(df, config)
    required_keys = ['trades', 'portfolio_values', 'final_capital']
    for key in required_keys:
        assert key in result
    
    # 테스트 3: 성과 지표 계산
    trades = [
        {'action': 'BUY', 'price': 100, 'quantity': 10, 'commission': 30},
        {'action': 'SELL', 'price': 110, 'quantity': 10, 'commission': 33, 'net_profit': 67}
    ]
    
    portfolio_values = [
        {'portfolio_value': 1000000},
        {'portfolio_value': 1000067}
    ]
    
    config = {'initial_capital': 1000000}
    
    metrics = calculate_metrics({
        'trades': trades,
        'portfolio_values': portfolio_values,
        'final_capital': 1000067
    }, config)
    
    required_metrics = ['total_return', 'max_drawdown', 'sharpe_ratio', 'total_trades']
    for metric in required_metrics:
        assert metric in metrics

def main():
    """메인 테스트 실행"""
    print("🚀 SectorFlow Lite - 테스트 실행")
    print("=" * 50)
    
    tests = [
        (test_features, "피처 계산 테스트"),
        (test_dataio, "데이터 I/O 테스트"),
        (test_rules, "매매 룰 테스트"),
        (test_backtest, "백테스트 테스트")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트가 성공적으로 통과했습니다!")
        return True
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
