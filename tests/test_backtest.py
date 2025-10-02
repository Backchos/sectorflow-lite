#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Backtest Module Tests
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest import run_backtest, execute_trading_strategy, calculate_metrics

class TestBacktest:
    """백테스트 테스트"""
    
    def setup_method(self):
        """테스트 데이터 설정"""
        # 샘플 데이터 생성
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # 가격 데이터 생성
        price = 100
        prices = [price]
        for _ in range(99):
            price *= (1 + np.random.normal(0, 0.02))
            prices.append(price)
        
        self.df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(100000, 10000000, 100),
            'trading_signal': np.random.choice(['BUY', 'HOLD'], 100, p=[0.2, 0.8])
        })
        
        # High, Low 조정
        self.df['high'] = np.maximum(self.df['high'], self.df['close'])
        self.df['low'] = np.minimum(self.df['low'], self.df['close'])
    
    def test_run_backtest(self):
        """백테스트 실행 테스트"""
        result = run_backtest(self.df)
        
        # 필수 키 확인
        required_keys = ['trades', 'portfolio_values', 'metrics', 'config']
        for key in required_keys:
            assert key in result
        
        # 거래 리스트 확인
        assert isinstance(result['trades'], list)
        
        # 포트폴리오 가치 확인
        assert isinstance(result['portfolio_values'], list)
        assert len(result['portfolio_values']) == len(self.df)
        
        # 성과 지표 확인
        metrics = result['metrics']
        required_metrics = ['total_return', 'max_drawdown', 'sharpe_ratio', 'total_trades']
        for metric in required_metrics:
            assert metric in metrics
    
    def test_trading_strategy_execution(self):
        """거래 전략 실행 테스트"""
        config = {
            'initial_capital': 1000000,
            'close_col': 'close',
            'open_col': 'open',
            'signal_col': 'trading_signal'
        }
        
        result = execute_trading_strategy(self.df, config)
        
        # 필수 키 확인
        required_keys = ['trades', 'portfolio_values', 'final_capital']
        for key in required_keys:
            assert key in result
        
        # 거래 수 확인
        assert len(result['trades']) >= 0
        
        # 포트폴리오 가치 수 확인
        assert len(result['portfolio_values']) == len(self.df)
        
        # 최종 자본이 양수인지 확인
        assert result['final_capital'] > 0
    
    def test_metrics_calculation(self):
        """성과 지표 계산 테스트"""
        # 샘플 거래 데이터 생성
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
        
        # 기본 지표 확인
        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'total_trades' in metrics
        
        # 수익률이 계산되었는지 확인
        assert isinstance(metrics['total_return'], (int, float))
        
        # 거래 수 확인
        assert metrics['total_trades'] == 1  # SELL 거래만 카운트
    
    def test_reproducibility(self):
        """재현성 테스트"""
        # 같은 데이터로 여러 번 실행했을 때 결과가 일치하는지 확인
        result1 = run_backtest(self.df)
        result2 = run_backtest(self.df)
        
        # 거래 수가 일치하는지 확인
        assert len(result1['trades']) == len(result2['trades'])
        
        # 최종 자본이 일치하는지 확인
        assert result1['metrics']['final_capital'] == result2['metrics']['final_capital']
    
    def test_no_signals(self):
        """신호가 없는 경우 테스트"""
        df_no_signals = self.df.copy()
        df_no_signals['trading_signal'] = 'HOLD'
        
        result = run_backtest(df_no_signals)
        
        # 거래가 없어야 함
        assert len(result['trades']) == 0
        
        # 최종 자본이 초기 자본과 같아야 함
        assert result['metrics']['final_capital'] == result['config']['initial_capital']
    
    def test_all_buy_signals(self):
        """모든 신호가 BUY인 경우 테스트"""
        df_all_buy = self.df.copy()
        df_all_buy['trading_signal'] = 'BUY'
        
        result = run_backtest(df_all_buy)
        
        # 거래가 있어야 함
        assert len(result['trades']) > 0
        
        # 모든 거래가 BUY여야 함
        buy_trades = [t for t in result['trades'] if t['action'] == 'BUY']
        assert len(buy_trades) > 0
    
    def test_missing_columns(self):
        """필수 컬럼 누락 테스트"""
        df_no_close = self.df.drop('close', axis=1)
        
        with pytest.raises(ValueError):
            run_backtest(df_no_close)
        
        df_no_open = self.df.drop('open', axis=1)
        
        with pytest.raises(ValueError):
            run_backtest(df_no_open)
    
    def test_custom_config(self):
        """사용자 정의 설정 테스트"""
        config = {
            'initial_capital': 2000000,
            'commission_rate': 0.001,
            'close_col': 'close',
            'open_col': 'open',
            'signal_col': 'trading_signal'
        }
        
        result = run_backtest(self.df, config)
        
        # 설정이 적용되었는지 확인
        assert result['config']['initial_capital'] == 2000000
        assert result['config']['commission_rate'] == 0.001
    
    def test_portfolio_value_consistency(self):
        """포트폴리오 가치 일관성 테스트"""
        result = run_backtest(self.df)
        
        portfolio_values = result['portfolio_values']
        
        # 포트폴리오 가치가 양수인지 확인
        for pv in portfolio_values:
            assert pv['portfolio_value'] > 0
        
        # 포트폴리오 가치가 시간순으로 정렬되어 있는지 확인
        values = [pv['portfolio_value'] for pv in portfolio_values]
        assert values == sorted(values) or values == sorted(values, reverse=True)
    
    def test_trade_structure(self):
        """거래 구조 테스트"""
        result = run_backtest(self.df)
        
        for trade in result['trades']:
            # 필수 키 확인
            required_keys = ['action', 'price', 'quantity']
            for key in required_keys:
                assert key in trade
            
            # 액션이 유효한 값인지 확인
            assert trade['action'] in ['BUY', 'SELL']
            
            # 가격과 수량이 양수인지 확인
            assert trade['price'] > 0
            assert trade['quantity'] > 0

if __name__ == "__main__":
    pytest.main([__file__])






