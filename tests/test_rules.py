#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Rules Module Tests
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rules import generate_trading_signals, apply_trading_rules

class TestRules:
    """매매 룰 테스트"""
    
    def setup_method(self):
        """테스트 데이터 설정"""
        # 샘플 데이터 생성
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        self.df = pd.DataFrame({
            'date': dates,
            'z20': np.random.normal(0, 1, 100),
            'rs_4w': np.random.uniform(0.5, 2.0, 100),
            'close': np.random.uniform(50000, 100000, 100),
            'volume': np.random.randint(100000, 10000000, 100)
        })
    
    def test_generate_trading_signals(self):
        """매매 신호 생성 테스트"""
        result_df = generate_trading_signals(self.df)
        
        # 필수 컬럼 확인
        assert 'trading_signal' in result_df.columns
        assert 'signal_strength' in result_df.columns
        assert 'signal_duration' in result_df.columns
        
        # 신호가 'BUY' 또는 'HOLD'인지 확인
        valid_signals = result_df['trading_signal'].unique()
        assert set(valid_signals).issubset({'BUY', 'HOLD'})
        
        # 신호 강도가 0-100 범위인지 확인
        assert result_df['signal_strength'].min() >= 0
        assert result_df['signal_strength'].max() <= 100
        
        # 신호 지속 기간이 양수인지 확인
        assert (result_df['signal_duration'] > 0).all()
    
    def test_signal_conditions(self):
        """신호 조건 테스트"""
        # 명확한 BUY 조건 데이터 생성
        df_buy = self.df.copy()
        df_buy['z20'] = 2.0  # 임계값 이상
        df_buy['rs_4w'] = 1.5  # 임계값 이상
        
        result_df = generate_trading_signals(df_buy)
        
        # 모든 신호가 BUY인지 확인
        assert (result_df['trading_signal'] == 'BUY').all()
        
        # 명확한 HOLD 조건 데이터 생성
        df_hold = self.df.copy()
        df_hold['z20'] = 0.5  # 임계값 미만
        df_hold['rs_4w'] = 0.8  # 임계값 미만
        
        result_df = generate_trading_signals(df_hold)
        
        # 모든 신호가 HOLD인지 확인
        assert (result_df['trading_signal'] == 'HOLD').all()
    
    def test_apply_trading_rules(self):
        """거래 룰 적용 테스트"""
        # 신호가 포함된 데이터 생성
        df_with_signals = generate_trading_signals(self.df)
        
        # 거래 룰 적용
        result_df = apply_trading_rules(df_with_signals)
        
        # 필수 컬럼 확인
        assert 'trading_signal' in result_df.columns
        
        # 신호가 여전히 유효한 값인지 확인
        valid_signals = result_df['trading_signal'].unique()
        assert set(valid_signals).issubset({'BUY', 'HOLD', 'SELL'})
    
    def test_signal_count_consistency(self):
        """신호 개수 일관성 테스트"""
        # 같은 데이터로 여러 번 처리했을 때 신호 개수가 일치하는지 확인
        result1 = generate_trading_signals(self.df)
        result2 = generate_trading_signals(self.df)
        
        # 신호 분포가 일치하는지 확인
        signal_counts1 = result1['trading_signal'].value_counts()
        signal_counts2 = result2['trading_signal'].value_counts()
        
        pd.testing.assert_series_equal(signal_counts1, signal_counts2)
    
    def test_missing_columns(self):
        """필수 컬럼 누락 테스트"""
        df_no_z20 = self.df.drop('z20', axis=1)
        
        with pytest.raises(ValueError):
            generate_trading_signals(df_no_z20)
        
        df_no_rs = self.df.drop('rs_4w', axis=1)
        
        with pytest.raises(ValueError):
            generate_trading_signals(df_no_rs)
    
    def test_custom_thresholds(self):
        """사용자 정의 임계값 테스트"""
        config = {
            'z20_threshold': 0.5,
            'rs_threshold': 0.8
        }
        
        result_df = generate_trading_signals(self.df, config)
        
        # 임계값이 적용되었는지 확인
        buy_signals = result_df[result_df['trading_signal'] == 'BUY']
        
        if len(buy_signals) > 0:
            # BUY 신호의 z20이 임계값 이상인지 확인
            assert (buy_signals['z20'] >= 0.5).all()
            # BUY 신호의 rs_4w가 임계값 이상인지 확인
            assert (buy_signals['rs_4w'] > 0.8).all()
    
    def test_signal_strength_calculation(self):
        """신호 강도 계산 테스트"""
        result_df = generate_trading_signals(self.df)
        
        # 신호 강도가 NaN이 아닌지 확인
        assert not result_df['signal_strength'].isna().any()
        
        # 신호 강도가 숫자인지 확인
        assert result_df['signal_strength'].dtype in ['float64', 'int64']
    
    def test_signal_duration_calculation(self):
        """신호 지속 기간 계산 테스트"""
        result_df = generate_trading_signals(self.df)
        
        # 신호 지속 기간이 양수인지 확인
        assert (result_df['signal_duration'] > 0).all()
        
        # 신호 지속 기간이 정수인지 확인
        assert result_df['signal_duration'].dtype in ['int64', 'float64']
    
    def test_empty_dataframe(self):
        """빈 데이터프레임 처리 테스트"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            generate_trading_signals(empty_df)
    
    def test_single_row_dataframe(self):
        """단일 행 데이터프레임 테스트"""
        single_row_df = self.df.iloc[:1].copy()
        
        result_df = generate_trading_signals(single_row_df)
        
        # 결과가 1행인지 확인
        assert len(result_df) == 1
        
        # 신호가 유효한 값인지 확인
        assert result_df['trading_signal'].iloc[0] in ['BUY', 'HOLD']

if __name__ == "__main__":
    pytest.main([__file__])






