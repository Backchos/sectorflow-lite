#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Features Module Tests
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import calculate_trading_value_zscore, calculate_rs_indicator, process_features

class TestFeatures:
    """피처 계산 테스트"""
    
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
            'volume': np.random.randint(100000, 10000000, 100)
        })
        
        # High, Low 조정
        self.df['high'] = np.maximum(self.df['high'], self.df['close'])
        self.df['low'] = np.minimum(self.df['low'], self.df['close'])
    
    def test_trading_value_zscore(self):
        """거래대금 Z-score 계산 테스트"""
        result_df = calculate_trading_value_zscore(self.df)
        
        # 필수 컬럼 확인
        assert 'z20' in result_df.columns
        assert 'trading_value' in result_df.columns
        assert 'trading_value_mean' in result_df.columns
        assert 'trading_value_std' in result_df.columns
        
        # NaN 값 확인 (처음 20개는 NaN이어야 함)
        assert result_df['z20'].isna().sum() >= 20
        
        # 무한대 값 없음 확인
        assert not np.isinf(result_df['z20']).any()
    
    def test_rs_indicator(self):
        """RS 지표 계산 테스트"""
        result_df = calculate_rs_indicator(self.df)
        
        # 필수 컬럼 확인
        assert 'rs_4w' in result_df.columns
        
        # NaN 값 확인 (처음 20개는 NaN이어야 함)
        assert result_df['rs_4w'].isna().sum() >= 20
        
        # 무한대 값 없음 확인
        assert not np.isinf(result_df['rs_4w']).any()
        
        # RS 값이 양수인지 확인
        valid_rs = result_df['rs_4w'].dropna()
        assert (valid_rs >= 0).all()
    
    def test_process_features(self):
        """전체 피처 처리 테스트"""
        result_df = process_features(self.df)
        
        # 필수 컬럼 확인
        required_cols = ['z20', 'rs_4w']
        for col in required_cols:
            assert col in result_df.columns
        
        # 데이터 무결성 확인
        assert len(result_df) == len(self.df)
        
        # NaN 비율 확인 (너무 많으면 안됨)
        for col in required_cols:
            nan_ratio = result_df[col].isna().sum() / len(result_df)
            assert nan_ratio < 0.5, f"{col}의 NaN 비율이 너무 높음: {nan_ratio:.2%}"
    
    def test_feature_consistency(self):
        """피처 일관성 테스트"""
        # 같은 데이터로 여러 번 처리했을 때 결과가 일치하는지 확인
        result1 = process_features(self.df)
        result2 = process_features(self.df)
        
        # z20과 rs_4w 값이 일치하는지 확인
        pd.testing.assert_series_equal(result1['z20'], result2['z20'])
        pd.testing.assert_series_equal(result1['rs_4w'], result2['rs_4w'])
    
    def test_empty_dataframe(self):
        """빈 데이터프레임 처리 테스트"""
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            calculate_trading_value_zscore(empty_df)
        
        with pytest.raises((ValueError, KeyError)):
            calculate_rs_indicator(empty_df)
    
    def test_missing_columns(self):
        """필수 컬럼 누락 테스트"""
        df_no_close = self.df.drop('close', axis=1)
        
        with pytest.raises(ValueError):
            calculate_rs_indicator(df_no_close)
    
    def test_single_row_dataframe(self):
        """단일 행 데이터프레임 테스트"""
        single_row_df = self.df.iloc[:1].copy()
        
        # Z-score는 최소 2개 행이 필요
        with pytest.raises((ValueError, IndexError)):
            calculate_trading_value_zscore(single_row_df)
        
        # RS도 최소 2개 행이 필요
        with pytest.raises((ValueError, IndexError)):
            calculate_rs_indicator(single_row_df)

if __name__ == "__main__":
    pytest.main([__file__])






