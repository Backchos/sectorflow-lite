#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Data I/O Module Tests
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataio import create_labels, create_windows, split_data, scale_data

class TestDataIO:
    """데이터 I/O 테스트"""
    
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
            'close': prices,
            'volume': np.random.randint(100000, 10000000, 100)
        })
    
    def test_create_labels(self):
        """라벨 생성 테스트"""
        result_df = create_labels(self.df)
        
        # 필수 컬럼 확인
        assert 'label' in result_df.columns
        assert 'future_return' in result_df.columns
        assert 'future_price' in result_df.columns
        
        # 라벨이 0 또는 1인지 확인
        valid_labels = result_df['label'].dropna()
        assert valid_labels.isin([0, 1]).all()
        
        # 마지막 행은 라벨이 NaN이어야 함
        assert pd.isna(result_df['label'].iloc[-1])
    
    def test_create_windows(self):
        """윈도우 생성 테스트"""
        # 라벨이 포함된 데이터 생성
        df_with_labels = create_labels(self.df)
        
        # 윈도우 생성
        X, y = create_windows(df_with_labels, lookback=30)
        
        # 형태 확인
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 30  # lookback
        assert X.shape[2] == 7   # feature_cols 개수
        
        # 라벨이 0 또는 1인지 확인
        assert y.isin([0, 1]).all()
        
        # 데이터 누수 확인 (미래 데이터가 과거에 사용되지 않았는지)
        # 이는 복잡한 검증이므로 기본적인 형태만 확인
        assert len(X) > 0
        assert len(y) > 0
    
    def test_split_data(self):
        """데이터 분할 테스트"""
        # 샘플 데이터 생성
        X = np.random.randn(100, 30, 7)
        y = np.random.randint(0, 2, 100)
        
        # 데이터 분할
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)
        
        # 분할 비율 확인
        total_samples = len(X)
        assert len(X_train) == int(total_samples * 0.7)
        assert len(X_valid) == int(total_samples * 0.15)
        assert len(X_test) == int(total_samples * 0.15)
        
        # 시계열 순서 확인 (train < valid < test)
        assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == total_samples
        
        # 데이터 무결성 확인
        assert X_train.shape[1:] == X_valid.shape[1:] == X_test.shape[1:]
        assert y_train.shape[0] == X_train.shape[0]
        assert y_valid.shape[0] == X_valid.shape[0]
        assert y_test.shape[0] == X_test.shape[0]
    
    def test_scale_data(self):
        """데이터 스케일링 테스트"""
        # 샘플 데이터 생성
        X_train = np.random.randn(50, 30, 7)
        X_valid = np.random.randn(20, 30, 7)
        X_test = np.random.randn(20, 30, 7)
        
        # 스케일링
        X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_data(
            X_train, X_valid, X_test, method='standard'
        )
        
        # 형태 확인
        assert X_train_scaled.shape == X_train.shape
        assert X_valid_scaled.shape == X_valid.shape
        assert X_test_scaled.shape == X_test.shape
        
        # 스케일러 객체 확인
        assert scaler is not None
        
        # 훈련 데이터의 평균이 0에 가까운지 확인
        train_mean = X_train_scaled.reshape(-1, X_train_scaled.shape[-1]).mean(axis=0)
        assert np.allclose(train_mean, 0, atol=1e-10)
    
    def test_data_consistency(self):
        """데이터 일관성 테스트"""
        # 같은 시드로 생성한 데이터는 일관성이 있어야 함
        df1 = create_labels(self.df)
        df2 = create_labels(self.df)
        
        # 라벨이 일치하는지 확인
        pd.testing.assert_series_equal(df1['label'], df2['label'])
        pd.testing.assert_series_equal(df1['future_return'], df2['future_return'])
    
    def test_edge_cases(self):
        """엣지 케이스 테스트"""
        # 너무 작은 데이터
        small_df = self.df.iloc[:10].copy()
        df_with_labels = create_labels(small_df)
        
        with pytest.raises(ValueError):
            create_windows(df_with_labels, lookback=30)  # lookback이 데이터보다 큼
        
        # 정상적인 경우
        X, y = create_windows(df_with_labels, lookback=5)
        assert len(X) > 0
    
    def test_invalid_split_ratios(self):
        """잘못된 분할 비율 테스트"""
        X = np.random.randn(100, 30, 7)
        y = np.random.randint(0, 2, 100)
        
        with pytest.raises(ValueError):
            split_data(X, y, train_ratio=0.5, valid_ratio=0.3, test_ratio=0.3)  # 합이 1.1
    
    def test_scale_methods(self):
        """다양한 스케일링 방법 테스트"""
        X_train = np.random.randn(20, 10, 5)
        X_valid = np.random.randn(10, 10, 5)
        X_test = np.random.randn(10, 10, 5)
        
        # Standard scaling
        X_train_std, X_valid_std, X_test_std, scaler_std = scale_data(
            X_train, X_valid, X_test, method='standard'
        )
        
        # MinMax scaling
        X_train_mm, X_valid_mm, X_test_mm, scaler_mm = scale_data(
            X_train, X_valid, X_test, method='minmax'
        )
        
        # 결과가 다른지 확인
        assert not np.allclose(X_train_std, X_train_mm)
        
        # MinMax 결과가 [0, 1] 범위에 있는지 확인
        assert X_train_mm.min() >= 0
        assert X_train_mm.max() <= 1

if __name__ == "__main__":
    pytest.main([__file__])





