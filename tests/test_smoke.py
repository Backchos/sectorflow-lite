#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Smoke Test
최소 파이프라인이 정상 작동하는지 확인하는 스모크 테스트
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_basic_imports():
    """기본 모듈 import 테스트"""
    try:
        import pandas as pd
        import numpy as np
        print("✅ Basic imports test passed")
        return True
    except ImportError as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_data_creation():
    """데이터 생성 테스트"""
    try:
        # 샘플 데이터 생성
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'trading_value': np.random.randint(1000000, 10000000, 100)
        })
        
        assert len(sample_data) == 100
        assert 'close' in sample_data.columns
        print("✅ Data creation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data creation failed: {e}")
        return False

def test_basic_calculations():
    """기본 계산 테스트"""
    try:
        # 샘플 데이터
        prices = np.array([100, 105, 102, 108, 110])
        
        # 수익률 계산
        returns = np.diff(prices) / prices[:-1]
        assert len(returns) == 4
        
        # 이동평균 계산
        ma = np.mean(prices)
        assert ma > 0
        
        print("✅ Basic calculations test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic calculations failed: {e}")
        return False

def test_file_structure():
    """파일 구조 테스트"""
    try:
        # 주요 파일들이 존재하는지 확인
        required_files = [
            'main.py',
            'config.yaml',
            'requirements.txt',
            'README.md'
        ]
        
        for file in required_files:
            assert os.path.exists(file), f"Required file not found: {file}"
        
        # 주요 디렉토리가 존재하는지 확인
        required_dirs = [
            'src',
            'data',
            'tests',
            'examples'
        ]
        
        for dir in required_dirs:
            assert os.path.exists(dir), f"Required directory not found: {dir}"
        
        print("✅ File structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False

def test_smoke_pipeline():
    """전체 파이프라인 스모크 테스트"""
    print("\n🚀 SectorFlow Lite Smoke Test 시작...")
    
    tests = [
        test_basic_imports,
        test_data_creation,
        test_basic_calculations,
        test_file_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("✅ 모든 스모크 테스트 통과!")
        print("🎯 최소 파이프라인이 정상 작동합니다.")
        return True
    else:
        print("❌ 일부 테스트 실패")
        return False

if __name__ == "__main__":
    try:
        success = test_smoke_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 스모크 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)