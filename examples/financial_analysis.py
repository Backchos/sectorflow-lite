#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
금융 데이터 분석 기본 스크립트
현재 사용 가능한 패키지: pandas, numpy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_stock_data(symbol="AAPL", days=30):
    """
    샘플 주식 데이터 생성
    실제 데이터가 없으므로 랜덤하게 생성
    """
    print(f"📊 {symbol} 주식 데이터 생성 중...")
    
    # 날짜 범위 생성
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 초기 가격 설정
    initial_price = 150.0
    
    # 랜덤 워크로 주가 생성 (현실적인 변동성)
    prices = [initial_price]
    for i in range(1, len(dates)):
        # 일일 수익률 (-3% ~ +3% 범위)
        daily_return = random.uniform(-0.03, 0.03)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * random.uniform(1.0, 1.02) for p in prices],
        'Low': [p * random.uniform(0.98, 1.0) for p in prices],
        'Close': prices,
        'Volume': [random.randint(1000000, 5000000) for _ in range(len(dates))]
    })
    
    # High, Low 조정 (High >= Close, Low <= Close)
    df['High'] = np.maximum(df['High'], df['Close'])
    df['Low'] = np.minimum(df['Low'], df['Close'])
    
    return df

def calculate_returns(df):
    """수익률 계산"""
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    return df

def basic_statistics(df):
    """기본 통계 분석"""
    print("\n📈 기본 통계 분석")
    print("=" * 50)
    
    # 가격 통계
    print(f"최고가: ${df['High'].max():.2f}")
    print(f"최저가: ${df['Low'].min():.2f}")
    print(f"평균가: ${df['Close'].mean():.2f}")
    print(f"표준편차: ${df['Close'].std():.2f}")
    
    # 수익률 통계
    daily_returns = df['Daily_Return'].dropna()
    print(f"\n일일 수익률 통계:")
    print(f"평균 수익률: {daily_returns.mean()*100:.2f}%")
    print(f"수익률 표준편차: {daily_returns.std()*100:.2f}%")
    print(f"최대 일일 수익률: {daily_returns.max()*100:.2f}%")
    print(f"최대 일일 손실: {daily_returns.min()*100:.2f}%")
    
    # 변동성 (연간화)
    annual_volatility = daily_returns.std() * np.sqrt(252)
    print(f"연간 변동성: {annual_volatility*100:.2f}%")
    
    return {
        'max_price': df['High'].max(),
        'min_price': df['Low'].min(),
        'avg_price': df['Close'].mean(),
        'volatility': annual_volatility,
        'avg_return': daily_returns.mean()
    }

def simple_technical_analysis(df):
    """간단한 기술적 분석"""
    print("\n🔍 기술적 분석")
    print("=" * 50)
    
    # 이동평균 계산
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    
    # 현재 가격과 이동평균 비교
    current_price = df['Close'].iloc[-1]
    ma_5 = df['MA_5'].iloc[-1]
    ma_10 = df['MA_10'].iloc[-1]
    
    print(f"현재 가격: ${current_price:.2f}")
    print(f"5일 이동평균: ${ma_5:.2f}")
    print(f"10일 이동평균: ${ma_10:.2f}")
    
    # 매매 신호
    if current_price > ma_5 > ma_10:
        signal = "🟢 매수 신호 (상승 추세)"
    elif current_price < ma_5 < ma_10:
        signal = "🔴 매도 신호 (하락 추세)"
    else:
        signal = "🟡 중립 (횡보)"
    
    print(f"매매 신호: {signal}")
    
    return df

def main():
    """메인 함수"""
    print("🚀 금융 데이터 분석 시작!")
    print("=" * 50)
    
    # 1. 샘플 데이터 생성
    df = create_sample_stock_data("AAPL", 30)
    print(f"✅ {len(df)}일간의 데이터 생성 완료")
    
    # 2. 수익률 계산
    df = calculate_returns(df)
    
    # 3. 기본 통계
    stats = basic_statistics(df)
    
    # 4. 기술적 분석
    df = simple_technical_analysis(df)
    
    # 5. 데이터 미리보기
    print("\n📋 데이터 미리보기 (최근 5일)")
    print("=" * 50)
    print(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']].tail())
    
    print("\n✅ 분석 완료!")
    return df, stats

if __name__ == "__main__":
    df, stats = main()

