#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - 코스피 상위 30개 종목 분석 및 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

# seaborn이 없어도 작동하도록
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("⚠️ seaborn이 설치되지 않았습니다. 기본 matplotlib 스타일을 사용합니다.")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_kospi_data():
    """코스피 데이터 로드"""
    print("📊 코스피 상위 30개 종목 데이터 로드 중...")
    
    # 메타데이터 로드
    with open("data/raw/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # 개별 종목 데이터 로드
    all_data = {}
    symbols = metadata['symbols']
    
    for symbol in symbols:
        try:
            df = pd.read_csv(f"data/raw/{symbol}.csv", encoding='utf-8')
            df['date'] = pd.to_datetime(df['date'])
            all_data[symbol] = df
        except FileNotFoundError:
            print(f"⚠️ {symbol} 데이터 파일을 찾을 수 없습니다.")
    
    print(f"✅ {len(all_data)}개 종목 데이터 로드 완료")
    return all_data, metadata

def calculate_returns_and_features(all_data):
    """수익률 및 피처 계산"""
    print("🔧 수익률 및 피처 계산 중...")
    
    results = {}
    
    for symbol, df in all_data.items():
        if len(df) < 20:  # 최소 20일 데이터 필요
            continue
            
        # 수익률 계산
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 이동평균
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        
        # 변동성 (20일 롤링)
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # 거래대금 Z-score
        df['trading_value'] = df['close'] * df['volume']
        df['trading_value_mean'] = df['trading_value'].rolling(window=20).mean()
        df['trading_value_std'] = df['trading_value'].rolling(window=20).std()
        df['z20'] = (df['trading_value'] - df['trading_value_mean']) / df['trading_value_std']
        
        # RS 지표 (상대강도)
        df['rs_4w'] = df['close'] / df['close'].rolling(window=20).mean()
        
        # 최근 데이터만 사용 (9월 19일 기준)
        recent_df = df.tail(30).copy()  # 최근 30일
        
        results[symbol] = {
            'data': recent_df,
            'total_return': (recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / recent_df['close'].iloc[0] * 100,
            'volatility': recent_df['volatility'].mean(),
            'avg_volume': recent_df['volume'].mean(),
            'avg_trading_value': recent_df['trading_value'].mean(),
            'max_drawdown': calculate_max_drawdown(recent_df['close']),
            'sharpe_ratio': calculate_sharpe_ratio(recent_df['returns'].dropna())
        }
    
    return results

def calculate_max_drawdown(prices):
    """최대 낙폭 계산"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min() * 100

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """샤프 비율 계산"""
    if len(returns) == 0 or returns.std() == 0:
        return 0
    excess_returns = returns.mean() * 252 - risk_free_rate
    return excess_returns / (returns.std() * np.sqrt(252))

def create_comprehensive_analysis(results):
    """종합 분석 및 시각화"""
    print("📈 종합 분석 및 시각화 생성 중...")
    
    # 결과를 DataFrame으로 변환
    analysis_data = []
    for symbol, data in results.items():
        analysis_data.append({
            'symbol': symbol,
            'total_return': data['total_return'],
            'volatility': data['volatility'],
            'avg_volume': data['avg_volume'],
            'avg_trading_value': data['avg_trading_value'],
            'max_drawdown': data['max_drawdown'],
            'sharpe_ratio': data['sharpe_ratio']
        })
    
    df_analysis = pd.DataFrame(analysis_data)
    df_analysis = df_analysis.sort_values('total_return', ascending=False)
    
    # 시각화 생성
    create_visualizations(df_analysis, results)
    
    return df_analysis

def create_visualizations(df_analysis, results):
    """시각화 생성"""
    # 차트 저장 디렉토리 생성
    os.makedirs("reports/charts", exist_ok=True)
    
    # 1. 수익률 순위 차트
    plt.figure(figsize=(15, 10))
    
    # 상위 15개 종목
    top_15 = df_analysis.head(15)
    
    plt.subplot(2, 2, 1)
    colors = ['#e74c3c' if x < 0 else '#27ae60' for x in top_15['total_return']]
    bars = plt.barh(range(len(top_15)), top_15['total_return'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_15)), top_15['symbol'])
    plt.xlabel('수익률 (%)')
    plt.title('코스피 상위 15개 종목 수익률 (9월 19일 기준)')
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for i, (bar, value) in enumerate(zip(bars, top_15['total_return'])):
        plt.text(bar.get_width() + (0.5 if value > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%', ha='left' if value > 0 else 'right', va='center')
    
    # 2. 수익률 vs 변동성 산점도
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(df_analysis['volatility'], df_analysis['total_return'], 
                         c=df_analysis['sharpe_ratio'], cmap='RdYlGn', alpha=0.7, s=100)
    plt.xlabel('변동성 (%)')
    plt.ylabel('수익률 (%)')
    plt.title('수익률 vs 변동성 (색상: 샤프 비율)')
    plt.colorbar(scatter, label='샤프 비율')
    plt.grid(True, alpha=0.3)
    
    # 3. 거래량 상위 10개 종목
    plt.subplot(2, 2, 3)
    top_volume = df_analysis.nlargest(10, 'avg_trading_value')
    bars = plt.bar(range(len(top_volume)), top_volume['avg_trading_value'] / 1e8, 
                   color='#3498db', alpha=0.7)
    plt.xticks(range(len(top_volume)), top_volume['symbol'], rotation=45)
    plt.ylabel('평균 거래대금 (억원)')
    plt.title('거래대금 상위 10개 종목')
    plt.grid(True, alpha=0.3)
    
    # 4. 최대 낙폭 분포
    plt.subplot(2, 2, 4)
    plt.hist(df_analysis['max_drawdown'], bins=10, color='#9b59b6', alpha=0.7, edgecolor='black')
    plt.xlabel('최대 낙폭 (%)')
    plt.ylabel('종목 수')
    plt.title('최대 낙폭 분포')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/charts/kospi_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 개별 종목 상세 차트 (상위 5개)
    create_individual_stock_charts(results, df_analysis.head(5))
    
    # 6. 시장 전체 트렌드
    create_market_trend_chart(results)

def create_individual_stock_charts(results, top_stocks):
    """개별 종목 상세 차트"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (_, stock) in enumerate(top_stocks.iterrows()):
        if i >= 6:
            break
            
        symbol = stock['symbol']
        data = results[symbol]['data']
        
        ax = axes[i]
        
        # 가격 차트
        ax.plot(data['date'], data['close'], label='종가', linewidth=2, color='#2c3e50')
        ax.plot(data['date'], data['ma_5'], label='MA5', alpha=0.7, color='#e74c3c')
        ax.plot(data['date'], data['ma_20'], label='MA20', alpha=0.7, color='#3498db')
        
        ax.set_title(f'{symbol} - 수익률: {stock["total_return"]:.1f}%')
        ax.set_ylabel('가격 (원)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # x축 날짜 포맷
        ax.tick_params(axis='x', rotation=45)
    
    # 마지막 subplot 제거
    if len(top_stocks) < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('reports/charts/top_stocks_detail.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_market_trend_chart(results):
    """시장 전체 트렌드 차트"""
    plt.figure(figsize=(15, 8))
    
    # 모든 종목의 평균 가격 지수 생성
    all_dates = None
    price_matrix = []
    
    for symbol, data in results.items():
        if all_dates is None:
            all_dates = data['data']['date'].values
        price_matrix.append(data['data']['close'].values)
    
    if price_matrix:
        price_matrix = np.array(price_matrix)
        market_index = np.mean(price_matrix, axis=0)
        
        plt.subplot(2, 1, 1)
        plt.plot(all_dates, market_index, linewidth=2, color='#2c3e50', label='시장 평균')
        plt.title('코스피 상위 30개 종목 평균 가격 지수')
        plt.ylabel('평균 가격 (원)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 일일 수익률
        daily_returns = np.diff(market_index) / market_index[:-1] * 100
        plt.subplot(2, 1, 2)
        plt.plot(all_dates[1:], daily_returns, alpha=0.7, color='#e74c3c')
        plt.title('일일 수익률')
        plt.ylabel('수익률 (%)')
        plt.xlabel('날짜')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/charts/market_trend.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(df_analysis):
    """요약 리포트 생성"""
    print("\n📊 분석 결과 요약")
    print("=" * 60)
    
    print(f"📅 분석 기간: 2024년 6월 21일 ~ 9월 19일 (3개월)")
    print(f"📈 분석 종목: {len(df_analysis)}개")
    print()
    
    print("🏆 상위 5개 종목:")
    for i, (_, row) in enumerate(df_analysis.head().iterrows(), 1):
        print(f"   {i}. {row['symbol']}: {row['total_return']:.2f}% (변동성: {row['volatility']:.2f}%)")
    
    print("\n📉 하위 5개 종목:")
    for i, (_, row) in enumerate(df_analysis.tail().iterrows(), 1):
        print(f"   {i}. {row['symbol']}: {row['total_return']:.2f}% (변동성: {row['volatility']:.2f}%)")
    
    print(f"\n📊 전체 통계:")
    print(f"   - 평균 수익률: {df_analysis['total_return'].mean():.2f}%")
    print(f"   - 중간 수익률: {df_analysis['total_return'].median():.2f}%")
    print(f"   - 최고 수익률: {df_analysis['total_return'].max():.2f}%")
    print(f"   - 최저 수익률: {df_analysis['total_return'].min():.2f}%")
    print(f"   - 평균 변동성: {df_analysis['volatility'].mean():.2f}%")
    print(f"   - 평균 샤프 비율: {df_analysis['sharpe_ratio'].mean():.2f}")
    
    # 수익률 분포
    positive_returns = (df_analysis['total_return'] > 0).sum()
    print(f"   - 상승 종목: {positive_returns}개 ({positive_returns/len(df_analysis)*100:.1f}%)")
    print(f"   - 하락 종목: {len(df_analysis) - positive_returns}개 ({(len(df_analysis) - positive_returns)/len(df_analysis)*100:.1f}%)")

def main():
    """메인 실행 함수"""
    print("🚀 SectorFlow Lite - 코스피 상위 30개 종목 분석")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        all_data, metadata = load_kospi_data()
        
        if not all_data:
            print("❌ 데이터를 로드할 수 없습니다.")
            return
        
        # 2. 수익률 및 피처 계산
        results = calculate_returns_and_features(all_data)
        
        if not results:
            print("❌ 분석할 데이터가 충분하지 않습니다.")
            return
        
        # 3. 종합 분석
        df_analysis = create_comprehensive_analysis(results)
        
        # 4. 요약 리포트
        generate_summary_report(df_analysis)
        
        # 5. 결과 저장
        df_analysis.to_csv("reports/kospi_analysis_results.csv", index=False, encoding='utf-8')
        print(f"\n💾 분석 결과가 저장되었습니다:")
        print(f"   - 상세 결과: reports/kospi_analysis_results.csv")
        print(f"   - 차트: reports/charts/")
        
        print("\n🎉 분석이 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
