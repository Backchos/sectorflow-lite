#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한글 폰트 문제 해결 및 차트 재생성
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 백엔드 사용
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

def setup_korean_font():
    """한글 폰트 설정"""
    print("🔧 한글 폰트 설정 중...")
    
    # Windows에서 사용 가능한 한글 폰트 찾기
    font_candidates = [
        'Malgun Gothic',  # 맑은 고딕
        'Microsoft YaHei',  # 마이크로소프트 야헤이
        'SimHei',  # 심헤이
        'Arial Unicode MS',  # Arial Unicode MS
        'DejaVu Sans'  # 기본 폰트
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"✅ 한글 폰트 설정 완료: {font}")
            return font
    
    # 폰트를 찾지 못한 경우
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    return 'DejaVu Sans'

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

def create_korean_charts(df_analysis, results):
    """한글 지원 차트 생성"""
    print("📈 한글 지원 차트 생성 중...")
    
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
    plt.xlabel('Return Rate (%)', fontsize=12)
    plt.title('Top 15 KOSPI Stocks Return Rate (Sep 19, 2024)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for i, (bar, value) in enumerate(zip(bars, top_15['total_return'])):
        plt.text(bar.get_width() + (0.5 if value > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%', ha='left' if value > 0 else 'right', va='center', fontsize=10)
    
    # 2. 수익률 vs 변동성 산점도
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(df_analysis['volatility'], df_analysis['total_return'], 
                         c=df_analysis['sharpe_ratio'], cmap='RdYlGn', alpha=0.7, s=100)
    plt.xlabel('Volatility (%)', fontsize=12)
    plt.ylabel('Return Rate (%)', fontsize=12)
    plt.title('Return Rate vs Volatility (Color: Sharpe Ratio)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # 3. 거래량 상위 10개 종목
    plt.subplot(2, 2, 3)
    top_volume = df_analysis.nlargest(10, 'avg_trading_value')
    bars = plt.bar(range(len(top_volume)), top_volume['avg_trading_value'] / 1e8, 
                   color='#3498db', alpha=0.7)
    plt.xticks(range(len(top_volume)), top_volume['symbol'], rotation=45)
    plt.ylabel('Average Trading Value (100M KRW)', fontsize=12)
    plt.title('Top 10 Stocks by Trading Value', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. 최대 낙폭 분포
    plt.subplot(2, 2, 4)
    plt.hist(df_analysis['max_drawdown'], bins=10, color='#9b59b6', alpha=0.7, edgecolor='black')
    plt.xlabel('Max Drawdown (%)', fontsize=12)
    plt.ylabel('Number of Stocks', fontsize=12)
    plt.title('Max Drawdown Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/charts/kospi_analysis_overview_korean.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 개별 종목 상세 차트 (상위 5개)
    create_individual_stock_charts_korean(results, df_analysis.head(5))
    
    # 6. 시장 전체 트렌드
    create_market_trend_chart_korean(results)
    
    print("✅ 한글 지원 차트 생성 완료!")

def create_individual_stock_charts_korean(results, top_stocks):
    """개별 종목 상세 차트 (한글 지원)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (_, stock) in enumerate(top_stocks.iterrows()):
        if i >= 6:
            break
            
        symbol = stock['symbol']
        data = results[symbol]['data']
        
        ax = axes[i]
        
        # 가격 차트
        ax.plot(data['date'], data['close'], label='Close Price', linewidth=2, color='#2c3e50')
        ax.plot(data['date'], data['ma_5'], label='MA5', alpha=0.7, color='#e74c3c')
        ax.plot(data['date'], data['ma_20'], label='MA20', alpha=0.7, color='#3498db')
        
        ax.set_title(f'{symbol} - Return: {stock["total_return"]:.1f}%', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (KRW)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # x축 날짜 포맷
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    # 마지막 subplot 제거
    if len(top_stocks) < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('reports/charts/top_stocks_detail_korean.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_market_trend_chart_korean(results):
    """시장 전체 트렌드 차트 (한글 지원)"""
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
        plt.plot(all_dates, market_index, linewidth=2, color='#2c3e50', label='Market Average')
        plt.title('KOSPI Top 30 Average Price Index', fontsize=14, fontweight='bold')
        plt.ylabel('Average Price (KRW)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 일일 수익률
        daily_returns = np.diff(market_index) / market_index[:-1] * 100
        plt.subplot(2, 1, 2)
        plt.plot(all_dates[1:], daily_returns, alpha=0.7, color='#e74c3c')
        plt.title('Daily Return Rate', fontsize=14, fontweight='bold')
        plt.ylabel('Return Rate (%)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/charts/market_trend_korean.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_enhanced_html_report(df_analysis):
    """향상된 HTML 리포트 생성 (한글 완벽 지원)"""
    print("🌐 향상된 HTML 리포트 생성 중...")
    
    # HTML 템플릿
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>코스피 상위 30개 종목 분석 - 9월 19일</title>
        <style>
            body {{
                font-family: 'Malgun Gothic', 'Microsoft YaHei', 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .content {{
                padding: 30px;
            }}
            .chart-container {{
                text-align: center;
                margin: 20px 0;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .stat-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                border-left: 5px solid #3498db;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .stat-label {{
                color: #7f8c8d;
                margin-top: 5px;
            }}
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-family: 'Malgun Gothic', sans-serif;
            }}
            .table th, .table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .table th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            .positive {{
                color: #27ae60;
                font-weight: bold;
            }}
            .negative {{
                color: #e74c3c;
                font-weight: bold;
            }}
            .summary-box {{
                margin-top: 40px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 5px solid #3498db;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 코스피 상위 30개 종목 분석</h1>
                <p>2024년 6월 21일 ~ 9월 19일 (3개월)</p>
                <p>한글 완벽 지원 버전</p>
            </div>
            
            <div class="content">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{len(df_analysis)}</div>
                        <div class="stat-label">분석 종목</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{df_analysis['total_return'].mean():.1f}%</div>
                        <div class="stat-label">평균 수익률</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{(df_analysis['total_return'] > 0).sum()}</div>
                        <div class="stat-label">상승 종목</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{df_analysis['volatility'].mean():.1f}%</div>
                        <div class="stat-label">평균 변동성</div>
                    </div>
                </div>
                
                <h2>📊 분석 차트</h2>
                <div class="chart-container">
                    <h3>종합 분석 차트</h3>
                    <img src="charts/kospi_analysis_overview_korean.png" alt="종합 분석 차트">
                </div>
                
                <div class="chart-container">
                    <h3>상위 종목 상세 차트</h3>
                    <img src="charts/top_stocks_detail_korean.png" alt="상위 종목 상세 차트">
                </div>
                
                <div class="chart-container">
                    <h3>시장 트렌드 차트</h3>
                    <img src="charts/market_trend_korean.png" alt="시장 트렌드 차트">
                </div>
                
                <h2>🏆 상위 10개 종목</h2>
                <table class="table">
                    <thead>
                        <tr>
                            <th>순위</th>
                            <th>종목코드</th>
                            <th>수익률</th>
                            <th>변동성</th>
                            <th>샤프비율</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # 상위 10개 종목 테이블
    for i, (_, row) in enumerate(df_analysis.head(10).iterrows(), 1):
        return_class = "positive" if row['total_return'] > 0 else "negative"
        html_content += f"""
                        <tr>
                            <td>{i}</td>
                            <td>{row['symbol']}</td>
                            <td class="{return_class}">{row['total_return']:.2f}%</td>
                            <td>{row['volatility']:.2f}%</td>
                            <td>{row['sharpe_ratio']:.2f}</td>
                        </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
                
                <h2>📉 하위 10개 종목</h2>
                <table class="table">
                    <thead>
                        <tr>
                            <th>순위</th>
                            <th>종목코드</th>
                            <th>수익률</th>
                            <th>변동성</th>
                            <th>샤프비율</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # 하위 10개 종목 테이블
    for i, (_, row) in enumerate(df_analysis.tail(10).iterrows(), 1):
        return_class = "positive" if row['total_return'] > 0 else "negative"
        html_content += f"""
                        <tr>
                            <td>{len(df_analysis) - 10 + i}</td>
                            <td>{row['symbol']}</td>
                            <td class="{return_class}">{row['total_return']:.2f}%</td>
                            <td>{row['volatility']:.2f}%</td>
                            <td>{row['sharpe_ratio']:.2f}</td>
                        </tr>
        """
    
    html_content += f"""
                    </tbody>
                </table>
                
                <div class="summary-box">
                    <h3>📝 분석 요약</h3>
                    <ul>
                        <li><strong>분석 기간:</strong> 2024년 6월 21일 ~ 9월 19일 (3개월)</li>
                        <li><strong>분석 종목:</strong> 코스피 상위 30개 종목</li>
                        <li><strong>평균 수익률:</strong> {df_analysis['total_return'].mean():.2f}%</li>
                        <li><strong>상승 종목 비율:</strong> {(df_analysis['total_return'] > 0).sum() / len(df_analysis) * 100:.1f}%</li>
                        <li><strong>평균 변동성:</strong> {df_analysis['volatility'].mean():.2f}%</li>
                        <li><strong>최고 수익률:</strong> {df_analysis['total_return'].max():.2f}%</li>
                        <li><strong>최저 수익률:</strong> {df_analysis['total_return'].min():.2f}%</li>
                    </ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    with open("reports/kospi_analysis_report_korean.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ 향상된 HTML 리포트 생성 완료: reports/kospi_analysis_report_korean.html")

def main():
    """메인 실행 함수"""
    print("🚀 SectorFlow Lite - 한글 폰트 문제 해결")
    print("=" * 60)
    
    try:
        # 1. 한글 폰트 설정
        font_name = setup_korean_font()
        
        # 2. 데이터 로드
        all_data, metadata = load_kospi_data()
        
        if not all_data:
            print("❌ 데이터를 로드할 수 없습니다.")
            return
        
        # 3. 수익률 및 피처 계산
        results = calculate_returns_and_features(all_data)
        
        if not results:
            print("❌ 분석할 데이터가 충분하지 않습니다.")
            return
        
        # 4. 결과를 DataFrame으로 변환
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
        
        # 5. 한글 지원 차트 생성
        create_korean_charts(df_analysis, results)
        
        # 6. 향상된 HTML 리포트 생성
        create_enhanced_html_report(df_analysis)
        
        print(f"\n💾 한글 지원 결과가 저장되었습니다:")
        print(f"   - 향상된 HTML 리포트: reports/kospi_analysis_report_korean.html")
        print(f"   - 한글 지원 차트: reports/charts/*_korean.png")
        
        print("\n🎉 한글 폰트 문제가 해결되었습니다!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
