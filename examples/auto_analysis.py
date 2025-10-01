#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - 자동 분석 도구
입력 없이 자동으로 실행
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 백엔드 사용
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_stock_data(symbols):
    """주식 데이터 로드"""
    data_dir = "data/raw"
    available_data = {}
    
    print(f"📁 데이터 디렉토리: {data_dir}")
    
    for symbol in symbols:
        file_path = f"{data_dir}/{symbol}.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df['date'] = pd.to_datetime(df['date'])
                available_data[symbol] = df
                print(f"✅ {symbol} 데이터 로드 완료 ({len(df)}개 레코드)")
            except Exception as e:
                print(f"❌ {symbol} 데이터 로드 실패: {e}")
        else:
            print(f"⚠️ {symbol} 파일 없음: {file_path}")
    
    return available_data

def analyze_stocks(data):
    """주식 분석"""
    results = []
    
    print(f"\n🔍 {len(data)}개 종목 분석 시작...")
    
    for symbol, df in data.items():
        if df.empty:
            continue
        
        print(f"  📊 {symbol} 분석 중...")
        
        # 기본 계산
        df['returns'] = df['close'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        # 변동성 (20일 롤링)
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # 샤프 비율
        if df['returns'].std() > 0:
            sharpe_ratio = (df['returns'].mean() * 252) / (df['returns'].std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # 최대 낙폭
        rolling_max = df['cumulative_returns'].expanding().max()
        drawdown = df['cumulative_returns'] - rolling_max
        max_drawdown = drawdown.min()
        
        # 결과 저장
        total_return = df['cumulative_returns'].iloc[-1] if not df.empty else 0
        volatility = df['volatility'].iloc[-1] if not df.empty else 0
        
        results.append({
            'symbol': symbol,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        })
    
    # 수익률 기준 정렬
    results.sort(key=lambda x: x['total_return'], reverse=True)
    return results

def create_chart(data, title, output_path):
    """차트 생성 및 저장"""
    plt.figure(figsize=(15, 8))
    
    for symbol, df in data.items():
        if not df.empty:
            plt.plot(df['date'], df['close'], label=symbol, linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('주가', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 차트 저장
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 차트 저장: {output_path}")

def save_results_to_html(results, stats, chart_path, output_path):
    """결과를 HTML 파일로 저장"""
    
    # 수익률 기준 색상 결정
    def get_return_class(return_val):
        return 'positive' if return_val > 0 else 'negative'
    
    # HTML 생성
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SectorFlow Lite - 분석 결과</title>
        <style>
            body {{
                font-family: 'Malgun Gothic', sans-serif;
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
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 SectorFlow Lite - 분석 결과</h1>
                <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{stats['total_stocks']}</div>
                        <div class="stat-label">분석 종목</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{(stats['avg_return'] * 100):.2f}%</div>
                        <div class="stat-label">평균 수익률</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['positive_count']}</div>
                        <div class="stat-label">상승 종목</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{(stats['avg_volatility'] * 100):.2f}%</div>
                        <div class="stat-label">평균 변동성</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>📈 주가 추이</h3>
                    <img src="{os.path.basename(chart_path)}" alt="주가 차트">
                </div>
                
                <h2>📊 종목별 분석 결과</h2>
                <table class="table">
                    <thead>
                        <tr>
                            <th>순위</th>
                            <th>종목코드</th>
                            <th>수익률</th>
                            <th>변동성</th>
                            <th>샤프비율</th>
                            <th>최대낙폭</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # 결과 테이블 추가
    for i, result in enumerate(results):
        return_class = get_return_class(result['total_return'])
        html_content += f"""
                        <tr>
                            <td>{i + 1}</td>
                            <td>{result['symbol']}</td>
                            <td class="{return_class}">{(result['total_return'] * 100):.2f}%</td>
                            <td>{(result['volatility'] * 100):.2f}%</td>
                            <td>{result['sharpe_ratio']:.2f}</td>
                            <td>{(result['max_drawdown'] * 100):.2f}%</td>
                        </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 HTML 결과 저장: {output_path}")

def main():
    """메인 함수 - 자동 실행"""
    print("🚀 SectorFlow Lite - 자동 분석 도구")
    print("=" * 50)
    
    # 사용 가능한 종목 목록 표시
    data_dir = "data/raw"
    if not os.path.exists(data_dir):
        print(f"❌ 데이터 디렉토리가 없습니다: {data_dir}")
        return
    
    available_symbols = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            symbol = file.replace('.csv', '')
            available_symbols.append(symbol)
    
    if not available_symbols:
        print("❌ 사용 가능한 데이터 파일이 없습니다.")
        return
    
    print(f"📁 사용 가능한 종목: {len(available_symbols)}개")
    print("종목 목록:", ", ".join(available_symbols[:10]) + ("..." if len(available_symbols) > 10 else ""))
    
    # 자동으로 KOSPI 상위 30개 분석
    print("\n🔍 KOSPI 상위 30개 종목 자동 분석 시작...")
    
    symbols = [
        "005930", "000660", "035720", "051910", "259960",
        "003550", "180640", "034730", "068270", "207940",
        "066570", "323410", "000270", "161890", "032830",
        "000810", "017670", "006400", "000720", "105560",
        "012330", "003670", "015760", "018260", "086280",
        "003490", "024110", "000990", "011200", "128940"
    ]
    
    print(f"📊 {len(symbols)}개 종목 분석 시작...")
    
    # 데이터 로드
    stock_data = load_stock_data(symbols)
    
    if not stock_data:
        print("❌ 사용 가능한 데이터가 없습니다.")
        return
    
    # 분석 실행
    results = analyze_stocks(stock_data)
    
    if not results:
        print("❌ 분석 결과가 없습니다.")
        return
    
    # 통계 계산
    total_return = np.mean([r['total_return'] for r in results])
    positive_count = sum(1 for r in results if r['total_return'] > 0)
    avg_volatility = np.mean([r['volatility'] for r in results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
    
    stats = {
        'total_stocks': len(results),
        'avg_return': total_return,
        'positive_count': positive_count,
        'avg_volatility': avg_volatility,
        'avg_sharpe': avg_sharpe
    }
    
    # 결과 출력
    print(f"\n📈 분석 완료!")
    print(f"  📊 분석 종목: {stats['total_stocks']}개")
    print(f"  📈 평균 수익률: {stats['avg_return']*100:.2f}%")
    print(f"  📈 상승 종목: {stats['positive_count']}개")
    print(f"  📈 평균 변동성: {stats['avg_volatility']*100:.2f}%")
    print(f"  📈 평균 샤프비율: {stats['avg_sharpe']:.2f}")
    
    # 상위 5개 종목 출력
    print(f"\n🏆 상위 5개 종목:")
    for i, result in enumerate(results[:5]):
        print(f"  {i+1}. {result['symbol']}: {result['total_return']*100:.2f}%")
    
    # 차트 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = f"analysis_chart_KOSPI30_{timestamp}.png"
    create_chart(stock_data, "KOSPI 상위 30개 분석 결과", chart_path)
    
    # HTML 결과 저장
    html_path = f"analysis_result_KOSPI30_{timestamp}.html"
    save_results_to_html(results, stats, chart_path, html_path)
    
    print(f"\n✅ 분석 완료!")
    print(f"  📈 차트: {chart_path}")
    print(f"  📄 HTML 결과: {html_path}")
    print(f"\n🌐 HTML 파일을 브라우저에서 열어보세요!")
    
    # 잠시 대기
    input("\n⏸️ 아무 키나 누르면 종료됩니다...")

if __name__ == "__main__":
    main()

