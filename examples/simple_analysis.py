#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - 간단한 분석 도구
한 번에 모든 것을 보여주는 버전
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 백엔드 사용
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def get_stock_name(symbol):
    """종목 코드를 종목 이름으로 변환"""
    stock_names = {
        "005930": "삼성전자",
        "000660": "SK하이닉스", 
        "035720": "카카오",
        "051910": "LG화학",
        "259960": "크래프톤",
        "003550": "LG",
        "180640": "한진칼",
        "034730": "SK",
        "068270": "셀트리온",
        "207940": "삼성바이오로직스",
        "066570": "LG전자",
        "323410": "카카오뱅크",
        "000270": "기아",
        "161890": "한국전력",
        "032830": "삼성생명",
        "000810": "삼성화재",
        "017670": "SK텔레콤",
        "006400": "삼성SDI",
        "000720": "현대건설",
        "105560": "KB금융",
        "012330": "현대모비스",
        "003670": "포스코홀딩스",
        "015760": "한국전력공사",
        "018260": "삼성에스디에스",
        "086280": "현대글로비스",
        "003490": "대한항공",
        "024110": "기업은행",
        "000990": "DB하이텍",
        "011200": "HMM",
        "128940": "한미반도체"
    }
    return stock_names.get(symbol, symbol)

def main():
    """메인 함수"""
    print("=" * 60)
    print("🚀 SectorFlow Lite - 간단한 주식 분석 도구")
    print("=" * 60)
    
    # 데이터 디렉토리 확인
    data_dir = "data/raw"
    if not os.path.exists(data_dir):
        print(f"❌ 데이터 디렉토리가 없습니다: {data_dir}")
        return
    
    # 사용 가능한 종목 목록
    available_symbols = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            symbol = file.replace('.csv', '')
            available_symbols.append(symbol)
    
    print(f"📁 사용 가능한 종목: {len(available_symbols)}개")
    print("종목 목록:", ", ".join(available_symbols[:10]) + ("..." if len(available_symbols) > 10 else ""))
    
    # KOSPI 상위 30개 종목
    kospi30 = [
        "005930", "000660", "035720", "051910", "259960",
        "003550", "180640", "034730", "068270", "207940",
        "066570", "323410", "000270", "161890", "032830",
        "000810", "017670", "006400", "000720", "105560",
        "012330", "003670", "015760", "018260", "086280",
        "003490", "024110", "000990", "011200", "128940"
    ]
    
    print(f"\n🔍 KOSPI 상위 30개 종목 분석 시작...")
    
    # 데이터 로드 및 분석
    results = []
    loaded_data = {}
    
    for symbol in kospi30:
        file_path = f"{data_dir}/{symbol}.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df['date'] = pd.to_datetime(df['date'])
                
                # 기본 계산
                df['returns'] = df['close'].pct_change()
                df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
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
                
                loaded_data[symbol] = df
                print(f"✅ {symbol} 분석 완료")
                
            except Exception as e:
                print(f"❌ {symbol} 분석 실패: {e}")
    
    if not results:
        print("❌ 분석 결과가 없습니다.")
        return
    
    # 수익률 기준 정렬
    results.sort(key=lambda x: x['total_return'], reverse=True)
    
    # 통계 계산
    total_return = np.mean([r['total_return'] for r in results])
    positive_count = sum(1 for r in results if r['total_return'] > 0)
    avg_volatility = np.mean([r['volatility'] for r in results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
    
    # 결과 출력
    print(f"\n📈 분석 완료!")
    print(f"  📊 분석 종목: {len(results)}개")
    print(f"  📈 평균 수익률: {total_return*100:.2f}%")
    print(f"  📈 상승 종목: {positive_count}개")
    print(f"  📈 평균 변동성: {avg_volatility*100:.2f}%")
    print(f"  📈 평균 샤프비율: {avg_sharpe:.2f}")
    
    # 상위 10개 종목 출력
    print(f"\n🏆 상위 10개 종목:")
    for i, result in enumerate(results[:10]):
        status = "📈" if result['total_return'] > 0 else "📉"
        stock_name = get_stock_name(result['symbol'])
        print(f"  {i+1:2d}. {result['symbol']} ({stock_name}): {result['total_return']*100:7.2f}% {status}")
    
    # 차트 생성
    if loaded_data:
        plt.figure(figsize=(15, 8))
        
        for symbol, df in loaded_data.items():
            if not df.empty:
                plt.plot(df['date'], df['close'], label=symbol, linewidth=2)
        
        plt.title("KOSPI 상위 30개 종목 주가 추이", fontsize=16, fontweight='bold')
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('주가', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 차트 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = f"kospi30_chart_{timestamp}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📈 차트 저장: {chart_path}")
    
    # HTML 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = f"kospi30_result_{timestamp}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>KOSPI 상위 30개 분석 결과</title>
        <style>
            body {{ font-family: 'Malgun Gothic', sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; }}
            .stat-number {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
            .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
            .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            .table th {{ background-color: #f8f9fa; font-weight: bold; }}
            .positive {{ color: #27ae60; font-weight: bold; }}
            .negative {{ color: #e74c3c; font-weight: bold; }}
            .chart-container {{ text-align: center; margin: 20px 0; }}
            .chart-container img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 KOSPI 상위 30개 종목 분석 결과</h1>
                <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{len(results)}</div>
                    <div class="stat-label">분석 종목</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{(total_return * 100):.2f}%</div>
                    <div class="stat-label">평균 수익률</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{positive_count}</div>
                    <div class="stat-label">상승 종목</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{(avg_volatility * 100):.2f}%</div>
                    <div class="stat-label">평균 변동성</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>📈 주가 추이</h3>
                <img src="{chart_path}" alt="주가 차트">
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
        return_class = 'positive' if result['total_return'] > 0 else 'negative'
        stock_name = get_stock_name(result['symbol'])
        html_content += f"""
                    <tr>
                        <td>{i + 1}</td>
                        <td>{result['symbol']}<br><small>({stock_name})</small></td>
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
    </body>
    </html>
    """
    
    # HTML 파일 저장
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 HTML 결과 저장: {html_path}")
    
    print(f"\n✅ 모든 분석이 완료되었습니다!")
    print(f"  📈 차트: {chart_path}")
    print(f"  📄 HTML 결과: {html_path}")
    print(f"\n🌐 HTML 파일을 브라우저에서 열어보세요!")

if __name__ == "__main__":
    main()