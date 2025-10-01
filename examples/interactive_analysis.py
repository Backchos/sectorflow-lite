#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - 사용자 입력 기반 주식 분석 도구
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
import yfinance as yf
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def get_user_input():
    """사용자로부터 분석할 종목 입력받기"""
    print("=" * 60)
    print("📈 SectorFlow Lite - 사용자 맞춤 주식 분석")
    print("=" * 60)
    
    print("\n🔍 분석할 종목을 입력하세요:")
    print("1. 종목코드 (예: 005930, 000660)")
    print("2. 종목명 (예: 삼성전자, SK하이닉스)")
    print("3. 여러 종목 (쉼표로 구분: 005930,000660,035720)")
    print("4. KOSPI 상위 30개 (기본값)")
    
    choice = input("\n선택하세요 (1-4, 기본값: 4): ").strip()
    
    if choice == "1":
        symbol = input("종목코드를 입력하세요 (예: 005930): ").strip()
        return [symbol], f"{symbol} 종목"
    
    elif choice == "2":
        symbol_name = input("종목명을 입력하세요 (예: 삼성전자): ").strip()
        # 종목명을 코드로 변환하는 간단한 매핑
        name_to_code = {
            "삼성전자": "005930.KS",
            "SK하이닉스": "000660.KS", 
            "카카오": "035720.KS",
            "LG화학": "051910.KS",
            "크래프톤": "259960.KS",
            "LG": "003550.KS",
            "한진칼": "180640.KS",
            "SK": "034730.KS",
            "셀트리온": "068270.KS",
            "삼성바이오로직스": "207940.KS"
        }
        
        if symbol_name in name_to_code:
            return [name_to_code[symbol_name]], f"{symbol_name} 종목"
        else:
            print(f"⚠️ '{symbol_name}' 종목을 찾을 수 없습니다.")
            print("사용 가능한 종목:", list(name_to_code.keys()))
            return get_user_input()
    
    elif choice == "3":
        symbols_input = input("종목코드들을 쉼표로 구분하여 입력하세요: ").strip()
        symbols = [s.strip() + ".KS" for s in symbols_input.split(",")]
        return symbols, f"{len(symbols)}개 종목"
    
    else:  # 기본값: KOSPI 상위 30개
        kospi_symbols = [
            "005930.KS", "000660.KS", "035720.KS", "051910.KS", "259960.KS",
            "003550.KS", "180640.KS", "034730.KS", "068270.KS", "207940.KS",
            "066570.KS", "323410.KS", "000270.KS", "161890.KS", "032830.KS",
            "000810.KS", "017670.KS", "006400.KS", "000720.KS", "105560.KS",
            "012330.KS", "003670.KS", "015760.KS", "018260.KS", "086280.KS",
            "003490.KS", "024110.KS", "000990.KS", "011200.KS", "128940.KS"
        ]
        return kospi_symbols, "KOSPI 상위 30개 종목"

def get_date_range():
    """분석 기간 설정"""
    print("\n📅 분석 기간을 설정하세요:")
    print("1. 최근 1개월")
    print("2. 최근 3개월 (기본값)")
    print("3. 최근 6개월")
    print("4. 최근 1년")
    print("5. 사용자 지정")
    
    choice = input("선택하세요 (1-5, 기본값: 2): ").strip()
    
    end_date = datetime.now()
    
    if choice == "1":
        start_date = end_date - timedelta(days=30)
    elif choice == "3":
        start_date = end_date - timedelta(days=180)
    elif choice == "4":
        start_date = end_date - timedelta(days=365)
    elif choice == "5":
        try:
            start_str = input("시작일을 입력하세요 (YYYY-MM-DD): ").strip()
            end_str = input("종료일을 입력하세요 (YYYY-MM-DD, 기본값: 오늘): ").strip()
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            if end_str:
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
        except ValueError:
            print("⚠️ 날짜 형식이 올바르지 않습니다. 기본값(3개월)을 사용합니다.")
            start_date = end_date - timedelta(days=90)
    else:  # 기본값: 3개월
        start_date = end_date - timedelta(days=90)
    
    return start_date, end_date

def fetch_stock_data(symbols, start_date, end_date):
    """주식 데이터 수집"""
    print(f"\n📊 {len(symbols)}개 종목 데이터 수집 중...")
    print(f"📅 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    
    all_data = {}
    failed_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"  [{i}/{len(symbols)}] {symbol} 수집 중...", end=" ")
            
            # yfinance로 데이터 수집
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print("❌ 데이터 없음")
                failed_symbols.append(symbol)
                continue
            
            # 컬럼명 정리
            df.reset_index(inplace=True)
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            all_data[symbol] = df
            print("✅")
            
        except Exception as e:
            print(f"❌ 오류: {str(e)[:50]}")
            failed_symbols.append(symbol)
    
    if failed_symbols:
        print(f"\n⚠️ {len(failed_symbols)}개 종목 수집 실패: {failed_symbols}")
    
    print(f"✅ {len(all_data)}개 종목 데이터 수집 완료")
    return all_data

def calculate_analysis_metrics(all_data):
    """분석 지표 계산"""
    print("\n🔧 분석 지표 계산 중...")
    
    results = {}
    
    for symbol, df in all_data.items():
        if df.empty:
            continue
            
        # 기본 수익률 계산
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 누적 수익률
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        # 변동성 (20일 롤링)
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # 샤프 비율 (연간화)
        if df['returns'].std() > 0:
            sharpe_ratio = (df['returns'].mean() * 252) / (df['returns'].std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # 최대 낙폭 (Max Drawdown)
        rolling_max = df['cumulative_returns'].expanding().max()
        drawdown = df['cumulative_returns'] - rolling_max
        max_drawdown = drawdown.min()
        
        # 거래량 분석
        avg_volume = df['volume'].mean()
        volume_volatility = df['volume'].std() / avg_volume if avg_volume > 0 else 0
        
        results[symbol] = {
            'total_return': df['cumulative_returns'].iloc[-1] if not df.empty else 0,
            'volatility': df['volatility'].iloc[-1] if not df.empty else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_volume': avg_volume,
            'volume_volatility': volume_volatility,
            'data': df
        }
    
    return results

def generate_analysis_report(results, analysis_title, start_date, end_date):
    """분석 보고서 생성"""
    print(f"\n📝 {analysis_title} 분석 보고서 생성 중...")
    
    # 결과 정렬
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_return'], reverse=True)
    
    # HTML 보고서 생성
    html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{analysis_title} - SectorFlow Lite 분석</title>
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
        .summary {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 {analysis_title}</h1>
            <p>SectorFlow Lite - AI 기반 주식 분석 시스템</p>
            <p>분석 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}</p>
        </div>
        
        <div class="content">
"""
    
    # 통계 요약
    total_return = np.mean([r['total_return'] for r in results.values()])
    positive_count = sum(1 for r in results.values() if r['total_return'] > 0)
    avg_volatility = np.mean([r['volatility'] for r in results.values()])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results.values()])
    
    html_content += f"""
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{len(results)}</div>
                    <div class="stat-label">분석 종목</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{total_return:.2%}</div>
                    <div class="stat-label">평균 수익률</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{positive_count}</div>
                    <div class="stat-label">상승 종목</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{avg_volatility:.2%}</div>
                    <div class="stat-label">평균 변동성</div>
                </div>
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
    
    # 종목별 결과 테이블
    for i, (symbol, metrics) in enumerate(sorted_results, 1):
        return_class = "positive" if metrics['total_return'] > 0 else "negative"
        html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{symbol}</td>
                        <td class="{return_class}">{metrics['total_return']:.2%}</td>
                        <td>{metrics['volatility']:.2%}</td>
                        <td>{metrics['sharpe_ratio']:.2f}</td>
                        <td>{metrics['max_drawdown']:.2%}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
            
            <div class="summary">
                <h3>📈 분석 요약</h3>
                <ul>
                    <li><strong>분석 시점:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</li>
                    <li><strong>분석 도구:</strong> SectorFlow Lite v1.0</li>
                    <li><strong>데이터 소스:</strong> Yahoo Finance (yfinance)</li>
                    <li><strong>분석 방법:</strong> 기술적 분석 + 리스크 지표</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # HTML 파일 저장
    filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 분석 보고서 생성 완료: {filename}")
    return filename

def main():
    """메인 실행 함수"""
    try:
        # 사용자 입력 받기
        symbols, analysis_title = get_user_input()
        start_date, end_date = get_date_range()
        
        # 데이터 수집
        all_data = fetch_stock_data(symbols, start_date, end_date)
        
        if not all_data:
            print("❌ 수집된 데이터가 없습니다. 프로그램을 종료합니다.")
            return
        
        # 분석 지표 계산
        results = calculate_analysis_metrics(all_data)
        
        # 보고서 생성
        report_file = generate_analysis_report(results, analysis_title, start_date, end_date)
        
        print(f"\n🎉 분석 완료!")
        print(f"📊 분석 대상: {analysis_title}")
        print(f"📅 분석 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print(f"📝 보고서 파일: {report_file}")
        print(f"\n💡 보고서를 열려면 파일을 더블클릭하세요!")
        
    except KeyboardInterrupt:
        print("\n\n👋 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()

