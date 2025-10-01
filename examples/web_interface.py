#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - 웹 인터페이스
외부에서 브라우저로 접근하여 분석 실행 가능
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import subprocess
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__, template_folder='../templates')

def get_kospi_top30():
    """코스피 상위 30개 종목 코드 반환"""
    return [
        "005930", "000660", "035420", "005380", "006400",
        "051910", "035720", "000270", "068270", "207940",
        "066570", "323410", "105560", "055550", "012330",
        "003550", "096770", "017670", "018260", "086790",
        "032830", "003490", "015760", "000810", "034730",
        "161890", "259960", "180640", "302440", "024110"
    ]

def generate_sample_data(symbol, start_date, end_date):
    """샘플 데이터 생성"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = [d for d in dates if d.weekday() < 5]  # 주말 제외
    
    # 초기 가격 설정
    base_prices = {
        "005930": 70000, "000660": 120000, "035420": 180000,
        "005380": 250000, "006400": 400000, "051910": 600000,
        "035720": 50000, "000270": 100000, "068270": 200000,
        "207940": 800000
    }
    
    base_price = base_prices.get(symbol, 50000)
    
    # 가격 데이터 생성
    np.random.seed(hash(symbol) % 2**32)
    prices = [base_price]
    
    for i in range(len(dates) - 1):
        daily_return = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 1000))
    
    # OHLC 데이터 생성
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
        
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.randint(100000, 10000000)
        
        data.append({
            'date': date,
            'open': round(open_price, 0),
            'high': round(high, 0),
            'low': round(low, 0),
            'close': round(close, 0),
            'volume': volume,
            'trading_value': round(close * volume, 0)
        })
    
    return pd.DataFrame(data)

def analyze_stocks(symbols, start_date, end_date):
    """종목 분석 실행"""
    print(f"📊 {len(symbols)}개 종목 분석 시작...")
    
    results = []
    
    for symbol in symbols:
        try:
            # 데이터 생성
            df = generate_sample_data(symbol, start_date, end_date)
            
            if len(df) < 20:
                continue
            
            # 수익률 계산
            df['returns'] = df['close'].pct_change()
            df['ma_5'] = df['close'].rolling(window=5).mean()
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            
            # 거래대금 Z-score
            df['trading_value'] = df['close'] * df['volume']
            df['trading_value_mean'] = df['trading_value'].rolling(window=20).mean()
            df['trading_value_std'] = df['trading_value'].rolling(window=20).std()
            df['z20'] = (df['trading_value'] - df['trading_value_mean']) / df['trading_value_std']
            
            # RS 지표
            df['rs_4w'] = df['close'] / df['close'].rolling(window=20).mean()
            
            # 최근 데이터 분석
            recent_df = df.tail(30).copy()
            
            total_return = (recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / recent_df['close'].iloc[0] * 100
            volatility = recent_df['volatility'].mean()
            avg_volume = recent_df['volume'].mean()
            avg_trading_value = recent_df['trading_value'].mean()
            
            # 최대 낙폭 계산
            peak = recent_df['close'].expanding().max()
            drawdown = (recent_df['close'] - peak) / peak
            max_drawdown = drawdown.min() * 100
            
            # 샤프 비율 계산
            returns = recent_df['returns'].dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0
            
            results.append({
                'symbol': symbol,
                'total_return': round(total_return, 2),
                'volatility': round(volatility, 2),
                'avg_volume': int(avg_volume),
                'avg_trading_value': int(avg_trading_value),
                'max_drawdown': round(max_drawdown, 2),
                'sharpe_ratio': round(sharpe_ratio, 2)
            })
            
        except Exception as e:
            print(f"⚠️ {symbol} 분석 실패: {e}")
            continue
    
    return pd.DataFrame(results).sort_values('total_return', ascending=False)

def create_charts(df_analysis):
    """차트 생성"""
    os.makedirs("static/charts", exist_ok=True)
    
    # 1. 수익률 순위 차트
    plt.figure(figsize=(12, 8))
    top_15 = df_analysis.head(15)
    
    colors = ['#e74c3c' if x < 0 else '#27ae60' for x in top_15['total_return']]
    bars = plt.barh(range(len(top_15)), top_15['total_return'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_15)), top_15['symbol'])
    plt.xlabel('수익률 (%)')
    plt.title('코스피 상위 15개 종목 수익률')
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for i, (bar, value) in enumerate(zip(bars, top_15['total_return'])):
        plt.text(bar.get_width() + (0.5 if value > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%', ha='left' if value > 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig('static/charts/returns_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 수익률 vs 변동성 산점도
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_analysis['volatility'], df_analysis['total_return'], 
                         c=df_analysis['sharpe_ratio'], cmap='RdYlGn', alpha=0.7, s=100)
    plt.xlabel('변동성 (%)')
    plt.ylabel('수익률 (%)')
    plt.title('수익률 vs 변동성 (색상: 샤프 비율)')
    plt.colorbar(scatter, label='샤프 비율')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/charts/scatter_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 거래량 상위 10개 종목
    plt.figure(figsize=(12, 6))
    top_volume = df_analysis.nlargest(10, 'avg_trading_value')
    bars = plt.bar(range(len(top_volume)), top_volume['avg_trading_value'] / 1e8, 
                   color='#3498db', alpha=0.7)
    plt.xticks(range(len(top_volume)), top_volume['symbol'], rotation=45)
    plt.ylabel('평균 거래대금 (억원)')
    plt.title('거래대금 상위 10개 종목')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/charts/volume_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """분석 API"""
    try:
        data = request.json
        symbols = data.get('symbols', get_kospi_top30()[:10])  # 기본 10개
        start_date = data.get('start_date', '2024-06-21')
        end_date = data.get('end_date', '2024-09-19')
        
        # 분석 실행
        df_analysis = analyze_stocks(symbols, start_date, end_date)
        
        # 차트 생성
        create_charts(df_analysis)
        
        # 결과 반환
        result = {
            'success': True,
            'data': df_analysis.to_dict('records'),
            'summary': {
                'total_stocks': len(df_analysis),
                'avg_return': round(df_analysis['total_return'].mean(), 2),
                'positive_stocks': len(df_analysis[df_analysis['total_return'] > 0]),
                'negative_stocks': len(df_analysis[df_analysis['total_return'] <= 0]),
                'avg_volatility': round(df_analysis['volatility'].mean(), 2),
                'avg_sharpe': round(df_analysis['sharpe_ratio'].mean(), 2)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/symbols')
def get_symbols():
    """종목 코드 목록 반환"""
    return jsonify({'symbols': get_kospi_top30()})

@app.route('/charts/<filename>')
def get_chart(filename):
    """차트 이미지 반환"""
    return send_file(f'static/charts/{filename}')

if __name__ == '__main__':
    # 템플릿 디렉토리 생성
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/charts', exist_ok=True)
    
    print("🚀 SectorFlow Lite 웹 인터페이스 시작!")
    print("📱 브라우저에서 http://localhost:3000 접속하세요")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=3000)

def create_html_template():
    """HTML 템플릿 생성"""
    html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SectorFlow Lite - 웹 인터페이스</title>
    <style>
        body {
            font-family: 'Malgun Gothic', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .content {
            padding: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #2c3e50;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        .btn {
            background: #3498db;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        .btn:hover {
            background: #2980b9;
        }
        .btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .results {
            margin-top: 30px;
            display: none;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 5px solid #3498db;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        .table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .table th, .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .positive {
            color: #27ae60;
            font-weight: bold;
        }
        .negative {
            color: #e74c3c;
            font-weight: bold;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 SectorFlow Lite</h1>
            <p>AI 기반 주식 분석 시스템 - 웹 인터페이스</p>
        </div>
        
        <div class="content">
            <h2>🔧 분석 설정</h2>
            <form id="analysisForm">
                <div class="form-group">
                    <label for="symbols">분석할 종목 (쉼표로 구분):</label>
                    <input type="text" id="symbols" placeholder="예: 005930,000660,035420" value="005930,000660,035420,005380,006400">
                    <small>종목 코드를 쉼표로 구분하여 입력하세요. 비워두면 상위 10개 종목을 분석합니다.</small>
                </div>
                
                <div class="form-group">
                    <label for="start_date">시작 날짜:</label>
                    <input type="date" id="start_date" value="2024-06-21">
                </div>
                
                <div class="form-group">
                    <label for="end_date">종료 날짜:</label>
                    <input type="date" id="end_date" value="2024-09-19">
                </div>
                
                <button type="submit" class="btn" id="analyzeBtn">📊 분석 실행</button>
            </form>
            
            <div class="loading" id="loading">
                <h3>🔄 분석 중...</h3>
                <p>잠시만 기다려주세요...</p>
            </div>
            
            <div class="results" id="results">
                <h2>📊 분석 결과</h2>
                
                <div class="stats-grid" id="statsGrid">
                    <!-- 통계 카드들이 여기에 동적으로 생성됩니다 -->
                </div>
                
                <div class="chart-container">
                    <h3>수익률 순위 차트</h3>
                    <img id="returnsChart" src="" alt="수익률 차트">
                </div>
                
                <div class="chart-container">
                    <h3>수익률 vs 변동성 산점도</h3>
                    <img id="scatterChart" src="" alt="산점도 차트">
                </div>
                
                <div class="chart-container">
                    <h3>거래대금 상위 종목</h3>
                    <img id="volumeChart" src="" alt="거래량 차트">
                </div>
                
                <h3>📋 상세 결과</h3>
                <table class="table" id="resultsTable">
                    <thead>
                        <tr>
                            <th>순위</th>
                            <th>종목코드</th>
                            <th>수익률</th>
                            <th>변동성</th>
                            <th>샤프비율</th>
                            <th>거래대금</th>
                        </tr>
                    </thead>
                    <tbody id="resultsTableBody">
                        <!-- 결과가 여기에 동적으로 생성됩니다 -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            // UI 상태 변경
            analyzeBtn.disabled = true;
            loading.style.display = 'block';
            results.style.display = 'none';
            
            try {
                // 분석 요청
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symbols: document.getElementById('symbols').value.split(',').map(s => s.trim()).filter(s => s),
                        start_date: document.getElementById('start_date').value,
                        end_date: document.getElementById('end_date').value
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data);
                } else {
                    alert('분석 중 오류가 발생했습니다: ' + data.error);
                }
                
            } catch (error) {
                alert('네트워크 오류가 발생했습니다: ' + error.message);
            } finally {
                analyzeBtn.disabled = false;
                loading.style.display = 'none';
            }
        });
        
        function displayResults(data) {
            const results = document.getElementById('results');
            const statsGrid = document.getElementById('statsGrid');
            const resultsTableBody = document.getElementById('resultsTableBody');
            
            // 통계 카드 생성
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-number">${data.summary.total_stocks}</div>
                    <div class="stat-label">분석 종목</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${data.summary.avg_return}%</div>
                    <div class="stat-label">평균 수익률</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${data.summary.positive_stocks}</div>
                    <div class="stat-label">상승 종목</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${data.summary.avg_volatility}%</div>
                    <div class="stat-label">평균 변동성</div>
                </div>
            `;
            
            // 테이블 생성
            resultsTableBody.innerHTML = data.data.map((row, index) => `
                <tr>
                    <td>${index + 1}</td>
                    <td>${row.symbol}</td>
                    <td class="${row.total_return > 0 ? 'positive' : 'negative'}">${row.total_return}%</td>
                    <td>${row.volatility}%</td>
                    <td>${row.sharpe_ratio}</td>
                    <td>${Math.round(row.avg_trading_value / 100000000)}억원</td>
                </tr>
            `).join('');
            
            // 차트 이미지 업데이트
            document.getElementById('returnsChart').src = '/charts/returns_chart.png?' + Date.now();
            document.getElementById('scatterChart').src = '/charts/scatter_chart.png?' + Date.now();
            document.getElementById('volumeChart').src = '/charts/volume_chart.png?' + Date.now();
            
            results.style.display = 'block';
        }
    </script>
</body>
</html>
    """
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
