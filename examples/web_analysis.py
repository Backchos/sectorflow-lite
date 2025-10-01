#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - 웹 기반 주식 분석 도구
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 백엔드 사용
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
import warnings
import io
import base64
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

def load_stock_data(symbols):
    """주식 데이터 로드"""
    data_dir = "data/raw"
    available_data = {}
    
    for symbol in symbols:
        file_path = f"{data_dir}/{symbol}.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df['date'] = pd.to_datetime(df['date'])
                available_data[symbol] = df
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
    
    return available_data

def analyze_stocks(data):
    """주식 분석"""
    results = []
    
    for symbol, df in data.items():
        if df.empty:
            continue
        
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

def create_chart(data, title):
    """차트 생성"""
    plt.figure(figsize=(12, 6))
    
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
    
    # 이미지를 base64로 변환
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('analysis.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """분석 실행"""
    try:
        data = request.get_json()
        selected_symbols = data.get('symbols', [])
        analysis_type = data.get('type', 'individual')
        
        if analysis_type == 'kospi30':
            # KOSPI 상위 30개 종목
            symbols = [
                "005930", "000660", "035720", "051910", "259960",
                "003550", "180640", "034730", "068270", "207940",
                "066570", "323410", "000270", "161890", "032830",
                "000810", "017670", "006400", "000720", "105560",
                "012330", "003670", "015760", "018260", "086280",
                "003490", "024110", "000990", "011200", "128940"
            ]
        else:
            symbols = selected_symbols
        
        # 데이터 로드
        stock_data = load_stock_data(symbols)
        
        if not stock_data:
            return jsonify({'error': '사용 가능한 데이터가 없습니다.'})
        
        # 분석 실행
        results = analyze_stocks(stock_data)
        
        # 차트 생성
        chart_image = create_chart(stock_data, f"{len(results)}개 종목 분석 결과")
        
        # 통계 계산
        total_return = np.mean([r['total_return'] for r in results])
        positive_count = sum(1 for r in results if r['total_return'] > 0)
        avg_volatility = np.mean([r['volatility'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        
        return jsonify({
            'success': True,
            'results': results,
            'chart': chart_image,
            'stats': {
                'total_stocks': len(results),
                'avg_return': total_return,
                'positive_count': positive_count,
                'avg_volatility': avg_volatility,
                'avg_sharpe': avg_sharpe
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/symbols')
def get_symbols():
    """사용 가능한 종목 목록 반환"""
    data_dir = "data/raw"
    symbols = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                symbol = file.replace('.csv', '')
                symbols.append(symbol)
    
    return jsonify({'symbols': symbols})

if __name__ == '__main__':
    # templates 폴더가 없으면 생성
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("🌐 SectorFlow Lite 웹 서버 시작 중...")
    print("📱 브라우저에서 http://localhost:3000 접속하세요")
    app.run(debug=True, host='127.0.0.1', port=3000)
