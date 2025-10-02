#!/usr/bin/env python3
"""
SectorFlow Lite - Simple Flask App for Cloud Run
Google Cloud Run 배포용 간단한 Flask 애플리케이션
"""

import os
from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    """메인 페이지"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>SectorFlow Lite</title>
        <meta charset="utf-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 40px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: rgba(255,255,255,0.1);
                padding: 40px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }
            h1 { 
                color: #fff; 
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .status { 
                background: rgba(255,255,255,0.2); 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0; 
            }
            .feature {
                background: rgba(255,255,255,0.1);
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
            }
            .success {
                color: #4CAF50;
                font-weight: bold;
            }
            .info {
                color: #81C784;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 SectorFlow Lite</h1>
            
            <div class="status">
                <h2>✅ 서비스 상태</h2>
                <p class="success">Flask 애플리케이션이 성공적으로 실행 중입니다!</p>
                <p class="info">Google Cloud Run에서 정상적으로 배포되었습니다.</p>
            </div>
            
            <div class="status">
                <h2>📊 주요 기능</h2>
                <div class="feature">
                    <strong>📈 한국 주식 시장 분석</strong><br>
                    KOSPI 200 종목의 기술적 분석 및 트렌드 예측
                </div>
                <div class="feature">
                    <strong>🤖 AI 기반 투자 전략</strong><br>
                    머신러닝 모델을 활용한 자동화된 투자 신호 생성
                </div>
                <div class="feature">
                    <strong>📋 포트폴리오 관리</strong><br>
                    리스크 관리 및 자산 배분 최적화
                </div>
                <div class="feature">
                    <strong>📊 백테스팅</strong><br>
                    과거 데이터를 활용한 전략 검증 및 성과 분석
                </div>
            </div>
            
            <div class="status">
                <h2>🚀 대시보드 접속</h2>
                <p><strong>Streamlit 대시보드</strong>에서 실제 분석 기능을 사용하세요!</p>
                <a href="/dashboard" style="display: inline-block; background: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; margin: 10px 0;">
                    📊 대시보드 열기
                </a>
                <p style="color: #81C784; font-size: 0.9em;">실시간 차트, AI 예측, 백테스팅 기능을 제공합니다.</p>
            </div>
            
            <div class="status">
                <h2>🔧 기술 스택</h2>
                <p><strong>Backend:</strong> Python, Flask, Pandas, NumPy</p>
                <p><strong>ML:</strong> Scikit-learn, XGBoost</p>
                <p><strong>Cloud:</strong> Google Cloud Run</p>
                <p><strong>Data:</strong> Yahoo Finance API</p>
            </div>
            
            <div class="status">
                <h2>📞 연락처</h2>
                <p>프로젝트 문의: qortls510@gmail.com</p>
                <p>GitHub: <a href="https://github.com/Backchos/sectorflow-lite" style="color: #81C784;">https://github.com/Backchos/sectorflow-lite</a></p>
            </div>
        </div>
    </body>
    </html>
    ''')

@app.route('/health')
def health():
    """헬스 체크 엔드포인트"""
    return {
        'status': 'healthy', 
        'service': 'sectorflow-lite',
        'version': '1.0.0',
        'message': 'SectorFlow Lite is running successfully!'
    }

@app.route('/api/status')
def api_status():
    """API 상태 확인"""
    return {
        'service': 'sectorflow-lite',
        'status': 'running',
        'port': os.environ.get('PORT', 8080),
        'environment': 'production'
    }

@app.route('/dashboard')
def dashboard():
    """통합된 대시보드"""
    import yfinance as yf
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    
    # 기본 설정
    ticker = "005930.KS"  # 삼성전자
    period = "6M"
    
    try:
        # 데이터 로딩
        data = yf.download(ticker, period=period, progress=False)
        
        if data.empty:
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>대시보드</title>
                <meta charset="utf-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                    .error { color: #d32f2f; background: #ffebee; padding: 15px; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📊 SectorFlow Lite 대시보드</h1>
                    <div class="error">
                        <h3>❌ 오류</h3>
                        <p>데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.</p>
                    </div>
                    <a href="/">← 메인 페이지로 돌아가기</a>
                </div>
            </body>
            </html>
            ''')
        
        # 차트 생성
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker
        ))
        
        fig.update_layout(
            title=f"{ticker} 가격 차트",
            xaxis_title="날짜",
            yaxis_title="가격 (원)",
            height=500
        )
        
        # 차트를 HTML로 변환
        chart_html = fig.to_html(include_plotlyjs='cdn', div_id="chart")
        
        # 현재 가격 정보
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>대시보드</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }
                .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .metric { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
                .metric-value { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
                .metric-label { color: #666; }
                .positive { color: #4caf50; }
                .negative { color: #f44336; }
                .chart-container { margin: 30px 0; }
                .nav { margin-bottom: 20px; }
                .nav a { color: #2196f3; text-decoration: none; margin-right: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 SectorFlow Lite 대시보드</h1>
                    <div class="nav">
                        <a href="/">← 메인 페이지</a>
                        <a href="/dashboard">🔄 새로고침</a>
                    </div>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{{ "%.0f"|format(current_price) }}원</div>
                        <div class="metric-label">현재가</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {{ 'positive' if change >= 0 else 'negative' }}">
                            {{ "+" if change >= 0 else "" }}{{ "%.0f"|format(change) }}원
                        </div>
                        <div class="metric-label">변동</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {{ 'positive' if change_pct >= 0 else 'negative' }}">
                            {{ "+" if change_pct >= 0 else "" }}{{ "%.2f"|format(change_pct) }}%
                        </div>
                        <div class="metric-label">변동률</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{ "{:,}".format(data['Volume'].iloc[-1]) }}</div>
                        <div class="metric-label">거래량</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h2>📈 가격 차트</h2>
                    ''' + chart_html + '''
                </div>
                
                <div style="margin-top: 30px; padding: 20px; background: #e3f2fd; border-radius: 8px;">
                    <h3>🎯 분석 모드</h3>
                    <p><strong>기본 분석:</strong> 현재 가격 차트와 기본 지표를 제공합니다.</p>
                    <p><strong>AI 예측:</strong> 머신러닝을 활용한 가격 예측 (개발 중)</p>
                    <p><strong>백테스팅:</strong> 과거 데이터를 활용한 전략 검증 (개발 중)</p>
                    <p><strong>포트폴리오:</strong> 자산 배분 및 리스크 관리 (개발 중)</p>
                </div>
            </div>
        </body>
        </html>
        ''', 
        current_price=current_price,
        change=change,
        change_pct=change_pct,
        data=data
        )
        
    except Exception as e:
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>대시보드</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                .error { color: #d32f2f; background: #ffebee; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 SectorFlow Lite 대시보드</h1>
                <div class="error">
                    <h3>❌ 오류</h3>
                    <p>오류가 발생했습니다: {{ error }}</p>
                </div>
                <a href="/">← 메인 페이지로 돌아가기</a>
            </div>
        </body>
        </html>
        ''', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)