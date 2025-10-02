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
    """Streamlit 대시보드 리다이렉트"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Streamlit Dashboard</title>
        <meta charset="utf-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 40px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container { 
                text-align: center;
                background: rgba(255,255,255,0.1);
                padding: 40px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }
            h1 { color: #fff; margin-bottom: 30px; }
            .loading { color: #81C784; margin: 20px 0; }
            .info { color: #B0BEC5; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Streamlit 대시보드</h1>
            <p class="loading">대시보드를 시작하는 중...</p>
            <p class="info">잠시 후 자동으로 리다이렉트됩니다.</p>
            <script>
                // Streamlit 대시보드로 리다이렉트
                setTimeout(function() {
                    window.location.href = '/streamlit';
                }, 2000);
            </script>
        </div>
    </body>
    </html>
    ''')

@app.route('/streamlit')
def streamlit_dashboard():
    """Streamlit 대시보드 실행"""
    import subprocess
    import threading
    
    def run_streamlit():
        subprocess.run([
            'python', '-m', 'streamlit', 'run', 
            'dashboard.py',
            '--server.port=8501',
            '--server.address=0.0.0.0',
            '--server.headless=true',
            '--server.enableCORS=false',
            '--server.enableXsrfProtection=false'
        ])
    
    # Streamlit을 백그라운드에서 시작
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Streamlit Dashboard</title>
        <meta charset="utf-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 40px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container { 
                text-align: center;
                background: rgba(255,255,255,0.1);
                padding: 40px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }
            h1 { color: #fff; margin-bottom: 30px; }
            .loading { color: #81C784; margin: 20px 0; }
            .info { color: #B0BEC5; margin: 10px 0; }
            .link { 
                display: inline-block; 
                background: #4CAF50; 
                color: white; 
                padding: 15px 30px; 
                text-decoration: none; 
                border-radius: 8px; 
                font-weight: bold; 
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Streamlit 대시보드</h1>
            <p class="loading">대시보드가 시작되었습니다!</p>
            <p class="info">아래 버튼을 클릭하여 대시보드에 접속하세요.</p>
            <a href="http://localhost:8501" target="_blank" class="link">
                📊 대시보드 열기
            </a>
            <p class="info">새 창에서 대시보드가 열립니다.</p>
        </div>
    </body>
    </html>
    ''')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)