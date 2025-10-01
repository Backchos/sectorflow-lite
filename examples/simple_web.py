#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 웹 서버 테스트
"""

from flask import Flask, render_template_string
import webbrowser
import threading
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SectorFlow Lite - 테스트</title>
    <style>
        body {
            font-family: 'Malgun Gothic', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 30px;
            text-align: center;
        }
        .header {
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .btn {
            background: #3498db;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            margin: 10px;
        }
        .btn:hover {
            background: #2980b9;
        }
        .status {
            margin: 20px 0;
            padding: 15px;
            background: #d5f4e6;
            border-radius: 5px;
            color: #27ae60;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SectorFlow Lite</h1>
            <p>AI 기반 주식 분석 시스템</p>
        </div>
        
        <div class="status">
            ✅ 웹 서버가 정상적으로 실행되었습니다!
        </div>
        
        <h2>🎯 사용 가능한 기능</h2>
        <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
            <li>📊 실시간 주식 분석</li>
            <li>📈 수익률 시각화</li>
            <li>📋 상세 데이터 테이블</li>
            <li>🎨 한글 완벽 지원</li>
        </ul>
        
        <div style="margin-top: 30px;">
            <button class="btn" onclick="alert('웹 서버가 정상 작동 중입니다!')">
                🧪 연결 테스트
            </button>
        </div>
        
        <div style="margin-top: 20px; color: #7f8c8d;">
            <p>포트: 8080</p>
            <p>주소: http://localhost:8080</p>
        </div>
    </div>
</body>
</html>
    ''')

if __name__ == '__main__':
    print("🚀 간단한 웹 서버 시작!")
    print("📱 브라우저에서 http://localhost:8080 접속하세요")
    print("=" * 50)
    
    # 자동으로 브라우저 열기
    def open_browser():
        time.sleep(1)
        webbrowser.open('http://localhost:8080')
    
    threading.Thread(target=open_browser).start()
    
    app.run(debug=True, host='0.0.0.0', port=8080)
