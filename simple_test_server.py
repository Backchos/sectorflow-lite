#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 테스트 서버
"""

from flask import Flask, jsonify
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.investment_log import InvestmentLogger

app = Flask(__name__)
logger = InvestmentLogger()

@app.route('/')
def index():
    return '''
    <h1>투자일지 테스트 서버</h1>
    <p>서버가 정상 작동합니다!</p>
    <p><a href="/test">테스트</a></p>
    '''

@app.route('/test')
def test():
    return jsonify({
        'status': 'success',
        'message': '투자일지 시스템이 정상 작동합니다!',
        'logs': logger.read_daily_log()
    })

if __name__ == '__main__':
    print("🚀 간단한 테스트 서버 시작!")
    print("📱 브라우저에서 http://localhost:3000 접속하세요")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='127.0.0.1', port=3000, use_reloader=False)
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        print("포트 3000이 사용 중일 수 있습니다. 다른 포트를 시도해보세요.")

