#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - 투자일지 웹 인터페이스
Flask 기반 투자일지 작성 및 관리 웹앱
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
from datetime import datetime
import json

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.investment_log import InvestmentLogger

app = Flask(__name__)

# 투자일지 로거 초기화
logger = InvestmentLogger()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('investment_log.html')

@app.route('/api/write_log', methods=['POST'])
def write_log():
    """투자일지 작성 API"""
    try:
        data = request.get_json()
        
        symbol = data.get('symbol', '').strip()
        action = data.get('action', '').strip()
        price = float(data.get('price', 0))
        quantity = int(data.get('quantity', 0))
        reason = data.get('reason', '').strip()
        
        if not all([symbol, action, price, quantity]):
            return jsonify({'success': False, 'error': '필수 항목을 모두 입력해주세요.'})
        
        if action not in ['매수', '매도']:
            return jsonify({'success': False, 'error': '거래 유형은 매수 또는 매도만 가능합니다.'})
        
        result = logger.write_log(symbol, action, price, quantity, reason)
        
        return jsonify({'success': True, 'message': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/read_logs')
def read_logs():
    """투자일지 읽기 API"""
    try:
        date = request.args.get('date')
        logs = logger.read_daily_log(date)
        
        return jsonify({'success': True, 'logs': logs})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/read_all_logs')
def read_all_logs():
    """모든 투자일지 읽기 API"""
    try:
        logs = logger.read_all_logs()
        
        return jsonify({'success': True, 'logs': logs})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload_to_db', methods=['POST'])
def upload_to_db():
    """데이터베이스 업로드 API"""
    try:
        data = request.get_json()
        date = data.get('date')
        
        result = logger.upload_to_database(date)
        
        return jsonify({'success': True, 'message': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/summary')
def get_summary():
    """투자일지 요약 API"""
    try:
        summary = logger.get_database_summary()
        
        return jsonify({'success': True, 'summary': summary})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export_excel')
def export_excel():
    """엑셀 파일 내보내기 API"""
    try:
        output_file = logger.export_to_excel()
        
        return send_file(output_file, as_attachment=True, 
                        download_name=f"investment_log_{datetime.now().strftime('%Y%m%d')}.xlsx")
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def create_html_template():
    """HTML 템플릿 생성"""
    html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SectorFlow Lite - 투자일지</title>
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
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            box-sizing: border-box;
        }
        .form-row {
            display: flex;
            gap: 15px;
        }
        .form-row .form-group {
            flex: 1;
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
            margin-right: 10px;
        }
        .btn:hover {
            background: #2980b9;
        }
        .btn-success {
            background: #27ae60;
        }
        .btn-success:hover {
            background: #229954;
        }
        .btn-warning {
            background: #f39c12;
        }
        .btn-warning:hover {
            background: #e67e22;
        }
        .btn-danger {
            background: #e74c3c;
        }
        .btn-danger:hover {
            background: #c0392b;
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
        .table tr:hover {
            background-color: #f5f5f5;
        }
        .positive {
            color: #27ae60;
            font-weight: bold;
        }
        .negative {
            color: #e74c3c;
            font-weight: bold;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .alert {
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .alert-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .alert-error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 5px solid #3498db;
        }
        .summary-number {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .summary-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 SectorFlow Lite - 투자일지</h1>
            <p>AI 기반 투자 전략 백테스팅 플랫폼</p>
        </div>
        
        <div class="content">
            <!-- 투자일지 작성 폼 -->
            <h2>📝 투자일지 작성</h2>
            <form id="logForm">
                <div class="form-row">
                    <div class="form-group">
                        <label for="symbol">종목코드:</label>
                        <input type="text" id="symbol" placeholder="예: 005930" required>
                    </div>
                    <div class="form-group">
                        <label for="action">거래유형:</label>
                        <select id="action" required>
                            <option value="">선택하세요</option>
                            <option value="매수">매수</option>
                            <option value="매도">매도</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="price">가격 (원):</label>
                        <input type="number" id="price" placeholder="예: 75000" required>
                    </div>
                    <div class="form-group">
                        <label for="quantity">수량 (주):</label>
                        <input type="number" id="quantity" placeholder="예: 10" required>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="reason">투자 이유:</label>
                    <textarea id="reason" rows="3" placeholder="투자 이유를 입력하세요..."></textarea>
                </div>
                
                <button type="submit" class="btn">📝 투자일지 작성</button>
            </form>
            
            <!-- 액션 버튼들 -->
            <h2>🔧 관리 기능</h2>
            <div style="margin: 20px 0;">
                <button class="btn btn-success" onclick="loadTodayLogs()">📅 오늘 투자일지</button>
                <button class="btn btn-warning" onclick="loadAllLogs()">📋 전체 투자일지</button>
                <button class="btn" onclick="uploadToDB()">💾 DB 업로드</button>
                <button class="btn btn-danger" onclick="exportExcel()">📊 엑셀 내보내기</button>
                <button class="btn" onclick="loadSummary()">📈 요약 정보</button>
            </div>
            
            <!-- 로딩 표시 -->
            <div class="loading" id="loading">
                <h3>🔄 처리 중...</h3>
                <p>잠시만 기다려주세요...</p>
            </div>
            
            <!-- 결과 표시 -->
            <div id="results"></div>
        </div>
    </div>

    <script>
        // 투자일지 작성 폼 제출
        document.getElementById('logForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = {
                symbol: document.getElementById('symbol').value,
                action: document.getElementById('action').value,
                price: parseFloat(document.getElementById('price').value),
                quantity: parseInt(document.getElementById('quantity').value),
                reason: document.getElementById('reason').value
            };
            
            showLoading(true);
            
            try {
                const response = await fetch('/api/write_log', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showAlert(data.message, 'success');
                    document.getElementById('logForm').reset();
                } else {
                    showAlert(data.error, 'error');
                }
                
            } catch (error) {
                showAlert('네트워크 오류: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        });
        
        // 오늘 투자일지 로드
        async function loadTodayLogs() {
            showLoading(true);
            
            try {
                const response = await fetch('/api/read_logs');
                const data = await response.json();
                
                if (data.success) {
                    displayLogs(data.logs, '오늘 투자일지');
                } else {
                    showAlert(data.error, 'error');
                }
                
            } catch (error) {
                showAlert('네트워크 오류: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        // 전체 투자일지 로드
        async function loadAllLogs() {
            showLoading(true);
            
            try {
                const response = await fetch('/api/read_all_logs');
                const data = await response.json();
                
                if (data.success) {
                    displayLogs(data.logs, '전체 투자일지');
                } else {
                    showAlert(data.error, 'error');
                }
                
            } catch (error) {
                showAlert('네트워크 오류: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        // 데이터베이스 업로드
        async function uploadToDB() {
            showLoading(true);
            
            try {
                const response = await fetch('/api/upload_to_db', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showAlert(data.message, 'success');
                } else {
                    showAlert(data.error, 'error');
                }
                
            } catch (error) {
                showAlert('네트워크 오류: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        // 엑셀 내보내기
        async function exportExcel() {
            try {
                const response = await fetch('/api/export_excel');
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'investment_log.xlsx';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                    
                    showAlert('엑셀 파일 다운로드 완료!', 'success');
                } else {
                    showAlert('엑셀 내보내기 실패', 'error');
                }
                
            } catch (error) {
                showAlert('네트워크 오류: ' + error.message, 'error');
            }
        }
        
        // 요약 정보 로드
        async function loadSummary() {
            showLoading(true);
            
            try {
                const response = await fetch('/api/summary');
                const data = await response.json();
                
                if (data.success) {
                    displaySummary(data.summary);
                } else {
                    showAlert(data.error, 'error');
                }
                
            } catch (error) {
                showAlert('네트워크 오류: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        // 투자일지 표시
        function displayLogs(logs, title) {
            const results = document.getElementById('results');
            
            if (logs.length === 0) {
                results.innerHTML = `<h3>${title}</h3><p>투자일지가 없습니다.</p>`;
                return;
            }
            
            let html = `<h3>${title} (${logs.length}개)</h3>`;
            html += '<table class="table">';
            html += '<thead><tr><th>날짜</th><th>종목</th><th>거래</th><th>가격</th><th>수량</th><th>이유</th></tr></thead>';
            html += '<tbody>';
            
            logs.forEach(log => {
                const actionClass = log.action === '매수' ? 'positive' : 'negative';
                html += `
                    <tr>
                        <td>${log.date}</td>
                        <td>${log.symbol}</td>
                        <td class="${actionClass}">${log.action}</td>
                        <td>${log.price.toLocaleString()}원</td>
                        <td>${log.quantity.toLocaleString()}주</td>
                        <td>${log.reason}</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            results.innerHTML = html;
        }
        
        // 요약 정보 표시
        function displaySummary(summary) {
            const results = document.getElementById('results');
            
            let html = '<h3>📈 투자일지 요약</h3>';
            html += '<div class="summary-cards">';
            html += `<div class="summary-card"><div class="summary-number">${summary.total_trades}</div><div class="summary-label">총 거래 수</div></div>`;
            html += `<div class="summary-card"><div class="summary-number">${summary.total_investment.toLocaleString()}</div><div class="summary-label">총 투자금액 (원)</div></div>`;
            html += '</div>';
            
            html += '<h4>거래 유형별</h4>';
            html += '<table class="table">';
            html += '<thead><tr><th>거래 유형</th><th>횟수</th></tr></thead>';
            html += '<tbody>';
            
            Object.entries(summary.action_counts).forEach(([action, count]) => {
                const actionClass = action === '매수' ? 'positive' : 'negative';
                html += `<tr><td class="${actionClass}">${action}</td><td>${count}회</td></tr>`;
            });
            
            html += '</tbody></table>';
            
            html += '<h4>종목별 거래</h4>';
            html += '<table class="table">';
            html += '<thead><tr><th>종목</th><th>거래 횟수</th></tr></thead>';
            html += '<tbody>';
            
            Object.entries(summary.symbol_counts).forEach(([symbol, count]) => {
                html += `<tr><td>${symbol}</td><td>${count}회</td></tr>`;
            });
            
            html += '</tbody></table>';
            
            results.innerHTML = html;
        }
        
        // 로딩 표시
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
        
        // 알림 표시
        function showAlert(message, type) {
            const results = document.getElementById('results');
            const alertClass = type === 'success' ? 'alert-success' : 'alert-error';
            results.innerHTML = `<div class="alert ${alertClass}">${message}</div>`;
        }
    </script>
</body>
</html>
    """
    
    # 템플릿 디렉토리 생성
    os.makedirs('templates', exist_ok=True)
    
    with open('templates/investment_log.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    # HTML 템플릿 생성
    create_html_template()
    
    print("🚀 SectorFlow Lite 투자일지 웹 인터페이스 시작!")
    print("📱 브라우저에서 http://localhost:3000 접속하세요")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=3000)
