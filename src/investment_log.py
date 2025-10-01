#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - 투자일지 관리 시스템
텍스트 파일 기반 투자일지 작성 및 데이터베이스 업로드 기능
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

class InvestmentLogger:
    """투자일지 관리 클래스"""
    
    def __init__(self, log_dir: str = "investment_logs"):
        """투자일지 초기화"""
        self.log_dir = log_dir
        self.db_path = os.path.join(log_dir, "investment_log.db")
        
        # 로그 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        
        # 데이터베이스 테이블 생성
        self._create_database()
    
    def _create_database(self):
        """데이터베이스 테이블 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS investment_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def write_log(self, symbol: str, action: str, price: float, 
                  quantity: int, reason: str = "") -> str:
        """투자일지 작성"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 텍스트 파일에 기록
        log_entry = {
            "date": today,
            "symbol": symbol,
            "action": action,
            "price": price,
            "quantity": quantity,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        # 일별 로그 파일에 추가
        daily_log_file = os.path.join(self.log_dir, f"investment_log_{today}.txt")
        
        with open(daily_log_file, "a", encoding="utf-8") as f:
            f.write(f"{today} | {symbol} | {action} | {price:,.0f} | {quantity:,} | {reason}\n")
        
        # JSON 형태로도 저장 (구조화된 데이터)
        json_log_file = os.path.join(self.log_dir, f"investment_log_{today}.json")
        
        # 기존 JSON 파일 로드
        if os.path.exists(json_log_file):
            with open(json_log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        # JSON 파일 저장
        with open(json_log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        
        return f"투자일지 작성 완료: {symbol} {action} {price:,.0f}원 {quantity:,}주"
    
    def read_daily_log(self, date: str = None) -> List[Dict]:
        """일별 투자일지 읽기"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        json_log_file = os.path.join(self.log_dir, f"investment_log_{date}.json")
        
        if os.path.exists(json_log_file):
            with open(json_log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return []
    
    def read_all_logs(self) -> List[Dict]:
        """모든 투자일지 읽기"""
        all_logs = []
        
        for filename in os.listdir(self.log_dir):
            if filename.startswith("investment_log_") and filename.endswith(".json"):
                file_path = os.path.join(self.log_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    daily_logs = json.load(f)
                    all_logs.extend(daily_logs)
        
        # 날짜순으로 정렬
        all_logs.sort(key=lambda x: x["date"])
        return all_logs
    
    def upload_to_database(self, date: str = None) -> str:
        """텍스트 파일을 데이터베이스로 업로드"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        json_log_file = os.path.join(self.log_dir, f"investment_log_{date}.json")
        
        if not os.path.exists(json_log_file):
            return f"해당 날짜({date})의 투자일지가 없습니다."
        
        # JSON 파일 로드
        with open(json_log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
        
        # 데이터베이스에 업로드
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        uploaded_count = 0
        for log in logs:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO investment_log 
                    (date, symbol, action, price, quantity, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    log["date"],
                    log["symbol"],
                    log["action"],
                    log["price"],
                    log["quantity"],
                    log["reason"]
                ))
                uploaded_count += 1
            except Exception as e:
                print(f"업로드 실패: {log} - {e}")
        
        conn.commit()
        conn.close()
        
        return f"데이터베이스 업로드 완료: {uploaded_count}개 항목"
    
    def get_database_summary(self) -> Dict:
        """데이터베이스 요약 정보"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 총 거래 수
        cursor.execute("SELECT COUNT(*) FROM investment_log")
        total_trades = cursor.fetchone()[0]
        
        # 매수/매도 수
        cursor.execute("SELECT action, COUNT(*) FROM investment_log GROUP BY action")
        action_counts = dict(cursor.fetchall())
        
        # 종목별 거래 수
        cursor.execute("SELECT symbol, COUNT(*) FROM investment_log GROUP BY symbol ORDER BY COUNT(*) DESC")
        symbol_counts = dict(cursor.fetchall())
        
        # 총 투자금액 (매수 기준)
        cursor.execute("SELECT SUM(price * quantity) FROM investment_log WHERE action = '매수'")
        total_investment = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_trades": total_trades,
            "action_counts": action_counts,
            "symbol_counts": symbol_counts,
            "total_investment": total_investment
        }
    
    def export_to_excel(self, output_file: str = None) -> str:
        """투자일지를 엑셀 파일로 내보내기"""
        if output_file is None:
            output_file = os.path.join(self.log_dir, f"investment_log_export_{datetime.now().strftime('%Y%m%d')}.xlsx")
        
        # 데이터베이스에서 데이터 읽기
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM investment_log ORDER BY date DESC", conn)
        conn.close()
        
        # 엑셀 파일로 저장
        df.to_excel(output_file, index=False, engine='openpyxl')
        
        return f"엑셀 파일 내보내기 완료: {output_file}"

def main():
    """투자일지 테스트"""
    logger = InvestmentLogger()
    
    # 테스트 데이터 작성
    print("=== 투자일지 테스트 ===")
    
    # 투자일지 작성
    logger.write_log("005930", "매수", 75000, 10, "삼성전자 매수 - AI 분석 결과 양호")
    logger.write_log("000660", "매수", 120000, 5, "SK하이닉스 매수 - 반도체 업황 개선")
    logger.write_log("035420", "매수", 200000, 3, "NAVER 매수 - 플랫폼 성장 기대")
    
    # 일별 로그 읽기
    print("\n=== 오늘 투자일지 ===")
    today_logs = logger.read_daily_log()
    for log in today_logs:
        print(f"{log['date']} | {log['symbol']} | {log['action']} | {log['price']:,}원 | {log['quantity']:,}주 | {log['reason']}")
    
    # 데이터베이스 업로드
    print("\n=== 데이터베이스 업로드 ===")
    upload_result = logger.upload_to_database()
    print(upload_result)
    
    # 데이터베이스 요약
    print("\n=== 투자일지 요약 ===")
    summary = logger.get_database_summary()
    print(f"총 거래 수: {summary['total_trades']}")
    print(f"거래 유형: {summary['action_counts']}")
    print(f"종목별 거래: {summary['symbol_counts']}")
    print(f"총 투자금액: {summary['total_investment']:,}원")

if __name__ == "__main__":
    main()
