#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - 투자일지 CLI 도구
명령줄에서 투자일지를 작성하고 관리하는 도구
"""

import sys
import os
import argparse
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.investment_log import InvestmentLogger

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='SectorFlow Lite - 투자일지 CLI 도구')
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # 투자일지 작성
    write_parser = subparsers.add_parser('write', help='투자일지 작성')
    write_parser.add_argument('--symbol', required=True, help='종목코드 (예: 005930)')
    write_parser.add_argument('--action', required=True, choices=['매수', '매도'], help='거래유형')
    write_parser.add_argument('--price', type=float, required=True, help='가격 (원)')
    write_parser.add_argument('--quantity', type=int, required=True, help='수량 (주)')
    write_parser.add_argument('--reason', default='', help='투자 이유')
    
    # 투자일지 읽기
    read_parser = subparsers.add_parser('read', help='투자일지 읽기')
    read_parser.add_argument('--date', help='날짜 (YYYY-MM-DD, 기본값: 오늘)')
    read_parser.add_argument('--all', action='store_true', help='모든 투자일지 읽기')
    
    # 데이터베이스 업로드
    upload_parser = subparsers.add_parser('upload', help='데이터베이스 업로드')
    upload_parser.add_argument('--date', help='날짜 (YYYY-MM-DD, 기본값: 오늘)')
    
    # 요약 정보
    summary_parser = subparsers.add_parser('summary', help='투자일지 요약')
    
    # 엑셀 내보내기
    export_parser = subparsers.add_parser('export', help='엑셀 파일 내보내기')
    export_parser.add_argument('--output', help='출력 파일명')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 투자일지 로거 초기화
    logger = InvestmentLogger()
    
    try:
        if args.command == 'write':
            # 투자일지 작성
            result = logger.write_log(
                symbol=args.symbol,
                action=args.action,
                price=args.price,
                quantity=args.quantity,
                reason=args.reason
            )
            print(f"✅ {result}")
            
        elif args.command == 'read':
            # 투자일지 읽기
            if args.all:
                logs = logger.read_all_logs()
                print(f"📋 전체 투자일지 ({len(logs)}개)")
            else:
                logs = logger.read_daily_log(args.date)
                date_str = args.date or datetime.now().strftime("%Y-%m-%d")
                print(f"📅 {date_str} 투자일지 ({len(logs)}개)")
            
            if not logs:
                print("투자일지가 없습니다.")
                return
            
            # 투자일지 출력
            print("\n" + "="*80)
            print(f"{'날짜':<12} {'종목':<8} {'거래':<4} {'가격':<12} {'수량':<8} {'이유'}")
            print("="*80)
            
            for log in logs:
                action_color = "🟢" if log['action'] == '매수' else "🔴"
                print(f"{log['date']:<12} {log['symbol']:<8} {action_color}{log['action']:<3} {log['price']:>10,.0f}원 {log['quantity']:>6,}주 {log['reason']}")
            
        elif args.command == 'upload':
            # 데이터베이스 업로드
            result = logger.upload_to_database(args.date)
            print(f"✅ {result}")
            
        elif args.command == 'summary':
            # 요약 정보
            summary = logger.get_database_summary()
            
            print("📈 투자일지 요약")
            print("="*50)
            print(f"총 거래 수: {summary['total_trades']}회")
            print(f"총 투자금액: {summary['total_investment']:,.0f}원")
            
            print("\n거래 유형별:")
            for action, count in summary['action_counts'].items():
                action_emoji = "🟢" if action == '매수' else "🔴"
                print(f"  {action_emoji} {action}: {count}회")
            
            print("\n종목별 거래:")
            for symbol, count in summary['symbol_counts'].items():
                print(f"  {symbol}: {count}회")
                
        elif args.command == 'export':
            # 엑셀 내보내기
            output_file = logger.export_to_excel(args.output)
            print(f"✅ 엑셀 파일 내보내기 완료: {output_file}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


