#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - 시작 분석 도구
가장 간단한 방법으로 분석 실행
"""

import os
import sys
import webbrowser
from datetime import datetime

def main():
    """메인 함수"""
    print("=" * 60)
    print("🚀 SectorFlow Lite - 주식 분석 도구")
    print("=" * 60)
    print()
    
    # 현재 디렉토리 확인
    current_dir = os.getcwd()
    print(f"📁 현재 위치: {current_dir}")
    
    # 파일 존재 확인
    files_to_check = [
        "simple_analysis.py",
        "data/raw/005930.csv",
        "run_analysis.bat"
    ]
    
    print("\n🔍 파일 확인:")
    for file in files_to_check:
        if os.path.exists(file):
            print(f"  ✅ {file} - 존재함")
        else:
            print(f"  ❌ {file} - 없음")
    
    print("\n🎯 분석 방법 선택:")
    print("1. 자동 분석 실행 (추천)")
    print("2. 웹 브라우저에서 결과 보기")
    print("3. 파일 탐색기 열기")
    print("4. 종료")
    
    try:
        choice = input("\n선택하세요 (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 자동 분석을 시작합니다...")
            os.system("python simple_analysis.py")
            
        elif choice == "2":
            print("\n🌐 웹 브라우저를 엽니다...")
            # 생성된 HTML 파일 찾기
            html_files = [f for f in os.listdir('.') if f.startswith('kospi30_result_') and f.endswith('.html')]
            if html_files:
                latest_file = max(html_files, key=os.path.getctime)
                file_path = os.path.abspath(latest_file)
                print(f"📄 파일 열기: {latest_file}")
                webbrowser.open(f"file:///{file_path}")
            else:
                print("❌ HTML 결과 파일이 없습니다. 먼저 분석을 실행하세요.")
                
        elif choice == "3":
            print("\n📁 파일 탐색기를 엽니다...")
            os.system("explorer .")
            
        elif choice == "4":
            print("\n👋 종료합니다.")
            return
            
        else:
            print("❌ 잘못된 선택입니다.")
            
    except KeyboardInterrupt:
        print("\n\n👋 사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    
    print("\n⏸️ 아무 키나 누르면 종료됩니다...")
    input()

if __name__ == "__main__":
    main()

