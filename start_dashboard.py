#!/usr/bin/env python3
"""
SectorFlow Lite 대시보드 자동 실행 스크립트
"""

import subprocess
import webbrowser
import time
import os
import sys

def start_streamlit():
    """Streamlit 대시보드 시작"""
    print("🚀 SectorFlow Lite 대시보드 시작 중...")
    
    # Streamlit 실행
    try:
        # 백그라운드에서 Streamlit 실행
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "examples/app_streamlit.py",
            "--server.headless", "false"
        ])
        
        print("⏳ Streamlit 서버 시작 대기 중...")
        time.sleep(3)  # 서버 시작 대기
        
        # 브라우저 자동 열기
        url = "http://localhost:8501"
        print(f"🌐 브라우저에서 {url} 열기 중...")
        webbrowser.open(url)
        
        print("✅ 대시보드가 실행되었습니다!")
        print("📊 브라우저에서 대시보드를 확인하세요.")
        print("🛑 종료하려면 Ctrl+C를 누르세요.")
        
        # 프로세스 대기
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 대시보드를 종료합니다...")
            process.terminate()
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    start_streamlit()



