#!/usr/bin/env python3
"""
SectorFlow Lite - Main Application Entry Point
Railway 배포용 메인 애플리케이션
"""

import os
import sys
import subprocess

def main():
    """메인 실행 함수"""
    # Streamlit 실행
    port = os.environ.get('PORT', '8501')
    
    cmd = [
        'python', '-m', 'streamlit', 'run', 
        'examples/app_streamlit.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true'
    ]
    
    print(f"🚀 Starting SectorFlow Lite on port {port}")
    print(f"Command: {' '.join(cmd)}")
    
    # Streamlit 실행
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
