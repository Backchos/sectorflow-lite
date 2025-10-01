#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Railway 배포 스크립트
Railway 배포를 위한 환경 설정 및 검증
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_railway_cli():
    """Railway CLI 설치 확인"""
    try:
        result = subprocess.run(['railway', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Railway CLI 설치됨: {result.stdout.strip()}")
            return True
        else:
            print("❌ Railway CLI가 설치되지 않음")
            return False
    except FileNotFoundError:
        print("❌ Railway CLI를 찾을 수 없음")
        return False

def check_git_repo():
    """Git 저장소 확인"""
    try:
        result = subprocess.run(['git', 'status'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git 저장소 확인됨")
            return True
        else:
            print("❌ Git 저장소가 아님")
            return False
    except FileNotFoundError:
        print("❌ Git이 설치되지 않음")
        return False

def check_required_files():
    """필수 파일 확인"""
    required_files = [
        'railway.json',
        'Procfile', 
        'railway_requirements.txt',
        'examples/app_streamlit.py',
        'config.yaml'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 누락된 파일: {missing_files}")
        return False
    else:
        print("✅ 모든 필수 파일 확인됨")
        return True

def check_environment():
    """환경 변수 확인"""
    env_vars = {
        'PORT': '8501',
        'PYTHON_VERSION': '3.11',
        'STREAMLIT_SERVER_HEADLESS': 'true'
    }
    
    print("🔧 권장 환경 변수:")
    for key, value in env_vars.items():
        print(f"   {key}={value}")
    
    return True

def create_railway_config():
    """Railway 설정 파일 생성"""
    config = {
        "build": {
            "builder": "NIXPACKS"
        },
        "deploy": {
            "startCommand": "streamlit run examples/app_streamlit.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true",
            "healthcheckPath": "/",
            "healthcheckTimeout": 100,
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 10
        }
    }
    
    with open('railway.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print("✅ railway.json 생성됨")

def create_procfile():
    """Procfile 생성"""
    procfile_content = "web: streamlit run examples/app_streamlit.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"
    
    with open('Procfile', 'w', encoding='utf-8') as f:
        f.write(procfile_content)
    
    print("✅ Procfile 생성됨")

def deploy_to_railway():
    """Railway에 배포"""
    print("\n🚀 Railway 배포 시작...")
    
    try:
        # Railway 로그인 확인
        result = subprocess.run(['railway', 'whoami'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Railway에 로그인되지 않음")
            print("다음 명령어로 로그인하세요: railway login")
            return False
        
        # 프로젝트 초기화
        print("📁 프로젝트 초기화 중...")
        subprocess.run(['railway', 'init'], check=True)
        
        # 환경 변수 설정
        print("🔧 환경 변수 설정 중...")
        env_vars = {
            'PORT': '8501',
            'PYTHON_VERSION': '3.11',
            'STREAMLIT_SERVER_HEADLESS': 'true',
            'STREAMLIT_SERVER_ENABLE_CORS': 'false'
        }
        
        for key, value in env_vars.items():
            subprocess.run(['railway', 'variables', 'set', f'{key}={value}'], 
                          check=True)
        
        # 배포 실행
        print("🚀 배포 실행 중...")
        subprocess.run(['railway', 'up'], check=True)
        
        print("✅ 배포 완료!")
        print("🌐 Railway 대시보드에서 URL을 확인하세요: https://railway.app/dashboard")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 배포 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 SectorFlow Lite Railway 배포 도우미")
    print("=" * 50)
    
    # 1. 환경 확인
    print("\n1️⃣ 환경 확인 중...")
    checks = [
        check_git_repo(),
        check_required_files(),
        check_environment()
    ]
    
    if not all(checks):
        print("\n❌ 환경 확인 실패. 문제를 해결한 후 다시 시도하세요.")
        return
    
    # 2. Railway CLI 확인
    print("\n2️⃣ Railway CLI 확인 중...")
    if not check_railway_cli():
        print("\n📥 Railway CLI 설치 방법:")
        print("Windows: iwr https://railway.app/install.ps1 -useb | iex")
        print("macOS/Linux: curl -fsSL https://railway.app/install.sh | sh")
        return
    
    # 3. 설정 파일 생성
    print("\n3️⃣ 설정 파일 생성 중...")
    create_railway_config()
    create_procfile()
    
    # 4. 배포 옵션
    print("\n4️⃣ 배포 옵션:")
    print("1. 자동 배포 (Railway CLI 사용)")
    print("2. 수동 배포 (웹 대시보드 사용)")
    print("3. 설정만 생성하고 종료")
    
    choice = input("\n선택하세요 (1-3): ").strip()
    
    if choice == "1":
        deploy_to_railway()
    elif choice == "2":
        print("\n🌐 웹 대시보드 배포 방법:")
        print("1. https://railway.app/dashboard 접속")
        print("2. 'New Project' → 'Deploy from GitHub repo' 선택")
        print("3. GitHub 저장소 선택")
        print("4. 환경 변수 설정 후 배포")
    elif choice == "3":
        print("\n✅ 설정 파일 생성 완료!")
        print("이제 Railway 웹 대시보드에서 배포하세요.")
    else:
        print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main()
