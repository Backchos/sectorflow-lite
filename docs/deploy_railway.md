# 🚀 SectorFlow Lite Railway 배포 가이드

이 문서는 SectorFlow Lite를 Railway에 배포하는 방법을 설명합니다.

## 📋 사전 요구사항

- Railway 계정 (https://railway.app)
- GitHub 저장소 (코드 업로드용)
- 도메인 (선택사항)

## 🚀 1. Railway 계정 설정

### 1.1 Railway 가입
1. https://railway.app 접속
2. GitHub 계정으로 로그인
3. 이메일 인증 완료

### 1.2 Railway CLI 설치 (선택사항)
```bash
# Windows (PowerShell)
iwr https://railway.app/install.ps1 -useb | iex

# macOS
curl -fsSL https://railway.app/install.sh | sh

# Linux
curl -fsSL https://railway.app/install.sh | sh
```

## 🔧 2. 프로젝트 준비

### 2.1 GitHub 저장소 생성
```bash
# 로컬에서 Git 초기화
git init
git add .
git commit -m "Initial commit: SectorFlow Lite"

# GitHub에 저장소 생성 후 연결
git remote add origin https://github.com/YOUR_USERNAME/sectorflow-lite.git
git push -u origin main
```

### 2.2 Railway 설정 파일 확인
프로젝트에 다음 파일들이 있는지 확인:
- `railway.json` ✅
- `Procfile` ✅
- `railway_requirements.txt` ✅

## 🌐 3. Railway 배포

### 3.1 웹 대시보드로 배포 (추천)

1. **Railway 대시보드 접속**
   - https://railway.app/dashboard

2. **새 프로젝트 생성**
   - "New Project" 클릭
   - "Deploy from GitHub repo" 선택
   - GitHub 저장소 선택

3. **환경 변수 설정**
   ```
   PORT=8501
   PYTHON_VERSION=3.11
   STREAMLIT_SERVER_HEADLESS=true
   STREAMLIT_SERVER_ENABLE_CORS=false
   ```

4. **배포 시작**
   - "Deploy" 버튼 클릭
   - 빌드 과정 모니터링 (5-10분 소요)

### 3.2 CLI로 배포 (고급 사용자)

```bash
# Railway 로그인
railway login

# 프로젝트 초기화
railway init

# 환경 변수 설정
railway variables set PORT=8501
railway variables set PYTHON_VERSION=3.11

# 배포
railway up
```

## 🔍 4. 배포 확인

### 4.1 서비스 상태 확인
- Railway 대시보드에서 "Deployments" 탭 확인
- 빌드 로그에서 오류 확인
- "View Logs"로 실시간 로그 모니터링

### 4.2 웹 접속 테스트
- Railway에서 제공하는 URL로 접속
- 예: `https://sectorflow-lite-production.up.railway.app`

### 4.3 기능 테스트
- 대시보드 로딩 확인
- 종목 선택 기능 테스트
- 차트 생성 확인

## ⚙️ 5. 환경 변수 설정

### 5.1 필수 환경 변수
```bash
# 포트 설정
PORT=8501

# Python 버전
PYTHON_VERSION=3.11

# Streamlit 설정
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_PORT=8501
```

### 5.2 선택적 환경 변수
```bash
# 데이터 API 키 (실시간 데이터용)
KRX_API_KEY=your_api_key
YAHOO_FINANCE_ENABLED=true

# 모니터링
SENTRY_DSN=your_sentry_dsn
```

## 🔄 6. 자동 배포 설정

### 6.1 GitHub 연동
- Railway에서 GitHub 저장소 연결 시 자동 배포 활성화
- `main` 브랜치에 push할 때마다 자동 배포

### 6.2 수동 배포
```bash
# 코드 변경 후
git add .
git commit -m "Update: 실시간 데이터 연동"
git push origin main

# Railway에서 자동으로 배포 시작
```

## 📊 7. 모니터링 및 로그

### 7.1 로그 확인
- Railway 대시보드 → "Deployments" → "View Logs"
- 실시간 로그 스트리밍
- 오류 및 경고 메시지 확인

### 7.2 메트릭 모니터링
- CPU 사용률
- 메모리 사용량
- 네트워크 트래픽
- 응답 시간

## 💰 8. 비용 관리

### 8.1 무료 티어
- **월 $5 크레딧**
- **512MB RAM**
- **1GB 디스크**
- **월 100GB 대역폭**

### 8.2 유료 플랜 (필요시)
- **Hobby**: $5/월
- **Pro**: $20/월
- **Team**: $99/월

### 8.3 비용 최적화 팁
```bash
# 불필요한 패키지 제거
pip freeze > current_requirements.txt
# railway_requirements.txt와 비교하여 최적화

# 메모리 사용량 모니터링
# 대시보드에서 메모리 사용량 확인
```

## 🛠️ 9. 문제 해결

### 9.1 일반적인 문제

**빌드 실패:**
```bash
# 로그 확인
railway logs

# 의존성 문제 해결
pip install -r railway_requirements.txt
```

**메모리 부족:**
```bash
# TensorFlow CPU 버전 사용 확인
tensorflow-cpu==2.16.1

# 불필요한 패키지 제거
```

**포트 충돌:**
```bash
# 환경 변수 확인
railway variables

# PORT=8501 설정 확인
```

### 9.2 로그 분석
```bash
# 실시간 로그 확인
railway logs --follow

# 특정 시간대 로그
railway logs --since 1h
```

## 🔒 10. 보안 설정

### 10.1 환경 변수 보안
- 민감한 정보는 Railway 환경 변수로 설정
- GitHub에 API 키 등 노출 금지

### 10.2 접근 제어
- Railway 대시보드에서 팀 멤버 관리
- 배포 권한 설정

## 📈 11. 성능 최적화

### 11.1 메모리 최적화
```python
# Streamlit 설정 최적화
import streamlit as st

st.set_page_config(
    page_title="SectorFlow Lite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### 11.2 캐싱 활용
```python
@st.cache_data
def load_data():
    # 데이터 로딩 함수
    pass
```

## 🚀 12. 다음 단계

### 12.1 실시간 데이터 연동
- KRX API 연동
- WebSocket 실시간 업데이트
- 알림 시스템 구축

### 12.2 고급 기능
- 사용자 인증
- 포트폴리오 관리
- 백테스팅 결과 저장

---

## ✅ 배포 체크리스트

- [ ] Railway 계정 생성
- [ ] GitHub 저장소 생성 및 연결
- [ ] Railway 설정 파일 확인
- [ ] 환경 변수 설정
- [ ] 배포 실행
- [ ] 웹 접속 테스트
- [ ] 기능 테스트
- [ ] 모니터링 설정
- [ ] 자동 배포 확인

배포 완료 후 Railway에서 제공하는 URL로 접속하여 대시보드를 확인하세요!

**예상 배포 시간: 10-15분**
**월 예상 비용: $0-5 (무료 티어)**
