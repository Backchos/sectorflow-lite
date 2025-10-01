# 🔄 SectorFlow Lite 업데이트 워크플로

이 문서는 SectorFlow Lite 프로젝트의 수정/개선 사항을 로컬에서 테스트하고 AWS EC2에 배포하는 워크플로를 설명합니다.

## 📋 워크플로 개요

```
로컬 개발 → 테스트 → Git 커밋 → 서버 배포 → 검증
    ↓         ↓        ↓         ↓        ↓
  기능 수정  로컬 테스트  Git Push  SSH 배포  운영 확인
```

## 🏠 1. 로컬 개발 및 테스트

### 1.1 개발 환경 설정
```bash
# 프로젝트 클론 (처음만)
git clone https://github.com/your-username/sectorflow-lite.git
cd sectorflow-lite

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 1.2 기능 수정/개선
- **대시보드 수정**: `examples/app_streamlit.py` 편집
- **핵심 로직 수정**: `src/` 디렉토리 파일들 편집
- **설정 변경**: `config.yaml` 수정
- **의존성 추가**: `requirements.txt` 수정

### 1.3 로컬 테스트
```bash
# Streamlit 대시보드 테스트
streamlit run examples/app_streamlit.py

# 전체 파이프라인 테스트
python main.py --mode full

# 개별 모듈 테스트
python main.py --mode train
python main.py --mode backtest

# 단위 테스트
pytest tests/
```

### 1.4 테스트 체크리스트
- [ ] 대시보드가 정상 로드되는가?
- [ ] 모든 탭이 올바르게 작동하는가?
- [ ] 필터링 기능이 정상 작동하는가?
- [ ] CSV 다운로드가 작동하는가?
- [ ] 차트가 올바르게 표시되는가?
- [ ] 에러 메시지가 적절한가?

## 📝 2. Git 커밋 및 푸시

### 2.1 변경사항 확인
```bash
# 변경된 파일 확인
git status

# 변경사항 차이 확인
git diff

# 특정 파일 변경사항 확인
git diff examples/app_streamlit.py
```

### 2.2 커밋 메시지 작성 가이드
```bash
# 기능 추가
git commit -m "feat: add interactive charts to dashboard"

# 버그 수정
git commit -m "fix: resolve CSV download encoding issue"

# 문서 업데이트
git commit -m "docs: update deployment guide"

# 의존성 업데이트
git commit -m "deps: update streamlit to 1.38.0"
```

### 2.3 푸시
```bash
# 현재 브랜치에 푸시
git push origin main

# 새 브랜치 생성 및 푸시 (큰 변경사항의 경우)
git checkout -b feature/new-dashboard
git push origin feature/new-dashboard
```

## 🚀 3. 서버 배포

### 3.1 수동 배포 (SSH)
```bash
# EC2 서버 접속
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# 프로젝트 디렉토리로 이동
cd /home/ubuntu/sft_lite

# 최신 코드 가져오기
git pull origin main

# 의존성 업데이트 (requirements.txt 변경시)
source .venv/bin/activate
pip install -r requirements.txt

# 서비스 재시작
sudo systemctl restart sft_streamlit

# 서비스 상태 확인
sudo systemctl status sft_streamlit
```

### 3.2 배포 스크립트 생성 (자동화)
```bash
# 배포 스크립트 생성
nano deploy.sh
```

다음 내용을 입력:
```bash
#!/bin/bash
set -e

echo "🚀 Starting deployment..."

# 프로젝트 디렉토리로 이동
cd /home/ubuntu/sft_lite

# 최신 코드 가져오기
echo "📥 Pulling latest changes..."
git pull origin main

# 가상환경 활성화 및 의존성 업데이트
echo "📦 Updating dependencies..."
source .venv/bin/activate
pip install -r requirements.txt

# 서비스 재시작
echo "🔄 Restarting service..."
sudo systemctl restart sft_streamlit

# 서비스 상태 확인
echo "✅ Checking service status..."
sudo systemctl status sft_streamlit --no-pager

echo "🎉 Deployment completed!"
```

스크립트 실행 권한 부여:
```bash
chmod +x deploy.sh
```

### 3.3 원격 배포 (로컬에서 실행)
```bash
# 로컬에서 원격 배포 실행
ssh -i your-key.pem ubuntu@YOUR_EC2_IP 'cd /home/ubuntu/sft_lite && ./deploy.sh'
```

## 🔍 4. 배포 검증

### 4.1 서비스 상태 확인
```bash
# SSH로 서버 접속 후
sudo systemctl status sft_streamlit
sudo journalctl -u sft_streamlit -f  # 실시간 로그 확인
```

### 4.2 웹 접속 확인
- 브라우저에서 `http://YOUR_EC2_IP` 또는 `https://your-domain.com` 접속
- 대시보드가 정상 로드되는지 확인
- 모든 기능이 작동하는지 테스트

### 4.3 성능 확인
```bash
# 서버 리소스 사용량 확인
htop
df -h  # 디스크 사용량
free -h  # 메모리 사용량
```

## 🤖 5. GitHub Actions 자동 배포 (고급)

### 5.1 GitHub Actions 워크플로 생성
`.github/workflows/deploy.yml` 파일 생성:

```yaml
name: Deploy to AWS EC2

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Deploy to EC2
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.EC2_HOST }}
        username: ubuntu
        key: ${{ secrets.EC2_SSH_KEY }}
        script: |
          cd /home/ubuntu/sft_lite
          git pull origin main
          source .venv/bin/activate
          pip install -r requirements.txt
          sudo systemctl restart sft_streamlit
```

### 5.2 GitHub Secrets 설정
GitHub 저장소 → Settings → Secrets and variables → Actions에서 다음 설정:

- `EC2_HOST`: EC2 인스턴스 IP 또는 도메인
- `EC2_SSH_KEY`: EC2 접속용 SSH 개인키

### 5.3 자동 배포 활성화
1. GitHub 저장소에 워크플로 파일 커밋
2. main 브랜치에 푸시하면 자동으로 배포 실행
3. Actions 탭에서 배포 상태 확인

## 🔄 6. 롤백 전략

### 6.1 이전 버전으로 롤백
```bash
# SSH로 서버 접속
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# 프로젝트 디렉토리로 이동
cd /home/ubuntu/sft_lite

# 이전 커밋으로 되돌리기
git log --oneline  # 커밋 히스토리 확인
git reset --hard COMMIT_HASH  # 특정 커밋으로 되돌리기

# 서비스 재시작
sudo systemctl restart sft_streamlit
```

### 6.2 롤백 스크립트
```bash
#!/bin/bash
# rollback.sh

if [ -z "$1" ]; then
    echo "Usage: ./rollback.sh <commit-hash>"
    exit 1
fi

echo "🔄 Rolling back to commit: $1"
cd /home/ubuntu/sft_lite
git reset --hard $1
sudo systemctl restart sft_streamlit
echo "✅ Rollback completed!"
```

## 📊 7. 모니터링 및 알림

### 7.1 헬스체크 스크립트
```bash
#!/bin/bash
# health_check.sh

URL="http://localhost:8501"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $URL)

if [ $RESPONSE -eq 200 ]; then
    echo "✅ Service is healthy"
    exit 0
else
    echo "❌ Service is down (HTTP $RESPONSE)"
    # 알림 전송 (Slack, Email 등)
    exit 1
fi
```

### 7.2 Cron 작업으로 헬스체크
```bash
# crontab -e
# 5분마다 헬스체크
*/5 * * * * /home/ubuntu/sft_lite/health_check.sh
```

## 🚨 8. 문제 해결

### 8.1 일반적인 문제들

**서비스가 시작되지 않는 경우:**
```bash
# 로그 확인
sudo journalctl -u sft_streamlit -f

# 포트 충돌 확인
sudo lsof -i :8501

# 권한 확인
ls -la /home/ubuntu/sft_lite/
```

**의존성 문제:**
```bash
# 가상환경 재생성
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**메모리 부족:**
```bash
# 메모리 사용량 확인
free -h
htop

# 스왑 파일 생성 (필요시)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 8.2 로그 분석
```bash
# Streamlit 로그
sudo journalctl -u sft_streamlit --since "1 hour ago"

# Nginx 로그
sudo tail -f /var/log/nginx/error.log

# 시스템 로그
sudo tail -f /var/log/syslog
```

## 📈 9. 성능 최적화

### 9.1 Streamlit 최적화
```python
# app_streamlit.py에서 캐싱 활용
@st.cache_data
def load_data():
    # 데이터 로딩 로직
    pass
```

### 9.2 Nginx 최적화
```nginx
# /etc/nginx/sites-available/sft_lite
server {
    # ... 기존 설정 ...
    
    # 캐싱 설정
    location ~* \.(css|js|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # 압축 설정
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
}
```

## ✅ 10. 배포 체크리스트

### 10.1 배포 전 체크리스트
- [ ] 로컬에서 모든 기능 테스트 완료
- [ ] Git 커밋 메시지가 명확한가?
- [ ] 변경사항이 문서화되었는가?
- [ ] 의존성 변경사항이 requirements.txt에 반영되었는가?

### 10.2 배포 후 체크리스트
- [ ] 서비스가 정상 시작되었는가?
- [ ] 웹 접속이 정상인가?
- [ ] 모든 탭이 작동하는가?
- [ ] 로그에 에러가 없는가?
- [ ] 성능이 이전과 동일한가?

---

## 🎯 워크플로 요약

1. **로컬 개발**: 기능 수정/개선
2. **로컬 테스트**: `streamlit run examples/app_streamlit.py`
3. **Git 커밋**: `git add . && git commit -m "message"`
4. **Git 푸시**: `git push origin main`
5. **서버 배포**: SSH 접속 → `git pull` → `sudo systemctl restart sft_streamlit`
6. **검증**: 브라우저 접속 확인

이 워크플로를 따라하면 안전하고 효율적으로 SectorFlow Lite를 업데이트할 수 있습니다!


