# 🚀 SectorFlow Lite AWS 배포 가이드

이 문서는 SectorFlow Lite를 AWS EC2에 배포하고 운영하는 방법을 설명합니다.

## 📋 사전 요구사항

- AWS 계정
- 도메인 (선택사항, IP로도 접속 가능)
- Git 저장소 접근 권한

## 🖥️ 1. EC2 인스턴스 생성

### 1.1 인스턴스 설정
- **인스턴스 타입**: t3.small (2 vCPU, 2GB RAM)
- **AMI**: Ubuntu Server 22.04 LTS
- **스토리지**: 20GB GP3
- **보안 그룹**: 
  - SSH (22) - 본인 IP만
  - HTTP (80) - 0.0.0.0/0
  - HTTPS (443) - 0.0.0.0/0

### 1.2 탄력적 IP 할당
1. EC2 콘솔 → 탄력적 IP → 탄력적 IP 주소 할당
2. 할당된 IP를 인스턴스에 연결
3. 도메인이 있다면 Route 53에서 A 레코드 설정

## 🔧 2. 서버 초기 설정

### 2.1 서버 접속 및 업데이트
```bash
# SSH 접속
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# 시스템 업데이트
sudo apt update && sudo apt -y upgrade

# 필수 패키지 설치
sudo apt -y install git python3-pip python3-venv nginx certbot python3-certbot-nginx
```

### 2.2 프로젝트 클론 및 환경 설정
```bash
# 프로젝트 클론 (YOUR_REPO_URL을 실제 저장소 URL로 변경)
git clone https://github.com/your-username/sectorflow-lite.git sft_lite
cd sft_lite

# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 🔄 3. Systemd 서비스 설정

### 3.1 서비스 파일 생성
```bash
sudo nano /etc/systemd/system/sft_streamlit.service
```

다음 내용을 입력:
```ini
[Unit]
Description=SectorFlow Streamlit Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/sft_lite
Environment="PATH=/home/ubuntu/sft_lite/.venv/bin"
ExecStart=/home/ubuntu/sft_lite/.venv/bin/streamlit run examples/app_streamlit.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 3.2 서비스 활성화
```bash
# 서비스 파일 리로드
sudo systemctl daemon-reload

# 서비스 활성화 (부팅 시 자동 시작)
sudo systemctl enable sft_streamlit

# 서비스 시작
sudo systemctl start sft_streamlit

# 서비스 상태 확인
sudo systemctl status sft_streamlit
```

## 🌐 4. Nginx 리버스 프록시 설정

### 4.1 Nginx 설정 파일 생성
```bash
sudo nano /etc/nginx/sites-available/sft_lite
```

다음 내용을 입력 (YOUR_DOMAIN을 실제 도메인으로 변경):
```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 4.2 Nginx 설정 활성화
```bash
# 설정 파일 활성화
sudo ln -s /etc/nginx/sites-available/sft_lite /etc/nginx/sites-enabled/sft_lite

# 기본 설정 비활성화 (충돌 방지)
sudo rm /etc/nginx/sites-enabled/default

# 설정 테스트
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx
```

## 🔒 5. SSL 인증서 설정 (Let's Encrypt)

### 5.1 SSL 인증서 발급
```bash
# 도메인이 있는 경우에만 실행
sudo certbot --nginx -d YOUR_DOMAIN

# IP만 있는 경우에는 이 단계를 건너뛰세요
```

### 5.2 자동 갱신 설정
```bash
# 자동 갱신 테스트
sudo certbot renew --dry-run

# crontab에 자동 갱신 추가
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

## ⏰ 6. 자동 실행 설정 (선택사항)

### 6.1 매일 자동 분석 실행
```bash
# crontab 편집
crontab -e

# 다음 라인 추가 (매일 06:00에 실행)
0 6 * * * cd /home/ubuntu/sft_lite && source .venv/bin/activate && python main.py --mode full --run_id auto_$(date +\%Y\%m\%d) >> runs/cron.log 2>&1
```

## 🔍 7. 배포 확인

### 7.1 서비스 상태 확인
```bash
# Streamlit 서비스 상태
sudo systemctl status sft_streamlit

# Nginx 상태
sudo systemctl status nginx

# 포트 확인
sudo netstat -tlnp | grep :80
sudo netstat -tlnp | grep :8501
```

### 7.2 브라우저 접속
- 도메인 사용: `https://your-domain.com`
- IP 사용: `http://YOUR_EC2_IP`

## 🛠️ 8. 로그 확인 및 디버깅

### 8.1 로그 파일 위치
```bash
# Streamlit 서비스 로그
sudo journalctl -u sft_streamlit -f

# Nginx 로그
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# 자동 실행 로그
tail -f /home/ubuntu/sft_lite/runs/cron.log
```

### 8.2 일반적인 문제 해결
```bash
# 서비스 재시작
sudo systemctl restart sft_streamlit

# Nginx 재시작
sudo systemctl restart nginx

# 포트 충돌 확인
sudo lsof -i :8501
sudo lsof -i :80
```

## 🔄 9. 업데이트 워크플로

### 9.1 수동 업데이트
```bash
# SSH 접속
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# 프로젝트 디렉토리로 이동
cd /home/ubuntu/sft_lite

# 최신 코드 가져오기
git pull origin main

# 의존성 업데이트 (필요시)
source .venv/bin/activate
pip install -r requirements.txt

# 서비스 재시작
sudo systemctl restart sft_streamlit
```

### 9.2 자동 배포 (GitHub Actions)
GitHub Actions를 사용한 자동 배포 설정은 `docs/update_workflow.md`를 참조하세요.

## 💰 10. 비용 최적화

### 10.1 인스턴스 최적화
- **t3.small**: 월 약 $15-20
- **스토리지**: 20GB GP3 월 약 $2
- **데이터 전송**: 월 1GB 무료, 초과시 $0.09/GB

### 10.2 비용 절약 팁
- 개발/테스트용: t3.micro 사용 (월 $8-10)
- 프로덕션: t3.small 이상 권장
- 스토리지: 필요에 따라 10-30GB 조정
- 자동 스케일링: CloudWatch + Lambda 활용

## 🔐 11. 보안 고려사항

### 11.1 방화벽 설정
```bash
# UFW 방화벽 설정
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
```

### 11.2 SSH 보안 강화
```bash
# SSH 키 인증만 허용
sudo nano /etc/ssh/sshd_config
# PasswordAuthentication no
# PermitRootLogin no
sudo systemctl restart ssh
```

## 📊 12. 모니터링 설정

### 12.1 CloudWatch 에이전트 (선택사항)
```bash
# CloudWatch 에이전트 설치
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb
```

### 12.2 기본 모니터링
- EC2 인스턴스 메트릭: CPU, 메모리, 디스크
- 애플리케이션 로그: systemd journal
- 웹 서버 로그: Nginx access/error logs

---

## ✅ 배포 체크리스트

- [ ] EC2 인스턴스 생성 및 보안 그룹 설정
- [ ] 탄력적 IP 할당 (선택사항)
- [ ] 서버 초기 설정 및 패키지 설치
- [ ] 프로젝트 클론 및 가상환경 설정
- [ ] Systemd 서비스 파일 생성 및 활성화
- [ ] Nginx 리버스 프록시 설정
- [ ] SSL 인증서 설정 (도메인 사용시)
- [ ] 자동 실행 설정 (선택사항)
- [ ] 브라우저 접속 확인
- [ ] 로그 모니터링 설정

배포 완료 후 `http://YOUR_EC2_IP` 또는 `https://your-domain.com`으로 접속하여 대시보드를 확인하세요!
