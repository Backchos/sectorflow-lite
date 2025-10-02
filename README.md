# SectorFlow Lite 🚀

**AI-Powered Korean Stock Trading System**

SectorFlow Lite는 한국 주식 시장을 대상으로 한 AI 기반 매매 시스템입니다. 머신러닝과 딥러닝 모델을 활용하여 주식 가격 상승을 예측하고, 백테스팅을 통해 전략의 성과를 검증합니다.

## ✨ 주요 기능

### 📊 데이터 파이프라인
- **라벨 생성**: 익일 종가 상승 여부 예측
- **윈도우링**: 시계열 데이터를 30일 윈도우로 변환
- **데이터 분할**: Train/Valid/Test 자동 분할
- **스케일링**: StandardScaler를 통한 데이터 정규화

### 🤖 머신러닝 모델
- **베이스라인 모델**: 로지스틱 회귀, 랜덤 포레스트, XGBoost
- **딥러닝 모델**: GRU, LSTM, Multi-Head Attention
- **하이브리드 모델**: GRU + LSTM 결합 모델
- **클래스 불균형 처리**: Class Weight, SMOTE 지원

### 💰 백테스팅 시스템
- **룰 기반 백테스트**: Z-score + RS 지표 기반 매매
- **모델 기반 백테스트**: AI 모델 예측 기반 매매
- **성과 비교**: 룰 기반 vs 모델 기반 성과 분석
- **상세 지표**: 수익률, 샤프비율, 최대낙폭, 승률 등

### 🎯 종목 클러스터링
- **PCA 차원 축소**: 고차원 데이터를 2D로 변환
- **K-Means 클러스터링**: 자동 최적 클러스터 수 탐색
- **클러스터 분석**: 종목별 특성 분석 및 분류
- **시각화**: 클러스터 분포 차트 생성

### 📈 자동 리포트
- **일일 리포트**: 데이터 범위, 파라미터, 성과표
- **Top-5 트레이드**: 최고 수익 거래 분석
- **에쿼티 커브**: 포트폴리오 가치 변화 차트
- **종합 요약**: 모든 결과를 통합한 최종 리포트

## 🏗️ 프로젝트 구조

```
sft_lite/
├── main.py                 # 메인 실행 스크립트
├── config.yaml            # 설정 파일
├── financial_analysis.py  # 기본 금융 분석
├── src/                   # 소스 코드
│   ├── dataio.py         # 데이터 파이프라인
│   ├── features.py       # 피처 계산
│   ├── rules.py          # 매매 룰
│   ├── baseline.py       # 베이스라인 모델
│   ├── gru.py            # 딥러닝 모델
│   ├── train.py          # 모델 훈련
│   ├── infer.py          # 모델 예측
│   ├── backtest.py       # 백테스팅
│   ├── clustering.py     # 종목 클러스터링
│   └── report_generator.py # 리포트 생성
├── data/                  # 데이터 저장소
│   ├── raw/              # 원본 데이터
│   └── interim/          # 중간 처리 데이터
├── reports/              # 생성된 리포트
│   └── charts/           # 차트 이미지
└── models/               # 훈련된 모델 (자동 생성)
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone [repository-url]
cd sft_lite

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. Smoke Test

```bash
# 테스트 실행
pytest -q

# Dry Run (실제 실행 없이 예상 출력만 확인)
python main.py --mode data --dry_run

# 테스트 실행 (시드 고정)
python main.py --mode train --run_id testseed
```

### 3. 설정 파일 수정

`config.yaml` 파일을 편집하여 분석할 종목과 기간을 설정합니다:

```yaml
data:
  tickers: ["005930", "000660", "035420", "005380", "006400"]  # 분석할 종목
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  lookback: 30

trading:
  fee_bps: 30      # 0.3% 수수료
  slippage_bps: 10 # 0.1% 슬리피지
  threshold: 0.5   # 분류 임계값

train:
  seed: 42         # 재현성을 위한 시드
  cv:
    timeseries: true
    folds: 5
```

### 4. 실행

```bash
# 전체 파이프라인 실행
python main.py --mode full

# 개별 모듈 실행
python main.py --mode data      # 데이터 파이프라인만
python main.py --mode train     # 모델 훈련만
python main.py --mode backtest  # 백테스팅만
python main.py --mode cluster   # 클러스터링만
python main.py --mode report    # 리포트 생성만

# 사용자 정의 실행 ID로 실행
python main.py --mode full --run_id demo_$(date +%Y%m%d)
```

### 5. 결과 확인

실행 후 다음 파일들이 생성됩니다:

```
runs/<run_id>/
├── config_snapshot.yaml    # 실행 시 설정 스냅샷
├── env_info.json          # 환경 정보
├── cv_metrics.json        # 교차 검증 결과
├── signals_model.csv      # 모델 신호
└── backtest_summary.json  # 백테스트 요약

reports/
└── summary_<날짜>_<run_id>.md  # 종합 리포트
```

## 📊 사용 예시

### 기본 실행
```python
from src.dataio import prepare_ml_data
from src.baseline import train_all_baselines
from src.backtest import run_backtest

# 1. 데이터 준비
config = {
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'lookback': 30,
    'feature_cols': ['close', 'volume', 'trading_value', 'returns']
}
processed_data = prepare_ml_data(['005930', '000660'], config)

# 2. 모델 훈련
baseline_results = train_all_baselines(processed_data)

# 3. 백테스팅
backtest_results = run_backtest(processed_data)
```

### 딥러닝 모델 훈련
```python
from src.train import train_all_models

# 딥러닝 모델 훈련
dl_results = train_all_models(
    processed_data, 
    model_types=['gru', 'lstm', 'attention'],
    config={'epochs': 50, 'batch_size': 32}
)
```

### 종목 클러스터링
```python
from src.clustering import prepare_clustering_data, perform_pca, perform_kmeans

# 클러스터링 실행
feature_array, symbols, metadata = prepare_clustering_data(processed_data)
pca_result = perform_pca(feature_array)
kmeans_result = perform_kmeans(pca_result['pca_result'])
```

## 📈 성과 지표

### 백테스팅 지표
- **총 수익률**: 전체 기간 수익률
- **연간화 수익률**: 연간 기준 수익률
- **최대 낙폭**: 최대 손실 구간
- **샤프 비율**: 위험 대비 수익률
- **승률**: 수익 거래 비율

### 모델 성능 지표
- **정확도**: 전체 예측 정확도
- **정밀도**: 양성 예측 정확도
- **재현율**: 실제 양성 탐지율
- **F1 점수**: 정밀도와 재현율의 조화평균
- **ROC AUC**: 분류 성능 종합 지표

## 🔧 설정 옵션

### 데이터 설정
```yaml
data:
  symbols: ["종목코드1", "종목코드2"]  # 분석할 종목
  start_date: "2024-01-01"            # 시작 날짜
  end_date: "2024-12-31"              # 종료 날짜
```

### 모델 설정
```yaml
model:
  sequence_length: 30    # 시계열 윈도우 크기
  hidden_size: 64        # 은닉층 크기
  num_layers: 2          # 레이어 수
  dropout: 0.2           # 드롭아웃 비율
  learning_rate: 0.001   # 학습률
  batch_size: 32         # 배치 크기
  epochs: 100            # 에포크 수
```

### 백테스트 설정
```yaml
backtest:
  initial_capital: 1000000  # 초기 자본 (원)
  commission_rate: 0.003    # 수수료율 (0.3%)
  position_size: 1.0        # 포지션 크기
```

## 📋 요구사항

### 필수 패키지
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
pyyaml>=6.0
pytest>=7.0.0         # 테스트용
```

### 선택적 패키지
```
tensorflow>=2.10.0    # 딥러닝 모델용
xgboost>=1.6.0        # XGBoost 모델용
```

## 🔧 주요 설정 파라미터

| 섹션 | 파라미터 | 설명 | 기본값 |
|------|----------|------|--------|
| `data` | `tickers` | 분석할 종목 코드 리스트 | `["005930", "000660"]` |
| `data` | `lookback` | 시계열 윈도우 크기 | `30` |
| `trading` | `fee_bps` | 수수료 (bps) | `30` (0.3%) |
| `trading` | `slippage_bps` | 슬리피지 (bps) | `10` (0.1%) |
| `train` | `seed` | 재현성을 위한 시드 | `42` |
| `train` | `cv.folds` | 교차 검증 폴드 수 | `5` |
| `model` | `sequence_length` | 시퀀스 길이 | `30` |
| `model` | `epochs` | 훈련 에포크 수 | `100` |

## 🔄 파이프라인 다이어그램

```
데이터 수집 → 피처 계산 → 라벨 생성 → 데이터 분할
     ↓
모델 훈련 → 교차 검증 → 모델 평가 → 신호 생성
     ↓
백테스트 → 성과 분석 → 클러스터링 → 리포트 생성
```

## 📈 예시 결과

### 백테스트 성과 (예시)
- **총 수익률**: 15.2%
- **연간화 수익률**: 18.5%
- **최대 낙폭**: -8.3%
- **샤프 비율**: 1.24
- **승률**: 64.2%

### 모델 성능 (예시)
- **정확도**: 0.752
- **F1 점수**: 0.681
- **ROC AUC**: 0.823

## 🔍 재현성 보장

### 시드 설정
모든 실행에서 동일한 시드(42)를 사용하여 재현성을 보장합니다:

```python
# 자동으로 설정됨
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)  # TensorFlow 사용 시
```

### 버전 관리
- Python 3.8+ 지원
- 패키지 버전 고정 (requirements.txt)
- Git hash 자동 기록

### 검증 명령어
```bash
# 설치 및 테스트
pip install -r requirements.txt
pytest -q

# Dry Run
python main.py --mode data --dry_run

# 재현성 테스트
python main.py --mode train --run_id testseed
python main.py --mode train --run_id testseed  # 동일한 결과 확인

# 데모 실행
python main.py --mode full --run_id demo_$(date +%Y%m%d)
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📊 데이터 소스

### 현재 상태
- **데이터 타입**: 샘플 데이터 (랜덤 생성)
- **데이터 경로**: `data/raw/` (자동 생성)
- **데이터 형식**: CSV (OHLCV)

### 향후 계획
- **실제 데이터**: KRX API 또는 데이터 제공업체 연동
- **데이터 수집**: `ingest.py` 모듈 추가 예정
- **지원 형식**: CSV, JSON, API

### 데이터 구조
```csv
date,open,high,low,close,volume
2024-01-01,100.0,102.0,98.0,100.0,1000000
2024-01-02,100.0,103.0,99.0,101.0,1100000
```

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다. 실제 투자에 사용할 경우 발생하는 모든 손실에 대해 개발자는 책임지지 않습니다. 투자 결정은 충분한 검토와 리스크 관리 하에 본인의 책임으로 하시기 바랍니다.

## 📞 문의

- **이메일**: [your-email@example.com]
- **GitHub Issues**: [repository-url]/issues
- **문서**: [documentation-url]

---

**SectorFlow Lite v1.0** - AI-Powered Korean Stock Trading System
