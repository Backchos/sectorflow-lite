#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Report Generator Module
자동 리포트 생성 및 문서화

Functions:
- generate_daily_report: 일일 리포트 생성
- create_equity_curve: 에쿼티 커브 생성
- generate_performance_table: 성과표 생성
- create_top_trades_report: Top-5 트레이드 리포트
- save_report: 리포트 저장
- generate_summary_report: 종합 요약 리포트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def generate_daily_report(date: str,
                         data_summary: Dict[str, Any],
                         model_results: Dict[str, Any],
                         backtest_results: Dict[str, Any],
                         config: Dict[str, Any]) -> str:
    """
    일일 리포트 생성
    
    Args:
        date: 리포트 날짜 (YYYYMMDD)
        data_summary: 데이터 요약 정보
        model_results: 모델 결과
        backtest_results: 백테스트 결과
        config: 설정 정보
        
    Returns:
        리포트 마크다운 문자열
    """
    print(f"📊 {date} 일일 리포트 생성 중...")
    
    # 리포트 헤더
    report = f"""# SectorFlow Lite - Daily Report
**Date:** {date}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📈 Executive Summary

### 데이터 범위
- **기간:** {config.get('start_date', 'N/A')} ~ {config.get('end_date', 'N/A')}
- **종목 수:** {data_summary.get('total_symbols', 0)}개
- **분석 종목:** {', '.join(data_summary.get('symbols', []))}

### 주요 성과 지표
"""
    
    # 백테스트 결과 요약
    if backtest_results:
        for strategy_name, results in backtest_results.items():
            if isinstance(results, dict) and 'metrics' in results:
                metrics = results['metrics']
                report += f"""
#### {strategy_name.replace('_', ' ').title()}
- **총 수익률:** {metrics.get('total_return', 0)*100:.2f}%
- **연간화 수익률:** {metrics.get('annualized_return', 0)*100:.2f}%
- **최대 낙폭:** {metrics.get('max_drawdown', 0)*100:.2f}%
- **샤프 비율:** {metrics.get('sharpe_ratio', 0):.2f}
- **총 거래 수:** {metrics.get('total_trades', 0)}회
- **승률:** {metrics.get('win_rate', 0)*100:.1f}%
"""
    
    # 모델 성과 요약
    if model_results:
        report += f"""
### 모델 성과
"""
        for symbol, results in model_results.items():
            if isinstance(results, dict) and 'models' in results:
                best_model = results.get('best_model', {})
                if best_model:
                    model_name = best_model.get('model_name', 'Unknown')
                    valid_metrics = best_model.get('valid_metrics', {})
                    report += f"""
#### {symbol}
- **최고 모델:** {model_name}
- **정확도:** {valid_metrics.get('accuracy', 0):.3f}
- **F1 점수:** {valid_metrics.get('f1_score', 0):.3f}
- **ROC AUC:** {valid_metrics.get('roc_auc', 0):.3f}
"""
    
    # 상세 분석 섹션
    report += f"""

---

## 📊 Detailed Analysis

### 데이터 품질
"""
    
    # 데이터 품질 정보
    if 'data_info' in data_summary:
        for symbol, info in data_summary['data_info'].items():
            report += f"""
#### {symbol}
- **훈련 샘플:** {info.get('train_samples', 0)}개
- **검증 샘플:** {info.get('valid_samples', 0)}개
- **테스트 샘플:** {info.get('test_samples', 0)}개
- **양성 비율:** {info.get('positive_ratio', 0):.1%}
"""
    
    # 모델 상세 성과
    if model_results:
        report += f"""
### 모델 상세 성과
"""
        for symbol, results in model_results.items():
            if isinstance(results, dict) and 'comparison' in results:
                comparison_df = results['comparison']
                report += f"""
#### {symbol} 모델 비교
{comparison_df.to_string(index=False)}
"""
    
    # 백테스트 상세 결과
    if backtest_results:
        report += f"""
### 백테스트 상세 결과
"""
        for strategy_name, results in backtest_results.items():
            if isinstance(results, dict) and 'metrics' in results:
                metrics = results['metrics']
                report += f"""
#### {strategy_name.replace('_', ' ').title()}
- **초기 자본:** {results.get('config', {}).get('initial_capital', 0):,}원
- **최종 자본:** {metrics.get('final_capital', 0):,.0f}원
- **총 수수료:** {metrics.get('total_commission', 0):,.0f}원
- **평균 수익:** {metrics.get('avg_profit', 0):,.0f}원
- **최대 수익:** {metrics.get('max_profit', 0):,.0f}원
- **최대 손실:** {metrics.get('max_loss', 0):,.0f}원
"""
    
    # 설정 정보
    report += f"""
### 설정 정보
- **Lookback Window:** {config.get('lookback', 'N/A')}일
- **Feature Columns:** {', '.join(config.get('feature_cols', []))}
- **Scale Method:** {config.get('scale_method', 'N/A')}
- **Commission Rate:** {config.get('commission_rate', 0)*100:.1f}%
"""
    
    # 푸터
    report += f"""

---

## 📝 Notes
- 이 리포트는 SectorFlow Lite 시스템에 의해 자동 생성되었습니다.
- 모든 수익률과 성과 지표는 백테스트 결과이며, 실제 투자 결과와 다를 수 있습니다.
- 투자 결정 시 충분한 검토와 리스크 관리가 필요합니다.

**Report Generated by:** SectorFlow Lite v1.0  
**Contact:** [Your Contact Information]
"""
    
    print(f"✅ {date} 일일 리포트 생성 완료!")
    return report

def create_equity_curve(portfolio_values: List[Dict[str, Any]], 
                       title: str = "Portfolio Equity Curve") -> str:
    """
    에쿼티 커브 생성
    
    Args:
        portfolio_values: 포트폴리오 가치 리스트
        title: 차트 제목
        
    Returns:
        차트 파일 경로
    """
    print("📈 에쿼티 커브 생성 중...")
    
    if not portfolio_values:
        print("⚠️ 포트폴리오 데이터가 없습니다.")
        return None
    
    # 데이터프레임 생성
    df = pd.DataFrame(portfolio_values)
    
    # 날짜 컬럼 처리
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        x_axis = df['date']
    else:
        x_axis = range(len(df))
    
    # 차트 생성
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, df['portfolio_value'], linewidth=2, color='blue', alpha=0.8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date' if 'date' in df.columns else 'Period')
    plt.ylabel('Portfolio Value (KRW)')
    plt.grid(True, alpha=0.3)
    
    # 포맷팅
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 파일 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"equity_curve_{timestamp}.png"
    filepath = os.path.join("reports", "charts", filename)
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 에쿼티 커브 저장 완료: {filepath}")
    return filepath

def generate_performance_table(all_results: Dict[str, Any]) -> pd.DataFrame:
    """
    성과표 생성
    
    Args:
        all_results: 모든 결과 딕셔너리
        
    Returns:
        성과표 데이터프레임
    """
    print("📊 성과표 생성 중...")
    
    performance_data = []
    
    for strategy_name, results in all_results.items():
        if isinstance(results, dict) and 'metrics' in results:
            metrics = results['metrics']
            
            performance_data.append({
                'Strategy': strategy_name.replace('_', ' ').title(),
                'Total Return (%)': metrics.get('total_return', 0) * 100,
                'Annualized Return (%)': metrics.get('annualized_return', 0) * 100,
                'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Total Trades': metrics.get('total_trades', 0),
                'Win Rate (%)': metrics.get('win_rate', 0) * 100,
                'Avg Profit (KRW)': metrics.get('avg_profit', 0),
                'Final Capital (KRW)': metrics.get('final_capital', 0)
            })
    
    df_performance = pd.DataFrame(performance_data)
    
    # 수익률 기준으로 정렬
    if not df_performance.empty:
        df_performance = df_performance.sort_values('Total Return (%)', ascending=False)
    
    print("✅ 성과표 생성 완료!")
    return df_performance

def create_top_trades_report(trades: List[Dict[str, Any]], 
                           top_n: int = 5) -> str:
    """
    Top-N 트레이드 리포트 생성
    
    Args:
        trades: 거래 리스트
        top_n: 상위 N개
        
    Returns:
        트레이드 리포트 문자열
    """
    print(f"🏆 Top-{top_n} 트레이드 리포트 생성 중...")
    
    if not trades:
        return "거래 데이터가 없습니다."
    
    # 매도 거래만 필터링 (수익/손실 계산된 거래)
    sell_trades = [t for t in trades if t.get('action') == 'SELL' and 'net_profit' in t]
    
    if not sell_trades:
        return "완료된 거래가 없습니다."
    
    # 수익률 기준으로 정렬
    sell_trades.sort(key=lambda x: x.get('net_profit', 0), reverse=True)
    
    # Top-N 선택
    top_trades = sell_trades[:top_n]
    
    report = f"""
## 🏆 Top-{top_n} Trades

| Rank | Date | Action | Price | Quantity | Net Profit | Return % |
|------|------|--------|-------|----------|------------|----------|
"""
    
    for i, trade in enumerate(top_trades, 1):
        date = trade.get('date', 'N/A')
        action = trade.get('action', 'N/A')
        price = trade.get('price', 0)
        quantity = trade.get('quantity', 0)
        net_profit = trade.get('net_profit', 0)
        
        # 수익률 계산
        if 'gross_profit' in trade and 'commission' in trade:
            gross_profit = trade['gross_profit']
            commission = trade['commission']
            total_cost = gross_profit - net_profit  # 대략적인 진입 비용
            return_pct = (net_profit / total_cost * 100) if total_cost > 0 else 0
        else:
            return_pct = 0
        
        report += f"| {i} | {date} | {action} | {price:,.0f} | {quantity:,.0f} | {net_profit:,.0f} | {return_pct:+.2f}% |\n"
    
    # 통계 요약
    total_profit = sum(t.get('net_profit', 0) for t in top_trades)
    avg_profit = total_profit / len(top_trades) if top_trades else 0
    
    report += f"""
### Summary
- **Total Profit:** {total_profit:,.0f} KRW
- **Average Profit:** {avg_profit:,.0f} KRW
- **Best Trade:** {top_trades[0].get('net_profit', 0):,.0f} KRW
- **Worst Trade:** {top_trades[-1].get('net_profit', 0):,.0f} KRW
"""
    
    print(f"✅ Top-{top_n} 트레이드 리포트 생성 완료!")
    return report

def save_report(report_content: str, 
                filename: str = None,
                directory: str = "reports") -> str:
    """
    리포트 저장
    
    Args:
        report_content: 리포트 내용
        filename: 파일명 (없으면 자동 생성)
        directory: 저장 디렉토리
        
    Returns:
        저장된 파일 경로
    """
    print("💾 리포트 저장 중...")
    
    # 디렉토리 생성
    os.makedirs(directory, exist_ok=True)
    
    # 파일명 생성
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"summary_{timestamp}.md"
    
    # 파일 경로
    filepath = os.path.join(directory, filename)
    
    # 파일 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 리포트 저장 완료: {filepath}")
    return filepath

def generate_summary_report(data_summary: Dict[str, Any],
                          model_results: Dict[str, Any],
                          backtest_results: Dict[str, Any],
                          config: Dict[str, Any],
                          include_charts: bool = True,
                          run_id: str = None) -> str:
    """
    종합 요약 리포트 생성
    
    Args:
        data_summary: 데이터 요약
        model_results: 모델 결과
        backtest_results: 백테스트 결과
        config: 설정
        include_charts: 차트 포함 여부
        
    Returns:
        리포트 파일 경로
    """
    print("📊 종합 요약 리포트 생성 시작...")
    
    # 현재 날짜
    current_date = datetime.now().strftime('%Y%m%d')
    
    # 1. 일일 리포트 생성
    daily_report = generate_daily_report(
        current_date, data_summary, model_results, backtest_results, config
    )
    
    # 2. 환경 정보 추가
    env_info = get_environment_info()
    daily_report += f"""

---

## 🔧 Environment Information

- **Python Version:** {env_info['python_version']}
- **Pandas Version:** {env_info['pandas_version']}
- **NumPy Version:** {env_info['numpy_version']}
- **Scikit-learn Version:** {env_info['sklearn_version']}
- **Run ID:** {run_id or 'N/A'}
- **Git Hash:** {env_info['git_hash']}
- **Execution Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # 2. 성과표 생성
    performance_table = generate_performance_table(backtest_results)
    
    # 3. 성과표를 마크다운 테이블로 변환
    performance_md = performance_table.to_string(index=False, float_format='%.2f')
    
    # 4. Top-5 트레이드 리포트 생성
    top_trades_report = ""
    if backtest_results:
        for strategy_name, results in backtest_results.items():
            if isinstance(results, dict) and 'trades' in results:
                trades = results['trades']
                top_trades_report += create_top_trades_report(trades, top_n=5)
                break  # 첫 번째 전략의 거래만 사용
    
    # 5. 차트 생성 (선택사항)
    chart_paths = []
    if include_charts and backtest_results:
        for strategy_name, results in backtest_results.items():
            if isinstance(results, dict) and 'portfolio_values' in results:
                chart_path = create_equity_curve(
                    results['portfolio_values'], 
                    f"{strategy_name.replace('_', ' ').title()} Equity Curve"
                )
                if chart_path:
                    chart_paths.append(chart_path)
    
    # 6. 최종 리포트 조합
    final_report = daily_report
    
    # 성과표 추가
    if not performance_table.empty:
        final_report += f"""

---

## 📊 Performance Comparison

{performance_md}
"""
    
    # Top 트레이드 추가
    if top_trades_report:
        final_report += f"""

---

{top_trades_report}
"""
    
    # 차트 추가
    if chart_paths:
        final_report += f"""

---

## 📈 Charts

"""
        for chart_path in chart_paths:
            chart_name = os.path.basename(chart_path)
            final_report += f"![{chart_name}]({chart_path})\n\n"
    
    # 7. 한계와 다음 단계 추가
    limitations_section = get_limitations_and_next_steps()
    final_report += limitations_section
    
    # 8. 리포트 저장
    filename = f"summary_{current_date}_{run_id}.md" if run_id else f"summary_{current_date}.md"
    report_path = save_report(final_report, filename)
    
    print("✅ 종합 요약 리포트 생성 완료!")
    return report_path

def get_environment_info() -> Dict[str, str]:
    """환경 정보 수집"""
    import sys
    import subprocess
    import platform
    
    try:
        import pandas as pd
        pandas_version = pd.__version__
    except ImportError:
        pandas_version = "N/A"
    
    try:
        import numpy as np
        numpy_version = np.__version__
    except ImportError:
        numpy_version = "N/A"
    
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "N/A"
    
    # Git hash
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()[:8]
    except:
        git_hash = "N/A"
    
    return {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'pandas_version': pandas_version,
        'numpy_version': numpy_version,
        'sklearn_version': sklearn_version,
        'platform': platform.platform(),
        'git_hash': git_hash
    }

def get_limitations_and_next_steps() -> str:
    """한계와 다음 단계 섹션 생성"""
    return f"""

---

## ⚠️ 한계와 다음 단계

### 현재 한계
1. **데이터 제한**: 현재 샘플 데이터 기반으로 실제 시장 데이터 연동 필요
2. **모델 단순화**: 복잡한 시장 상황과 뉴스/이벤트 반영 부족
3. **거래 제약**: 실제 거래 시 유동성, 시장 충격 등 고려 부족
4. **리스크 관리**: 포트폴리오 레벨 리스크 관리 기능 미흡

### 다음 단계
1. **실제 데이터 연동**: KRX API 또는 데이터 제공업체 연동
2. **고급 모델**: Transformer, 앙상블 모델 도입
3. **실시간 시스템**: 실시간 데이터 처리 및 신호 생성
4. **리스크 관리**: VaR, CVaR 등 고급 리스크 지표 추가

### 권장사항
- 백테스트 결과는 과거 데이터 기반이므로 실제 투자 시 주의
- 충분한 포워드 테스트 후 실제 투자 고려
- 지속적인 모델 성능 모니터링 필요

---

**Report Generated by:** SectorFlow Lite v1.0  
**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

def main():
    """테스트용 메인 함수"""
    print("🚀 SectorFlow Lite - Report Generator Module 테스트")
    print("=" * 60)
    
    # 테스트용 데이터 생성
    test_data_summary = {
        'total_symbols': 2,
        'symbols': ['005930', '000660'],
        'data_info': {
            '005930': {
                'train_samples': 100,
                'valid_samples': 20,
                'test_samples': 20,
                'positive_ratio': 0.45
            },
            '000660': {
                'train_samples': 100,
                'valid_samples': 20,
                'test_samples': 20,
                'positive_ratio': 0.52
            }
        }
    }
    
    test_model_results = {
        '005930': {
            'models': [{
                'model_name': 'XGBoost',
                'valid_metrics': {
                    'accuracy': 0.75,
                    'f1_score': 0.68,
                    'roc_auc': 0.82
                }
            }]
        }
    }
    
    test_backtest_results = {
        'rule_based': {
            'metrics': {
                'total_return': 0.15,
                'annualized_return': 0.18,
                'max_drawdown': -0.08,
                'sharpe_ratio': 1.2,
                'total_trades': 25,
                'win_rate': 0.64,
                'avg_profit': 50000,
                'final_capital': 1150000
            },
            'trades': [
                {'date': '2024-01-15', 'action': 'SELL', 'price': 75000, 'quantity': 10, 'net_profit': 50000},
                {'date': '2024-02-20', 'action': 'SELL', 'price': 80000, 'quantity': 8, 'net_profit': 30000}
            ],
            'portfolio_values': [
                {'date': '2024-01-01', 'portfolio_value': 1000000},
                {'date': '2024-01-15', 'portfolio_value': 1050000},
                {'date': '2024-02-20', 'portfolio_value': 1080000}
            ]
        }
    }
    
    test_config = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'lookback': 30,
        'feature_cols': ['close', 'volume', 'trading_value'],
        'scale_method': 'standard',
        'commission_rate': 0.003
    }
    
    # 리포트 생성
    report_path = generate_summary_report(
        test_data_summary,
        test_model_results,
        test_backtest_results,
        test_config,
        include_charts=False  # 테스트에서는 차트 생성 안함
    )
    
    print(f"\n📄 생성된 리포트: {report_path}")
    print("\n✅ Report Generator Module 테스트 완료!")
    return report_path

if __name__ == "__main__":
    report_path = main()
