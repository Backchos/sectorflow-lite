"""
SectorFlow Lite Streamlit Dashboard
한국 주식 시장 AI 투자 전략 백테스팅 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="SectorFlow Lite Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 환경 변수 설정
SERVER_PORT = os.getenv("SERVER_PORT", "3000")
SERVER_ADDR = os.getenv("SERVER_ADDR", "0.0.0.0")

# 유틸리티 함수들
@st.cache_data
def load_json_file(file_path):
    """JSON 파일 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"JSON 파일 로드 실패: {file_path} - {str(e)}")
        return None

@st.cache_data
def load_csv_file(file_path):
    """CSV 파일 로드"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"CSV 파일 로드 실패: {file_path} - {str(e)}")
        return None

def get_available_runs():
    """runs/ 디렉토리에서 사용 가능한 실행 결과들 가져오기"""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []
    
    run_dirs = []
    for item in runs_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            run_dirs.append(item.name)
    
    # 최근 순으로 정렬
    run_dirs.sort(reverse=True)
    return run_dirs[:10]  # 최근 10개만

def get_csv_download_link(df, filename):
    """CSV 다운로드 링크 생성"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">🔽 CSV 다운로드</a>'
    return href

# 메인 앱
def main():
    st.title("📈 SectorFlow Lite Dashboard")
    st.markdown("한국 주식 시장 AI 투자 전략 백테스팅 대시보드")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # Run 선택
        available_runs = get_available_runs()
        if not available_runs:
            st.error("사용 가능한 실행 결과가 없습니다. 먼저 main.py를 실행해주세요.")
            st.stop()
        
        selected_run = st.selectbox(
            "📁 Run 선택",
            available_runs,
            help="분석할 실행 결과를 선택하세요"
        )
        
        st.divider()
        
        # 날짜 범위
        st.subheader("📅 날짜 필터")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("시작일", value=datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("종료일", value=datetime.now())
        
        # 티커 다중선택
        st.subheader("🏷️ 종목 필터")
        # 기본 티커들 (config.yaml에서 가져올 수 있지만 하드코딩)
        default_tickers = ["005930", "000660", "035420", "005380", "006400"]
        selected_tickers = st.multiselect(
            "종목 선택",
            default_tickers,
            default=default_tickers,
            help="분석할 종목들을 선택하세요"
        )
        
        st.divider()
        
        # 전략 토글
        st.subheader("🎯 전략 설정")
        strategy_mode = st.selectbox(
            "전략 모드",
            ["모델 기반(확률)", "룰 기반", "둘 다"],
            help="사용할 전략을 선택하세요"
        )
        
        # 임계값 슬라이더
        threshold = st.slider(
            "임계값",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            help="모델 신호 필터링 임계값"
        )
        
        st.divider()
        
        # 도움말
        st.subheader("💡 도움말")
        st.info("""
        **사용법:**
        1. Run 선택 후 날짜/종목 필터 설정
        2. 전략 모드와 임계값 조정
        3. 각 탭에서 결과 확인
        
        **필요 파일:**
        - signals_model.csv (모델 신호)
        - trades.csv (거래 내역)
        - equity_curve.csv (자본 곡선)
        - cv_metrics.json (교차검증 결과)
        - backtest_summary.json (백테스트 요약)
        """)
    
    # 상단 KPI
    st.header("📊 핵심 지표")
    
    # 선택된 Run 정보 로드
    run_path = Path("runs") / selected_run
    
    # KPI 계산
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📁 선택된 Run", selected_run)
    
    with col2:
        cv_metrics_path = run_path / "cv_metrics.json"
        if cv_metrics_path.exists():
            cv_metrics = load_json_file(cv_metrics_path)
            if cv_metrics and 'auc_mean' in cv_metrics:
                st.metric("🎯 CV AUC (평균)", f"{cv_metrics['auc_mean']:.3f}")
            else:
                st.metric("🎯 CV AUC (평균)", "N/A")
        else:
            st.metric("🎯 CV AUC (평균)", "파일 없음")
    
    with col3:
        backtest_summary_path = run_path / "backtest_summary.json"
        if backtest_summary_path.exists():
            backtest_summary = load_json_file(backtest_summary_path)
            if backtest_summary and 'sharpe_ratio' in backtest_summary:
                st.metric("📈 Best Sharpe", f"{backtest_summary['sharpe_ratio']:.3f}")
            else:
                st.metric("📈 Best Sharpe", "N/A")
        else:
            st.metric("📈 Best Sharpe", "파일 없음")
    
    st.divider()
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 개요", "📊 차트", "📡 신호", "💰 트레이드", "🔍 클러스터링", "📄 리포트"
    ])
    
    # 1. 개요 탭
    with tab1:
        st.subheader("📋 실행 결과 개요")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 교차검증 결과")
            if cv_metrics_path.exists():
                cv_metrics = load_json_file(cv_metrics_path)
                if cv_metrics:
                    st.json(cv_metrics)
                else:
                    st.warning("교차검증 결과를 로드할 수 없습니다.")
            else:
                st.warning("cv_metrics.json 파일이 없습니다.")
        
        with col2:
            st.subheader("📊 백테스트 요약")
            if backtest_summary_path.exists():
                backtest_summary = load_json_file(backtest_summary_path)
                if backtest_summary:
                    st.json(backtest_summary)
                else:
                    st.warning("백테스트 요약을 로드할 수 없습니다.")
            else:
                st.warning("backtest_summary.json 파일이 없습니다.")
    
    # 2. 차트 탭
    with tab2:
        st.subheader("📊 시각화 차트")
        
        # Equity Curve
        st.subheader("📈 자본 곡선")
        equity_curve_path = run_path / "equity_curve.csv"
        equity_curve_png = run_path / "equity_curve.png"
        
        if equity_curve_path.exists():
            equity_df = load_csv_file(equity_curve_path)
            if equity_df is not None:
                # 날짜 필터 적용
                if 'date' in equity_df.columns:
                    equity_df['date'] = pd.to_datetime(equity_df['date'])
                    mask = (equity_df['date'] >= pd.to_datetime(start_date)) & (equity_df['date'] <= pd.to_datetime(end_date))
                    equity_df = equity_df[mask]
                
                # Plotly 차트
                fig = px.line(equity_df, x='date', y='equity', title='자본 곡선')
                fig.update_layout(xaxis_title="날짜", yaxis_title="자본")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("자본 곡선 데이터를 로드할 수 없습니다.")
        elif equity_curve_png.exists():
            st.image(str(equity_curve_png), caption="자본 곡선 차트")
        else:
            st.warning("자본 곡선 파일이 없습니다.")
        
        # Daily Returns
        st.subheader("📊 일일 수익률")
        daily_returns_path = run_path / "daily_returns.csv"
        
        if daily_returns_path.exists():
            returns_df = load_csv_file(daily_returns_path)
            if returns_df is not None:
                # 날짜 필터 적용
                if 'date' in returns_df.columns:
                    returns_df['date'] = pd.to_datetime(returns_df['date'])
                    mask = (returns_df['date'] >= pd.to_datetime(start_date)) & (returns_df['date'] <= pd.to_datetime(end_date))
                    returns_df = returns_df[mask]
                
                # 바 차트
                fig = px.bar(returns_df, x='date', y='returns', title='일일 수익률')
                fig.update_layout(xaxis_title="날짜", yaxis_title="수익률")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("일일 수익률 데이터를 로드할 수 없습니다.")
        else:
            st.warning("일일 수익률 파일이 없습니다.")
        
        # 전략 비교 (임계값 적용)
        st.subheader("🎯 전략 비교")
        st.info(f"현재 설정: {strategy_mode}, 임계값: {threshold}")
        
        # 모델 vs 룰 기반 비교 차트 (예시)
        if strategy_mode == "둘 다":
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🤖 모델 기반")
                st.info("모델 기반 전략 결과가 여기에 표시됩니다.")
            with col2:
                st.subheader("📏 룰 기반")
                st.info("룰 기반 전략 결과가 여기에 표시됩니다.")
    
    # 3. 신호 탭
    with tab3:
        st.subheader("📡 신호 분석")
        
        # 모델 신호
        signals_model_path = run_path / "signals_model.csv"
        signals_rule_path = run_path / "signals_rule.csv"
        
        if strategy_mode in ["모델 기반(확률)", "둘 다"]:
            st.subheader("🤖 모델 신호")
            if signals_model_path.exists():
                signals_df = load_csv_file(signals_model_path)
                if signals_df is not None:
                    # 날짜/티커 필터 적용
                    if 'date' in signals_df.columns:
                        signals_df['date'] = pd.to_datetime(signals_df['date'])
                        mask = (signals_df['date'] >= pd.to_datetime(start_date)) & (signals_df['date'] <= pd.to_datetime(end_date))
                        signals_df = signals_df[mask]
                    
                    if 'ticker' in signals_df.columns and selected_tickers:
                        signals_df = signals_df[signals_df['ticker'].isin(selected_tickers)]
                    
                    # 임계값 필터 적용
                    if 'probability' in signals_df.columns:
                        signals_df = signals_df[signals_df['probability'] >= threshold]
                    
                    # 정렬 및 표시
                    signals_df = signals_df.sort_values(['date', 'ticker'], ascending=[False, True])
                    
                    st.dataframe(signals_df, use_container_width=True)
                    
                    # CSV 다운로드
                    if st.button("🔽 모델 신호 CSV 다운로드"):
                        csv_link = get_csv_download_link(signals_df, f"signals_model_{selected_run}.csv")
                        st.markdown(csv_link, unsafe_allow_html=True)
                else:
                    st.error("모델 신호 데이터를 로드할 수 없습니다.")
            else:
                st.warning("signals_model.csv 파일이 없습니다.")
        
        if strategy_mode in ["룰 기반", "둘 다"]:
            st.subheader("📏 룰 기반 신호")
            if signals_rule_path.exists():
                signals_rule_df = load_csv_file(signals_rule_path)
                if signals_rule_df is not None:
                    # 날짜/티커 필터 적용
                    if 'date' in signals_rule_df.columns:
                        signals_rule_df['date'] = pd.to_datetime(signals_rule_df['date'])
                        mask = (signals_rule_df['date'] >= pd.to_datetime(start_date)) & (signals_rule_df['date'] <= pd.to_datetime(end_date))
                        signals_rule_df = signals_rule_df[mask]
                    
                    if 'ticker' in signals_rule_df.columns and selected_tickers:
                        signals_rule_df = signals_rule_df[signals_rule_df['ticker'].isin(selected_tickers)]
                    
                    # 정렬 및 표시
                    signals_rule_df = signals_rule_df.sort_values(['date', 'ticker'], ascending=[False, True])
                    
                    st.dataframe(signals_rule_df, use_container_width=True)
                    
                    # CSV 다운로드
                    if st.button("🔽 룰 신호 CSV 다운로드"):
                        csv_link = get_csv_download_link(signals_rule_df, f"signals_rule_{selected_run}.csv")
                        st.markdown(csv_link, unsafe_allow_html=True)
                else:
                    st.error("룰 기반 신호 데이터를 로드할 수 없습니다.")
            else:
                st.warning("signals_rule.csv 파일이 없습니다.")
    
    # 4. 트레이드 탭
    with tab4:
        st.subheader("💰 거래 내역")
        
        trades_path = run_path / "trades.csv"
        if trades_path.exists():
            trades_df = load_csv_file(trades_path)
            if trades_df is not None:
                # 날짜/티커 필터 적용
                if 'date' in trades_df.columns:
                    trades_df['date'] = pd.to_datetime(trades_df['date'])
                    mask = (trades_df['date'] >= pd.to_datetime(start_date)) & (trades_df['date'] <= pd.to_datetime(end_date))
                    trades_df = trades_df[mask]
                
                if 'ticker' in trades_df.columns and selected_tickers:
                    trades_df = trades_df[trades_df['ticker'].isin(selected_tickers)]
                
                # KPI 계산
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0
                    st.metric("💰 누적 PnL", f"{total_pnl:,.0f}원")
                
                with col2:
                    if 'pnl' in trades_df.columns:
                        win_rate = (trades_df['pnl'] > 0).mean() * 100
                        st.metric("🎯 승률", f"{win_rate:.1f}%")
                    else:
                        st.metric("🎯 승률", "N/A")
                
                with col3:
                    total_trades = len(trades_df)
                    st.metric("📊 총 거래수", f"{total_trades:,}")
                
                # 거래 내역 표
                st.dataframe(trades_df, use_container_width=True)
                
                # CSV 다운로드
                if st.button("🔽 거래 내역 CSV 다운로드"):
                    csv_link = get_csv_download_link(trades_df, f"trades_{selected_run}.csv")
                    st.markdown(csv_link, unsafe_allow_html=True)
            else:
                st.error("거래 내역 데이터를 로드할 수 없습니다.")
        else:
            st.warning("trades.csv 파일이 없습니다.")
    
    # 5. 클러스터링 탭
    with tab5:
        st.subheader("🔍 클러스터링 분석")
        
        # 클러스터링 결과 파일들 확인
        clustering_files = list(run_path.glob("*clustering*"))
        
        if clustering_files:
            st.info("클러스터링 결과 파일들을 찾았습니다.")
            for file in clustering_files:
                st.write(f"📁 {file.name}")
                
                if file.suffix == '.csv':
                    df = load_csv_file(file)
                    if df is not None:
                        st.dataframe(df.head())
                elif file.suffix == '.json':
                    data = load_json_file(file)
                    if data:
                        st.json(data)
        else:
            st.warning("클러스터링 결과 파일이 없습니다.")
            st.info("clustering.py를 실행하여 클러스터링 분석을 수행하세요.")
    
    # 6. 리포트 탭
    with tab6:
        st.subheader("📄 리포트")
        
        # reports/ 디렉토리에서 최신 summary 파일 찾기
        reports_dir = Path("reports")
        summary_files = list(reports_dir.glob("summary_*.md"))
        
        if summary_files:
            # 최신 파일 선택
            latest_summary = max(summary_files, key=os.path.getctime)
            
            st.subheader(f"📄 {latest_summary.name}")
            
            # 마크다운 파일 읽기 및 표시
            try:
                with open(latest_summary, 'r', encoding='utf-8') as f:
                    content = f.read()
                st.markdown(content)
                
                # 다운로드 버튼
                if st.button("🔽 리포트 다운로드"):
                    with open(latest_summary, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    st.download_button(
                        label="📄 마크다운 다운로드",
                        data=file_content,
                        file_name=latest_summary.name,
                        mime="text/markdown"
                    )
            except Exception as e:
                st.error(f"리포트 파일을 읽을 수 없습니다: {str(e)}")
        else:
            st.warning("summary_*.md 파일이 없습니다.")
            st.info("리포트를 생성하려면 main.py를 실행하세요.")

# 메인 실행
if __name__ == "__main__":
    main()

