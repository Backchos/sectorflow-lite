#!/usr/bin/env python3
"""
SectorFlow Lite - Streamlit Dashboard
실시간 주식 분석 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# 페이지 설정
st.set_page_config(
    page_title="SectorFlow Lite Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메인 타이틀
st.title("🚀 SectorFlow Lite Dashboard")
st.markdown("---")

# 사이드바
st.sidebar.title("📋 설정")

# 종목 선택
st.sidebar.subheader("📈 종목 선택")
ticker = st.sidebar.selectbox(
    "종목을 선택하세요:",
    ["005930.KS", "000660.KS", "035420.KS", "207940.KS", "006400.KS"],
    help="KOSPI 주요 종목"
)

# 기간 선택
st.sidebar.subheader("📅 분석 기간")
period = st.sidebar.selectbox(
    "기간을 선택하세요:",
    ["1M", "3M", "6M", "1Y", "2Y"],
    index=2
)

# 분석 모드
st.sidebar.subheader("🔍 분석 모드")
analysis_mode = st.sidebar.radio(
    "분석 모드를 선택하세요:",
    ["기본 분석", "AI 예측", "백테스팅", "포트폴리오"]
)

# 메인 컨텐츠
if analysis_mode == "기본 분석":
    st.header("📊 기본 분석")
    
    # 데이터 로딩
    with st.spinner("데이터를 불러오는 중..."):
        try:
            data = yf.download(ticker, period=period, progress=False)
            if data.empty:
                st.error("데이터를 불러올 수 없습니다.")
            else:
                # 기본 정보
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("현재가", f"{data['Close'].iloc[-1]:,.0f}원")
                
                with col2:
                    change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    st.metric("변동", f"{change:+,.0f}원")
                
                with col3:
                    change_pct = (change / data['Close'].iloc[-2]) * 100
                    st.metric("변동률", f"{change_pct:+.2f}%")
                
                with col4:
                    volume = data['Volume'].iloc[-1]
                    st.metric("거래량", f"{volume:,}")
                
                # 차트
                st.subheader("📈 가격 차트")
                
                # 캔들스틱 차트
                fig = go.Figure(data=go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=ticker
                ))
                
                fig.update_layout(
                    title=f"{ticker} 가격 차트",
                    xaxis_title="날짜",
                    yaxis_title="가격 (원)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 거래량 차트
                st.subheader("📊 거래량")
                fig_volume = px.bar(
                    x=data.index,
                    y=data['Volume'],
                    title="거래량 차트"
                )
                st.plotly_chart(fig_volume, use_container_width=True)
                
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")

elif analysis_mode == "AI 예측":
    st.header("🤖 AI 예측")
    
    st.info("AI 예측 기능은 개발 중입니다.")
    
    # 시뮬레이션 데이터
    st.subheader("📈 예측 결과 (시뮬레이션)")
    
    # 가상의 예측 데이터 생성
    dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
    current_price = 50000
    predictions = []
    
    for i in range(30):
        # 랜덤 워크 시뮬레이션
        change = np.random.normal(0, 0.02)
        current_price *= (1 + change)
        predictions.append(current_price)
    
    pred_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Price': predictions
    })
    
    fig = px.line(
        pred_df,
        x='Date',
        y='Predicted_Price',
        title="AI 가격 예측 (30일)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 예측 요약
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("30일 후 예상가", f"{predictions[-1]:,.0f}원")
    
    with col2:
        total_change = (predictions[-1] - 50000) / 50000 * 100
        st.metric("예상 수익률", f"{total_change:+.2f}%")

elif analysis_mode == "백테스팅":
    st.header("📊 백테스팅")
    
    st.info("백테스팅 기능은 개발 중입니다.")
    
    # 시뮬레이션 백테스팅 결과
    st.subheader("📈 백테스팅 결과 (시뮬레이션)")
    
    # 가상의 백테스팅 데이터
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), periods=365, freq='D')
    returns = np.random.normal(0.001, 0.02, 365)
    cumulative_returns = np.cumprod(1 + returns) * 10000
    
    backtest_df = pd.DataFrame({
        'Date': dates,
        'Portfolio_Value': cumulative_returns
    })
    
    fig = px.line(
        backtest_df,
        x='Date',
        y='Portfolio_Value',
        title="포트폴리오 가치 변화"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 성과 지표
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 수익률", f"{((cumulative_returns[-1] - 10000) / 10000 * 100):.2f}%")
    
    with col2:
        st.metric("연간 수익률", f"{(np.mean(returns) * 252 * 100):.2f}%")
    
    with col3:
        st.metric("변동성", f"{(np.std(returns) * np.sqrt(252) * 100):.2f}%")
    
    with col4:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        st.metric("샤프 비율", f"{sharpe:.2f}")

elif analysis_mode == "포트폴리오":
    st.header("📋 포트폴리오 관리")
    
    st.info("포트폴리오 관리 기능은 개발 중입니다.")
    
    # 포트폴리오 구성
    st.subheader("📊 포트폴리오 구성")
    
    # 가상의 포트폴리오 데이터
    portfolio_data = {
        '종목': ['삼성전자', 'SK하이닉스', 'NAVER', '카카오', 'LG화학'],
        '비중': [30, 25, 20, 15, 10],
        '수익률': [5.2, -2.1, 8.3, 12.1, -1.5]
    }
    
    portfolio_df = pd.DataFrame(portfolio_data)
    
    # 포트폴리오 테이블
    st.dataframe(portfolio_df, use_container_width=True)
    
    # 파이 차트
    fig_pie = px.pie(
        portfolio_df,
        values='비중',
        names='종목',
        title="포트폴리오 비중"
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # 성과 요약
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("총 자산", "1,000,000원")
    
    with col2:
        st.metric("총 수익", "45,000원")
    
    with col3:
        st.metric("수익률", "4.5%")

# 푸터
st.markdown("---")
st.markdown("**SectorFlow Lite** - AI 기반 주식 분석 플랫폼")
st.markdown("📧 문의: qortls510@gmail.com | 🔗 GitHub: [Backchos/sectorflow-lite](https://github.com/Backchos/sectorflow-lite)")
