#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Clustering Module
종목 클러스터링 및 분석

Functions:
- prepare_clustering_data: 클러스터링용 데이터 준비
- perform_pca: PCA 차원 축소
- perform_kmeans: K-Means 클러스터링
- analyze_clusters: 클러스터 분석
- visualize_clusters: 클러스터 시각화
- generate_cluster_report: 클러스터 리포트 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def prepare_clustering_data(processed_data: Dict[str, Any],
                           feature_cols: List[str] = None) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    클러스터링용 데이터 준비
    
    Args:
        processed_data: 처리된 데이터 딕셔너리
        feature_cols: 사용할 피처 컬럼들
        
    Returns:
        (피처 배열, 종목 리스트, 메타데이터)
    """
    print("🔧 클러스터링용 데이터 준비 중...")
    
    if feature_cols is None:
        feature_cols = ['close', 'volume', 'trading_value', 'returns', 'ma_5', 'ma_20', 'volatility']
    
    # 각 종목의 평균 피처값 계산
    symbol_features = []
    symbol_names = []
    metadata = {}
    
    for symbol, data in processed_data.items():
        if 'original_df' in data:
            df = data['original_df']
            
            # 사용 가능한 피처만 선택
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if available_cols:
                # 각 피처의 평균값 계산
                feature_values = []
                for col in available_cols:
                    if col in df.columns:
                        # NaN 값 제외하고 평균 계산
                        mean_val = df[col].dropna().mean()
                        feature_values.append(mean_val if not np.isnan(mean_val) else 0)
                    else:
                        feature_values.append(0)
                
                symbol_features.append(feature_values)
                symbol_names.append(symbol)
                
                # 메타데이터 저장
                metadata[symbol] = {
                    'feature_values': feature_values,
                    'available_features': available_cols,
                    'data_length': len(df),
                    'last_date': df['date'].max() if 'date' in df.columns else None
                }
    
    # 배열로 변환
    feature_array = np.array(symbol_features)
    
    print(f"✅ 클러스터링 데이터 준비 완료!")
    print(f"   - 종목 수: {len(symbol_names)}")
    print(f"   - 피처 수: {feature_array.shape[1]}")
    print(f"   - 사용된 피처: {available_cols}")
    
    return feature_array, symbol_names, metadata

def perform_pca(feature_array: np.ndarray,
                n_components: int = 2,
                explained_variance_threshold: float = 0.95) -> Dict[str, Any]:
    """
    PCA 차원 축소 수행
    
    Args:
        feature_array: 피처 배열
        n_components: 주성분 수
        explained_variance_threshold: 설명 분산 임계값
        
    Returns:
        PCA 결과 딕셔너리
    """
    print(f"🔍 PCA 차원 축소 수행 중... (목표: {n_components}개 주성분)")
    
    # 데이터 스케일링
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_array)
    
    # PCA 수행
    pca = PCA(n_components=min(n_components, feature_array.shape[1]))
    pca_result = pca.fit_transform(feature_scaled)
    
    # 설명 분산 비율 계산
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # 충분한 설명 분산을 가진 주성분 수 찾기
    sufficient_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
    sufficient_components = min(sufficient_components, feature_array.shape[1])
    
    print(f"✅ PCA 완료!")
    print(f"   - 원본 차원: {feature_array.shape[1]}")
    print(f"   - 축소된 차원: {pca_result.shape[1]}")
    print(f"   - 설명 분산 비율: {explained_variance_ratio}")
    print(f"   - 누적 설명 분산: {cumulative_variance}")
    print(f"   - {explained_variance_threshold*100}% 설명을 위한 주성분 수: {sufficient_components}")
    
    return {
        'pca_result': pca_result,
        'pca_model': pca,
        'scaler': scaler,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'sufficient_components': sufficient_components
    }

def perform_kmeans(pca_result: np.ndarray,
                   n_clusters: int = 3,
                   max_clusters: int = 10,
                   find_optimal: bool = True) -> Dict[str, Any]:
    """
    K-Means 클러스터링 수행
    
    Args:
        pca_result: PCA 결과
        n_clusters: 클러스터 수
        max_clusters: 최대 클러스터 수 (최적 클러스터 수 찾기용)
        find_optimal: 최적 클러스터 수 자동 탐색 여부
        
    Returns:
        K-Means 결과 딕셔너리
    """
    print(f"🎯 K-Means 클러스터링 수행 중...")
    
    if find_optimal:
        print("   - 최적 클러스터 수 탐색 중...")
        
        # 다양한 클러스터 수로 실험
        cluster_range = range(2, min(max_clusters + 1, len(pca_result)))
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(pca_result)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(pca_result, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(pca_result, cluster_labels))
        
        # 최적 클러스터 수 선택 (실루엣 점수 기준)
        optimal_k = cluster_range[np.argmax(silhouette_scores)]
        
        print(f"   - 최적 클러스터 수: {optimal_k} (실루엣 점수: {max(silhouette_scores):.3f})")
        
        # 최적 클러스터 수로 최종 모델 훈련
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        final_labels = final_kmeans.fit_predict(pca_result)
        
        n_clusters = optimal_k
    else:
        # 지정된 클러스터 수로 훈련
        final_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        final_labels = final_kmeans.fit_predict(pca_result)
        
        inertias = [final_kmeans.inertia_]
        silhouette_scores = [silhouette_score(pca_result, final_labels)]
        calinski_scores = [calinski_harabasz_score(pca_result, final_labels)]
    
    print(f"✅ K-Means 클러스터링 완료!")
    print(f"   - 클러스터 수: {n_clusters}")
    print(f"   - 실루엣 점수: {silhouette_scores[-1]:.3f}")
    print(f"   - Calinski-Harabasz 점수: {calinski_scores[-1]:.3f}")
    
    return {
        'cluster_labels': final_labels,
        'kmeans_model': final_kmeans,
        'n_clusters': n_clusters,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'cluster_centers': final_kmeans.cluster_centers_
    }

def analyze_clusters(symbol_names: List[str],
                    cluster_labels: np.ndarray,
                    feature_array: np.ndarray,
                    feature_cols: List[str],
                    metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    클러스터 분석
    
    Args:
        symbol_names: 종목 리스트
        cluster_labels: 클러스터 라벨
        feature_array: 피처 배열
        feature_cols: 피처 컬럼명
        metadata: 메타데이터
        
    Returns:
        클러스터 분석 결과
    """
    print("📊 클러스터 분석 중...")
    
    # 클러스터별 종목 그룹화
    cluster_analysis = {}
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_symbols = [symbol_names[i] for i in np.where(cluster_mask)[0]]
        cluster_features = feature_array[cluster_mask]
        
        # 클러스터 통계
        cluster_stats = {
            'symbols': cluster_symbols,
            'count': len(cluster_symbols),
            'avg_features': np.mean(cluster_features, axis=0),
            'std_features': np.std(cluster_features, axis=0)
        }
        
        # 각 피처별 평균값
        feature_means = {}
        for i, col in enumerate(feature_cols):
            if i < len(cluster_stats['avg_features']):
                feature_means[col] = cluster_stats['avg_features'][i]
        
        cluster_stats['feature_means'] = feature_means
        
        # 클러스터 특성 분석
        cluster_characteristics = analyze_cluster_characteristics(
            cluster_symbols, feature_means, metadata
        )
        cluster_stats['characteristics'] = cluster_characteristics
        
        cluster_analysis[f'cluster_{cluster_id}'] = cluster_stats
    
    # 전체 클러스터 요약
    total_analysis = {
        'cluster_analysis': cluster_analysis,
        'total_clusters': len(np.unique(cluster_labels)),
        'total_symbols': len(symbol_names),
        'cluster_distribution': {
            f'cluster_{i}': np.sum(cluster_labels == i) 
            for i in np.unique(cluster_labels)
        }
    }
    
    print("✅ 클러스터 분석 완료!")
    for cluster_id, stats in cluster_analysis.items():
        print(f"   - {cluster_id}: {stats['count']}개 종목")
    
    return total_analysis

def analyze_cluster_characteristics(symbols: List[str],
                                  feature_means: Dict[str, float],
                                  metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    클러스터 특성 분석
    
    Args:
        symbols: 클러스터 내 종목들
        feature_means: 피처 평균값
        metadata: 메타데이터
        
    Returns:
        클러스터 특성 설명
    """
    characteristics = []
    
    # 가격 특성
    if 'close' in feature_means:
        avg_price = feature_means['close']
        if avg_price > 100000:
            characteristics.append("고가주")
        elif avg_price > 50000:
            characteristics.append("중가주")
        else:
            characteristics.append("저가주")
    
    # 거래량 특성
    if 'volume' in feature_means:
        avg_volume = feature_means['volume']
        if avg_volume > 10000000:
            characteristics.append("고거래량")
        elif avg_volume > 1000000:
            characteristics.append("중거래량")
        else:
            characteristics.append("저거래량")
    
    # 변동성 특성
    if 'volatility' in feature_means:
        avg_volatility = feature_means['volatility']
        if avg_volatility > 0.03:
            characteristics.append("고변동성")
        elif avg_volatility > 0.02:
            characteristics.append("중변동성")
        else:
            characteristics.append("저변동성")
    
    # 수익률 특성
    if 'returns' in feature_means:
        avg_returns = feature_means['returns']
        if avg_returns > 0.01:
            characteristics.append("상승추세")
        elif avg_returns < -0.01:
            characteristics.append("하락추세")
        else:
            characteristics.append("횡보")
    
    return {
        'description': ", ".join(characteristics) if characteristics else "특성 불명",
        'characteristics': characteristics
    }

def visualize_clusters(pca_result: np.ndarray,
                      cluster_labels: np.ndarray,
                      symbol_names: List[str],
                      save_path: str = None) -> str:
    """
    클러스터 시각화
    
    Args:
        pca_result: PCA 결과
        cluster_labels: 클러스터 라벨
        symbol_names: 종목 리스트
        save_path: 저장 경로
        
    Returns:
        저장된 파일 경로
    """
    print("📈 클러스터 시각화 중...")
    
    # 2D 시각화
    plt.figure(figsize=(12, 8))
    
    # 클러스터별로 색상 다르게 표시
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        plt.scatter(
            pca_result[cluster_mask, 0], 
            pca_result[cluster_mask, 1],
            c=[colors[i]], 
            label=f'Cluster {cluster_id}',
            alpha=0.7,
            s=100
        )
        
        # 종목명 표시
        for j, symbol in enumerate(symbol_names):
            if cluster_mask[j]:
                plt.annotate(
                    symbol, 
                    (pca_result[j, 0], pca_result[j, 1]),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Stock Clustering Visualization (PCA + K-Means)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 저장
    if save_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"reports/charts/cluster_visualization_{timestamp}.png"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 클러스터 시각화 저장 완료: {save_path}")
    return save_path

def generate_cluster_report(cluster_analysis: Dict[str, Any],
                          pca_result: Dict[str, Any],
                          kmeans_result: Dict[str, Any],
                          symbol_names: List[str]) -> str:
    """
    클러스터 리포트 생성
    
    Args:
        cluster_analysis: 클러스터 분석 결과
        pca_result: PCA 결과
        kmeans_result: K-Means 결과
        symbol_names: 종목 리스트
        
    Returns:
        리포트 마크다운 문자열
    """
    print("📄 클러스터 리포트 생성 중...")
    
    report = f"""# SectorFlow Lite - Stock Clustering Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Executive Summary

### 클러스터링 결과
- **총 종목 수:** {cluster_analysis['total_symbols']}개
- **클러스터 수:** {cluster_analysis['total_clusters']}개
- **PCA 설명 분산:** {pca_result['cumulative_variance'][-1]:.1%}
- **실루엣 점수:** {kmeans_result['silhouette_scores'][-1]:.3f}

### 클러스터 분포
"""
    
    # 클러스터 분포 테이블
    for cluster_id, count in cluster_analysis['cluster_distribution'].items():
        percentage = count / cluster_analysis['total_symbols'] * 100
        report += f"- **{cluster_id}:** {count}개 종목 ({percentage:.1f}%)\n"
    
    # 각 클러스터 상세 분석
    report += f"""

---

## 🔍 Detailed Cluster Analysis

"""
    
    for cluster_id, analysis in cluster_analysis['cluster_analysis'].items():
        report += f"""
### {cluster_id.replace('_', ' ').title()}

**종목 수:** {analysis['count']}개  
**특성:** {analysis['characteristics']['description']}

**포함 종목:**
{', '.join(analysis['symbols'])}

**주요 피처 평균값:**
"""
        
        for feature, value in analysis['feature_means'].items():
            if isinstance(value, (int, float)):
                report += f"- **{feature}:** {value:,.2f}\n"
        
        report += "\n"
    
    # PCA 분석
    report += f"""
---

## 📈 PCA Analysis

### 설명 분산 비율
"""
    
    for i, ratio in enumerate(pca_result['explained_variance_ratio']):
        report += f"- **PC{i+1}:** {ratio:.1%}\n"
    
    report += f"""
### 누적 설명 분산
"""
    
    for i, cum_var in enumerate(pca_result['cumulative_variance']):
        report += f"- **PC{i+1}까지:** {cum_var:.1%}\n"
    
    # 클러스터링 품질 지표
    report += f"""
---

## 🎯 Clustering Quality Metrics

- **실루엣 점수:** {kmeans_result['silhouette_scores'][-1]:.3f}
- **Calinski-Harabasz 점수:** {kmeans_result['calinski_scores'][-1]:.3f}
- **클러스터 중심점 수:** {kmeans_result['n_clusters']}개

### 실루엣 점수 해석
"""
    
    silhouette_score = kmeans_result['silhouette_scores'][-1]
    if silhouette_score > 0.7:
        report += "✅ **우수한 클러스터링** (0.7 이상)"
    elif silhouette_score > 0.5:
        report += "✅ **적절한 클러스터링** (0.5-0.7)"
    elif silhouette_score > 0.3:
        report += "⚠️ **보통 클러스터링** (0.3-0.5)"
    else:
        report += "❌ **부적절한 클러스터링** (0.3 미만)"
    
    report += f"""

---

## 💡 Investment Insights

### 클러스터별 투자 전략 제안
"""
    
    for cluster_id, analysis in cluster_analysis['cluster_analysis'].items():
        characteristics = analysis['characteristics']['characteristics']
        
        report += f"""
#### {cluster_id.replace('_', ' ').title()}
- **특성:** {', '.join(characteristics)}
- **투자 전략:** """
        
        if '고변동성' in characteristics:
            report += "단기 트레이딩, 높은 리스크-수익"
        elif '저변동성' in characteristics:
            report += "안정적 장기 투자"
        elif '고거래량' in characteristics:
            report += "유동성 높은 종목, 적극적 거래"
        else:
            report += "균형 잡힌 포트폴리오 구성"
        
        report += f"\n- **권장 비중:** {analysis['count']/cluster_analysis['total_symbols']*100:.1f}%\n"
    
    report += f"""

---

## 📝 Notes
- 이 클러스터링은 과거 데이터를 기반으로 한 분석입니다.
- 시장 상황 변화에 따라 클러스터 특성이 변할 수 있습니다.
- 투자 결정 시 추가적인 펀더멘털 분석이 필요합니다.

**Report Generated by:** SectorFlow Lite v1.0
"""
    
    print("✅ 클러스터 리포트 생성 완료!")
    return report

def main():
    """테스트용 메인 함수"""
    print("🚀 SectorFlow Lite - Clustering Module 테스트")
    print("=" * 60)
    
    # 테스트용 데이터 생성
    np.random.seed(42)
    
    # 가상의 종목 데이터 생성
    symbols = ['005930', '000660', '035420', '005380', '006400', '000270', '035720', '207940']
    n_symbols = len(symbols)
    n_features = 7
    
    # 피처 데이터 생성 (각 종목별로 다른 특성)
    feature_data = []
    for i, symbol in enumerate(symbols):
        # 종목별로 다른 특성 부여
        base_features = np.random.normal(0, 1, n_features)
        
        # 특정 종목들에 특성 부여
        if i < 3:  # 첫 3개 종목은 고가주
            base_features[0] += 2  # close (가격)
        elif i < 5:  # 다음 2개 종목은 고거래량
            base_features[1] += 2  # volume
        else:  # 나머지는 고변동성
            base_features[6] += 2  # volatility
        
        feature_data.append(base_features)
    
    feature_array = np.array(feature_data)
    feature_cols = ['close', 'volume', 'trading_value', 'returns', 'ma_5', 'ma_20', 'volatility']
    
    # 메타데이터 생성
    metadata = {}
    for i, symbol in enumerate(symbols):
        metadata[symbol] = {
            'feature_values': feature_data[i],
            'available_features': feature_cols,
            'data_length': 100,
            'last_date': '2024-12-31'
        }
    
    print(f"📊 테스트 데이터 생성 완료: {n_symbols}개 종목, {n_features}개 피처")
    
    # 1. PCA 수행
    pca_result = perform_pca(feature_array, n_components=2)
    
    # 2. K-Means 클러스터링
    kmeans_result = perform_kmeans(pca_result['pca_result'], find_optimal=True)
    
    # 3. 클러스터 분석
    cluster_analysis = analyze_clusters(
        symbols, 
        kmeans_result['cluster_labels'], 
        feature_array, 
        feature_cols, 
        metadata
    )
    
    # 4. 시각화
    chart_path = visualize_clusters(
        pca_result['pca_result'], 
        kmeans_result['cluster_labels'], 
        symbols
    )
    
    # 5. 리포트 생성
    report = generate_cluster_report(cluster_analysis, pca_result, kmeans_result, symbols)
    
    # 6. 리포트 저장
    from report_generator import save_report
    report_path = save_report(report, "clustering_report.md")
    
    print(f"\n📄 생성된 리포트: {report_path}")
    print(f"📈 생성된 차트: {chart_path}")
    print("\n✅ Clustering Module 테스트 완료!")
    
    return {
        'cluster_analysis': cluster_analysis,
        'pca_result': pca_result,
        'kmeans_result': kmeans_result,
        'report_path': report_path,
        'chart_path': chart_path
    }

if __name__ == "__main__":
    results = main()





