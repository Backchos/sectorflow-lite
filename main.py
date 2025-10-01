#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Main Execution Script
전체 파이프라인 실행 및 통합

Usage:
    python main.py --mode [full|data|train|predict|backtest|report|cluster]
"""

import argparse
import sys
import os
import json
import hashlib
import random
import numpy as np
from datetime import datetime
import yaml
import warnings
warnings.filterwarnings('ignore')

# 시드 설정 함수
def set_seeds(seed: int = 42):
    """모든 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    
    # TensorFlow 시드 설정
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass
    
    # PyTorch 시드 설정
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f"🌱 시드 설정 완료: {seed}")

def generate_run_id() -> str:
    """실행 ID 생성"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_hash = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
    return f"{timestamp}_{random_hash}"

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def load_config(config_path: str = "config.yaml") -> dict:
    """설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 설정 파일 로드 완료: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        return {}

def setup_run_directory(run_id: str, config: dict) -> str:
    """실행 디렉토리 설정"""
    runs_dir = f"runs/{run_id}"
    os.makedirs(runs_dir, exist_ok=True)
    
    # 설정 스냅샷 저장
    config_snapshot_path = os.path.join(runs_dir, "config_snapshot.yaml")
    with open(config_snapshot_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 환경 정보 저장
    env_info = get_environment_info()
    env_info_path = os.path.join(runs_dir, "env_info.json")
    with open(env_info_path, 'w', encoding='utf-8') as f:
        json.dump(env_info, f, indent=2, ensure_ascii=False)
    
    print(f"📁 실행 디렉토리 설정: {runs_dir}")
    return runs_dir

def get_environment_info() -> dict:
    """환경 정보 수집"""
    import platform
    import subprocess
    
    env_info = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': platform.platform(),
        'timestamp': datetime.now().isoformat()
    }
    
    # 패키지 버전 정보
    packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
    for pkg in packages:
        try:
            module = __import__(pkg)
            env_info[f'{pkg}_version'] = getattr(module, '__version__', 'N/A')
        except ImportError:
            env_info[f'{pkg}_version'] = 'N/A'
    
    # Git 정보
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()[:8]
        env_info['git_hash'] = git_hash
    except:
        env_info['git_hash'] = 'N/A'
    
    return env_info

def run_data_pipeline(config: dict) -> dict:
    """데이터 파이프라인 실행"""
    print("\n🚀 데이터 파이프라인 실행 중...")
    
    from src.dataio import prepare_ml_data
    
    # 데이터 설정
    data_config = {
        'start_date': config.get('data', {}).get('start_date', '2024-01-01'),
        'end_date': config.get('data', {}).get('end_date', '2024-12-31'),
        'lookback': config.get('model', {}).get('sequence_length', 30),
        'feature_cols': config.get('features', {}).get('feature_cols', 
            ['close', 'volume', 'trading_value', 'returns', 'ma_5', 'ma_20', 'volatility']),
        'scale_method': 'standard'
    }
    
    symbols = config.get('data', {}).get('symbols', ['005930', '000660', '035420'])
    
    # 데이터 준비
    processed_data = prepare_ml_data(symbols, data_config)
    
    print(f"✅ 데이터 파이프라인 완료: {len(processed_data)}개 종목")
    return processed_data

def run_baseline_training(processed_data: dict) -> dict:
    """베이스라인 모델 훈련"""
    print("\n🤖 베이스라인 모델 훈련 중...")
    
    from src.baseline import train_all_baselines
    
    # 베이스라인 모델 훈련
    baseline_results = train_all_baselines(processed_data)
    
    print(f"✅ 베이스라인 모델 훈련 완료: {len(baseline_results)}개 종목")
    return baseline_results

def run_deep_learning_training(processed_data: dict, config: dict) -> dict:
    """딥러닝 모델 훈련"""
    print("\n🧠 딥러닝 모델 훈련 중...")
    
    try:
        from src.train import train_all_models
        
        # 훈련 설정
        training_config = {
            'epochs': config.get('model', {}).get('epochs', 50),
            'batch_size': config.get('model', {}).get('batch_size', 32),
            'patience': 15,
            'learning_rate': config.get('model', {}).get('learning_rate', 0.001),
            'class_imbalance_method': 'class_weight'
        }
        
        # 딥러닝 모델 훈련
        dl_results = train_all_models(
            processed_data, 
            model_types=['gru', 'lstm'], 
            config=training_config
        )
        
        print(f"✅ 딥러닝 모델 훈련 완료: {len(dl_results)}개 종목")
        return dl_results
        
    except ImportError as e:
        print(f"⚠️ 딥러닝 모델 훈련 건너뜀: {e}")
        return {}

def run_backtesting(processed_data: dict, model_results: dict, config: dict) -> dict:
    """백테스팅 실행"""
    print("\n💰 백테스팅 실행 중...")
    
    from src.backtest import run_backtest, run_model_backtest
    from src.rules import generate_trading_signals
    from src.features import process_features
    
    backtest_results = {}
    
    for symbol, data in processed_data.items():
        print(f"   📊 {symbol} 백테스팅 중...")
        
        # 원본 데이터 가져오기
        df = data['original_df'].copy()
        
        # 1. 룰 기반 백테스트
        try:
            # 피처 계산
            df_with_features = process_features(df)
            
            # 신호 생성
            df_with_signals = generate_trading_signals(df_with_features)
            
            # 백테스트 실행
            rule_results = run_backtest(df_with_signals)
            backtest_results[f'{symbol}_rule_based'] = rule_results
            
        except Exception as e:
            print(f"   ❌ {symbol} 룰 기반 백테스트 실패: {e}")
        
        # 2. 모델 기반 백테스트 (모델이 있는 경우)
        if symbol in model_results and 'models' in model_results[symbol]:
            try:
                # 테스트 데이터로 예측
                X_test = data['X_test']
                best_model = model_results[symbol]['best_model']['model']
                
                # 예측 확률 생성
                X_test_2d = X_test.reshape(X_test.shape[0], -1)
                predictions = best_model.predict_proba(X_test_2d)[:, 1]
                
                # 백테스트 실행
                model_results_backtest = run_model_backtest(df, predictions, threshold=0.5)
                backtest_results[f'{symbol}_model_based'] = model_results_backtest
                
            except Exception as e:
                print(f"   ❌ {symbol} 모델 기반 백테스트 실패: {e}")
    
    print(f"✅ 백테스팅 완료: {len(backtest_results)}개 전략")
    return backtest_results

def run_clustering(processed_data: dict) -> dict:
    """종목 클러스터링 실행"""
    print("\n🎯 종목 클러스터링 실행 중...")
    
    try:
        from src.clustering import (
            prepare_clustering_data, perform_pca, perform_kmeans,
            analyze_clusters, visualize_clusters, generate_cluster_report
        )
        from src.report_generator import save_report
        
        # 클러스터링 데이터 준비
        feature_array, symbol_names, metadata = prepare_clustering_data(processed_data)
        
        # PCA 수행
        pca_result = perform_pca(feature_array, n_components=2)
        
        # K-Means 클러스터링
        kmeans_result = perform_kmeans(pca_result['pca_result'], find_optimal=True)
        
        # 클러스터 분석
        cluster_analysis = analyze_clusters(
            symbol_names, kmeans_result['cluster_labels'], 
            feature_array, ['close', 'volume', 'trading_value', 'returns', 'ma_5', 'ma_20', 'volatility'], 
            metadata
        )
        
        # 시각화
        chart_path = visualize_clusters(
            pca_result['pca_result'], 
            kmeans_result['cluster_labels'], 
            symbol_names
        )
        
        # 리포트 생성
        report = generate_cluster_report(cluster_analysis, pca_result, kmeans_result, symbol_names)
        report_path = save_report(report, "clustering_report.md")
        
        clustering_results = {
            'cluster_analysis': cluster_analysis,
            'pca_result': pca_result,
            'kmeans_result': kmeans_result,
            'chart_path': chart_path,
            'report_path': report_path
        }
        
        print(f"✅ 클러스터링 완료: {cluster_analysis['total_clusters']}개 클러스터")
        return clustering_results
        
    except Exception as e:
        print(f"❌ 클러스터링 실패: {e}")
        return {}

def generate_comprehensive_report(processed_data: dict, 
                                baseline_results: dict,
                                dl_results: dict,
                                backtest_results: dict,
                                clustering_results: dict,
                                config: dict) -> str:
    """종합 리포트 생성"""
    print("\n📊 종합 리포트 생성 중...")
    
    try:
        from src.report_generator import generate_summary_report
        
        # 데이터 요약 생성
        data_summary = {
            'total_symbols': len(processed_data),
            'symbols': list(processed_data.keys()),
            'data_info': {}
        }
        
        for symbol, data in processed_data.items():
            data_summary['data_info'][symbol] = {
                'train_samples': len(data['X_train']),
                'valid_samples': len(data['X_valid']),
                'test_samples': len(data['X_test']),
                'feature_shape': data['X_train'].shape[1:],
                'positive_ratio': data['y_train'].mean()
            }
        
        # 모델 결과 통합
        all_model_results = {}
        all_model_results.update(baseline_results)
        all_model_results.update(dl_results)
        
        # 리포트 생성
        report_path = generate_summary_report(
            data_summary,
            all_model_results,
            backtest_results,
            config,
            include_charts=True
        )
        
        print(f"✅ 종합 리포트 생성 완료: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"❌ 리포트 생성 실패: {e}")
        return ""

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='SectorFlow Lite - AI Trading System')
    parser.add_argument('--mode', choices=['full', 'data', 'train', 'predict', 'backtest', 'report', 'cluster'], 
                       default='full', help='실행 모드')
    parser.add_argument('--config', default='config.yaml', help='설정 파일 경로')
    parser.add_argument('--run_id', help='실행 ID (지정하지 않으면 자동 생성)')
    parser.add_argument('--dry_run', action='store_true', help='실제 실행 없이 예상 파일만 표시')
    
    args = parser.parse_args()
    
    print("🚀 SectorFlow Lite 시작!")
    print("=" * 60)
    print(f"실행 모드: {args.mode}")
    print(f"설정 파일: {args.config}")
    print(f"Dry Run: {args.dry_run}")
    
    # 설정 로드
    config = load_config(args.config)
    if not config:
        print("❌ 설정 파일을 로드할 수 없습니다.")
        return
    
    # 실행 ID 생성
    run_id = args.run_id or generate_run_id()
    print(f"실행 ID: {run_id}")
    
    # 시드 설정
    seed = config.get('train', {}).get('seed', 42)
    set_seeds(seed)
    
    # 실행 디렉토리 설정
    if not args.dry_run:
        runs_dir = setup_run_directory(run_id, config)
    else:
        print("🔍 Dry Run 모드: 실제 파일 생성 없이 예상 출력만 표시")
        runs_dir = f"runs/{run_id}"
    
    # 결과 저장용
    results = {'run_id': run_id}
    
    try:
        # 1. 데이터 파이프라인
        if args.mode in ['full', 'data']:
            processed_data = run_data_pipeline(config)
            results['processed_data'] = processed_data
        else:
            print("⚠️ 데이터 파이프라인 건너뜀")
            processed_data = {}
        
        # 2. 베이스라인 모델 훈련
        if args.mode in ['full', 'train'] and processed_data:
            baseline_results = run_baseline_training(processed_data)
            results['baseline_results'] = baseline_results
        else:
            baseline_results = {}
        
        # 3. 딥러닝 모델 훈련
        if args.mode in ['full', 'train'] and processed_data:
            dl_results = run_deep_learning_training(processed_data, config)
            results['dl_results'] = dl_results
        else:
            dl_results = {}
        
        # 4. 백테스팅
        if args.mode in ['full', 'backtest'] and processed_data:
            all_model_results = {**baseline_results, **dl_results}
            backtest_results = run_backtesting(processed_data, all_model_results, config)
            results['backtest_results'] = backtest_results
        else:
            backtest_results = {}
        
        # 5. 클러스터링
        if args.mode in ['full', 'cluster'] and processed_data:
            clustering_results = run_clustering(processed_data)
            results['clustering_results'] = clustering_results
        else:
            clustering_results = {}
        
        # 6. 종합 리포트 생성
        if args.mode in ['full', 'report']:
            report_path = generate_comprehensive_report(
                processed_data, baseline_results, dl_results, 
                backtest_results, clustering_results, config
            )
            results['report_path'] = report_path
        
        # 7. 실행 완료
        print("\n🎉 SectorFlow Lite 실행 완료!")
        print("=" * 60)
        
        # 결과 요약
        if 'processed_data' in results:
            print(f"📊 처리된 종목: {len(results['processed_data'])}개")
        if 'baseline_results' in results:
            print(f"🤖 베이스라인 모델: {len(results['baseline_results'])}개 종목")
        if 'dl_results' in results:
            print(f"🧠 딥러닝 모델: {len(results['dl_results'])}개 종목")
        if 'backtest_results' in results:
            print(f"💰 백테스트 전략: {len(results['backtest_results'])}개")
        if 'clustering_results' in results:
            print(f"🎯 클러스터: {results['clustering_results'].get('cluster_analysis', {}).get('total_clusters', 0)}개")
        if 'report_path' in results:
            print(f"📄 리포트: {results['report_path']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
