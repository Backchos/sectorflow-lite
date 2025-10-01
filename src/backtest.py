#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - Backtest Module
백테스팅 모듈

Functions:
- run_backtest: 백테스팅 실행
- calculate_returns: 수익률 계산
- calculate_metrics: 성과 지표 계산
- generate_report: 백테스트 리포트 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def run_backtest(df: pd.DataFrame, 
                config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    백테스팅 실행
    
    전략: 오늘 종가에 매수 → 익일 시가 매도
    거래비용: 수수료 + 슬리피지 반영
    
    Args:
        df: 신호가 포함된 DataFrame
        config: 백테스트 설정 딕셔너리
        
    Returns:
        백테스트 결과 딕셔너리
    """
    if config is None:
        config = {
            'commission_rate': 0.003,  # 0.3% 수수료 (호환성)
            'initial_capital': 1000000,  # 초기 자본 100만원
            'position_size': 1.0,  # 포지션 크기 (전체 자본 대비)
            'close_col': 'close',
            'open_col': 'open',
            'signal_col': 'trading_signal'
        }
    
    # 거래비용 설정 (trading 섹션에서 가져오기)
    trading_config = config.get('trading', {})
    fee_bps = trading_config.get('fee_bps', 30)  # 30 bps = 0.3%
    slippage_bps = trading_config.get('slippage_bps', 10)  # 10 bps = 0.1%
    
    # bps를 비율로 변환
    commission_rate = fee_bps / 10000
    slippage_rate = slippage_bps / 10000
    
    # 기존 commission_rate가 있으면 사용 (호환성)
    if 'commission_rate' in config:
        commission_rate = config['commission_rate']
    
    # 총 거래비용 = 수수료 + 슬리피지
    total_cost_rate = commission_rate + slippage_rate
    
    df = df.copy()
    
    print("🚀 백테스팅 시작...")
    print(f"   - 초기 자본: {config['initial_capital']:,}원")
    print(f"   - 수수료율: {commission_rate*100:.1f}%")
    print(f"   - 슬리피지율: {slippage_rate*100:.1f}%")
    print(f"   - 총 거래비용: {total_cost_rate*100:.1f}%")
    
    # 필수 컬럼 확인
    required_cols = [config['close_col'], config['open_col'], config['signal_col']]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")
    
    # 백테스트 실행 (거래비용 포함)
    results = execute_trading_strategy(df, config, total_cost_rate)
    
    # 성과 지표 계산
    metrics = calculate_metrics(results, config)
    
    # 결과 정리
    backtest_results = {
        'trades': results['trades'],
        'portfolio_values': results['portfolio_values'],
        'metrics': metrics,
        'config': config,
        'trading_costs': {
            'commission_rate': commission_rate,
            'slippage_rate': slippage_rate,
            'total_cost_rate': total_cost_rate
        }
    }
    
    print("✅ 백테스팅 완료!")
    print(f"   - 총 거래 수: {len(results['trades'])}회")
    print(f"   - 최종 수익률: {metrics['total_return']*100:.2f}%")
    print(f"   - 최대 낙폭: {metrics['max_drawdown']*100:.2f}%")
    
    return backtest_results

def execute_trading_strategy(df: pd.DataFrame, config: Dict[str, Any], total_cost_rate: float = 0.003) -> Dict[str, Any]:
    """
    거래 전략 실행
    
    Args:
        df: 신호가 포함된 DataFrame
        config: 백테스트 설정
        
    Returns:
        거래 결과 딕셔너리
    """
    trades = []
    portfolio_values = []
    
    capital = config['initial_capital']
    position = 0  # 0: 없음, 1: 보유
    entry_price = 0
    entry_date = None
    
    close_col = config['close_col']
    open_col = config['open_col']
    signal_col = config['signal_col']
    commission_rate = total_cost_rate  # 총 거래비용 사용
    
    for i, row in df.iterrows():
        current_date = row.get('date', i)
        current_close = row[close_col]
        current_open = row[open_col]
        current_signal = row[signal_col]
        
        # 포트폴리오 가치 계산
        if position == 1:
            portfolio_value = capital + (current_close - entry_price) * (capital / entry_price)
        else:
            portfolio_value = capital
        
        portfolio_values.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'position': position,
            'price': current_close
        })
        
        # 거래 실행
        if current_signal == 'BUY' and position == 0:
            # 매수
            entry_price = current_close
            entry_date = current_date
            position = 1
            
            # 수수료 차감
            commission = capital * commission_rate
            capital -= commission
            
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'price': entry_price,
                'quantity': capital / entry_price,
                'commission': commission,
                'capital_after': capital
            })
            
        elif current_signal in ['HOLD', 'SELL'] and position == 1:
            # 매도 (익일 시가)
            if i + 1 < len(df):
                exit_price = df.iloc[i + 1][open_col]
                exit_date = df.iloc[i + 1].get('date', i + 1)
            else:
                # 마지막 날이면 종가로 매도
                exit_price = current_close
                exit_date = current_date
            
            # 수익 계산
            quantity = capital / entry_price
            gross_profit = (exit_price - entry_price) * quantity
            commission = capital * commission_rate
            net_profit = gross_profit - commission
            
            # 자본 업데이트
            capital += net_profit
            position = 0
            
            trades.append({
                'date': exit_date,
                'action': 'SELL',
                'price': exit_price,
                'quantity': quantity,
                'gross_profit': gross_profit,
                'commission': commission,
                'net_profit': net_profit,
                'capital_after': capital
            })
    
    return {
        'trades': trades,
        'portfolio_values': portfolio_values,
        'final_capital': capital
    }

def calculate_metrics(results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    성과 지표 계산
    
    Args:
        results: 백테스트 결과
        config: 백테스트 설정
        
    Returns:
        성과 지표 딕셔너리
    """
    trades = results['trades']
    portfolio_values = results['portfolio_values']
    final_capital = results['final_capital']
    initial_capital = config['initial_capital']
    
    # 기본 수익률
    total_return = (final_capital - initial_capital) / initial_capital
    
    # 거래 통계
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    
    total_trades = len(sell_trades)
    winning_trades = len([t for t in sell_trades if t['net_profit'] > 0])
    losing_trades = len([t for t in sell_trades if t['net_profit'] < 0])
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 수익률 통계
    if sell_trades:
        profits = [t['net_profit'] for t in sell_trades]
        avg_profit = np.mean(profits)
        max_profit = np.max(profits)
        max_loss = np.min(profits)
    else:
        avg_profit = max_profit = max_loss = 0
    
    # 포트폴리오 가치 시리즈
    portfolio_df = pd.DataFrame(portfolio_values)
    portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
    
    # 최대 낙폭 계산
    portfolio_df['cumulative'] = (1 + portfolio_df['returns'].fillna(0)).cumprod()
    portfolio_df['running_max'] = portfolio_df['cumulative'].expanding().max()
    portfolio_df['drawdown'] = (portfolio_df['cumulative'] - portfolio_df['running_max']) / portfolio_df['running_max']
    max_drawdown = portfolio_df['drawdown'].min()
    
    # 샤프 비율 계산
    if len(portfolio_df) > 1:
        returns = portfolio_df['returns'].dropna()
        if len(returns) > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
    
    # 연간화 수익률
    days = len(portfolio_df)
    annualized_return = (final_capital / initial_capital) ** (252 / days) - 1 if days > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'final_capital': final_capital,
        'total_commission': sum(t['commission'] for t in trades)
    }
    
    return metrics

def generate_report(backtest_results: Dict[str, Any]) -> str:
    """
    백테스트 리포트 생성
    
    Args:
        backtest_results: 백테스트 결과
        
    Returns:
        리포트 문자열
    """
    metrics = backtest_results['metrics']
    config = backtest_results['config']
    
    report = f"""
📊 SectorFlow Lite 백테스트 리포트
{'='*50}

💰 자본 관리
- 초기 자본: {config['initial_capital']:,}원
- 최종 자본: {metrics['final_capital']:,.0f}원
- 총 수수료: {metrics['total_commission']:,.0f}원

📈 수익률
- 총 수익률: {metrics['total_return']*100:.2f}%
- 연간화 수익률: {metrics['annualized_return']*100:.2f}%
- 최대 낙폭: {metrics['max_drawdown']*100:.2f}%
- 샤프 비율: {metrics['sharpe_ratio']:.2f}

📊 거래 통계
- 총 거래 수: {metrics['total_trades']}회
- 승률: {metrics['win_rate']*100:.1f}%
- 평균 수익: {metrics['avg_profit']:,.0f}원
- 최대 수익: {metrics['max_profit']:,.0f}원
- 최대 손실: {metrics['max_loss']:,.0f}원

{'='*50}
"""
    
    return report

def analyze_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    거래 분석
    
    Args:
        trades: 거래 리스트
        
    Returns:
        거래 분석 결과
    """
    if not trades:
        return {'error': '거래 데이터가 없습니다.'}
    
    # 매수/매도 쌍 분석
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    
    if len(buy_trades) != len(sell_trades):
        return {'error': '매수/매도 거래 수가 일치하지 않습니다.'}
    
    # 거래 쌍 생성
    trade_pairs = []
    for i in range(len(sell_trades)):
        buy_trade = buy_trades[i]
        sell_trade = sell_trades[i]
        
        trade_pairs.append({
            'buy_date': buy_trade['date'],
            'sell_date': sell_trade['date'],
            'buy_price': buy_trade['price'],
            'sell_price': sell_trade['price'],
            'net_profit': sell_trade['net_profit'],
            'return_pct': (sell_trade['price'] - buy_trade['price']) / buy_trade['price'] * 100,
            'holding_days': (pd.to_datetime(sell_trade['date']) - pd.to_datetime(buy_trade['date'])).days
        })
    
    # 분석 결과
    analysis = {
        'total_pairs': len(trade_pairs),
        'profitable_pairs': len([p for p in trade_pairs if p['net_profit'] > 0]),
        'losing_pairs': len([p for p in trade_pairs if p['net_profit'] < 0]),
        'avg_holding_days': np.mean([p['holding_days'] for p in trade_pairs]),
        'avg_return_pct': np.mean([p['return_pct'] for p in trade_pairs]),
        'max_return_pct': np.max([p['return_pct'] for p in trade_pairs]),
        'min_return_pct': np.min([p['return_pct'] for p in trade_pairs]),
        'trade_pairs': trade_pairs
    }
    
    return analysis

def run_model_backtest(df: pd.DataFrame, 
                       model_predictions: np.ndarray,
                       threshold: float = 0.5,
                       config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    모델 기반 백테스트 실행
    
    Args:
        df: OHLCV 데이터프레임
        model_predictions: 모델 예측 확률
        threshold: 분류 임계값
        config: 백테스트 설정
        
    Returns:
        백테스트 결과 딕셔너리
    """
    if config is None:
        config = {
            'commission_rate': 0.003,
            'initial_capital': 1000000,
            'position_size': 1.0,
            'close_col': 'close',
            'open_col': 'open'
        }
    
    df = df.copy()
    
    print("🤖 모델 기반 백테스트 시작...")
    print(f"   - 초기 자본: {config['initial_capital']:,}원")
    print(f"   - 수수료율: {config['commission_rate']*100:.1f}%")
    print(f"   - 분류 임계값: {threshold}")
    
    # 모델 신호 생성
    model_signals = (model_predictions > threshold).astype(int)
    df['model_signal'] = model_signals
    df['model_probability'] = model_predictions
    
    # 신호를 BUY/HOLD로 변환
    df['trading_signal'] = 'HOLD'
    df.loc[df['model_signal'] == 1, 'trading_signal'] = 'BUY'
    
    # 백테스트 실행
    results = execute_trading_strategy(df, config)
    
    # 성과 지표 계산
    metrics = calculate_metrics(results, config)
    
    # 모델 특화 지표 추가
    model_metrics = calculate_model_metrics(df, model_predictions, threshold)
    metrics.update(model_metrics)
    
    backtest_results = {
        'trades': results['trades'],
        'portfolio_values': results['portfolio_values'],
        'metrics': metrics,
        'config': config,
        'model_info': {
            'threshold': threshold,
            'predictions': model_predictions,
            'signal_counts': df['trading_signal'].value_counts().to_dict()
        }
    }
    
    print("✅ 모델 기반 백테스트 완료!")
    print(f"   - 총 거래 수: {len(results['trades'])}회")
    print(f"   - 최종 수익률: {metrics['total_return']*100:.2f}%")
    print(f"   - 최대 낙폭: {metrics['max_drawdown']*100:.2f}%")
    print(f"   - 모델 정확도: {model_metrics['model_accuracy']:.3f}")
    
    return backtest_results

def calculate_model_metrics(df: pd.DataFrame, 
                           predictions: np.ndarray, 
                           threshold: float) -> Dict[str, Any]:
    """
    모델 특화 지표 계산
    
    Args:
        df: 데이터프레임
        predictions: 모델 예측 확률
        threshold: 분류 임계값
        
    Returns:
        모델 지표 딕셔너리
    """
    # 실제 수익률 계산 (익일 종가 기준)
    df['actual_returns'] = df['close'].pct_change().shift(-1)
    df['actual_direction'] = (df['actual_returns'] > 0).astype(int)
    
    # 모델 예측
    model_predictions = (predictions > threshold).astype(int)
    
    # 정확도 계산
    valid_mask = ~df['actual_direction'].isna()
    if valid_mask.sum() > 0:
        accuracy = np.mean(model_predictions[valid_mask] == df['actual_direction'][valid_mask])
    else:
        accuracy = 0.0
    
    # 신호 통계
    buy_signals = np.sum(model_predictions == 1)
    total_signals = len(model_predictions)
    buy_ratio = buy_signals / total_signals if total_signals > 0 else 0
    
    # 신뢰도 통계
    confidence = np.abs(predictions - 0.5) * 2
    avg_confidence = np.mean(confidence)
    
    # 예측 확률 분포
    prob_stats = {
        'mean_prob': np.mean(predictions),
        'std_prob': np.std(predictions),
        'min_prob': np.min(predictions),
        'max_prob': np.max(predictions)
    }
    
    return {
        'model_accuracy': accuracy,
        'buy_signal_ratio': buy_ratio,
        'avg_confidence': avg_confidence,
        'prob_stats': prob_stats
    }

def compare_strategies(rule_results: Dict[str, Any], 
                      model_results: Dict[str, Any]) -> pd.DataFrame:
    """
    전략 성과 비교
    
    Args:
        rule_results: 룰 기반 백테스트 결과
        model_results: 모델 기반 백테스트 결과
        
    Returns:
        비교 결과 데이터프레임
    """
    comparison_data = []
    
    # 룰 기반 결과
    rule_metrics = rule_results['metrics']
    comparison_data.append({
        'Strategy': 'Rule-based',
        'Total Return': rule_metrics['total_return'],
        'Annualized Return': rule_metrics['annualized_return'],
        'Max Drawdown': rule_metrics['max_drawdown'],
        'Sharpe Ratio': rule_metrics['sharpe_ratio'],
        'Total Trades': rule_metrics['total_trades'],
        'Win Rate': rule_metrics['win_rate'],
        'Avg Profit': rule_metrics['avg_profit'],
        'Final Capital': rule_metrics['final_capital']
    })
    
    # 모델 기반 결과
    model_metrics = model_results['metrics']
    comparison_data.append({
        'Strategy': 'Model-based',
        'Total Return': model_metrics['total_return'],
        'Annualized Return': model_metrics['annualized_return'],
        'Max Drawdown': model_metrics['max_drawdown'],
        'Sharpe Ratio': model_metrics['sharpe_ratio'],
        'Total Trades': model_metrics['total_trades'],
        'Win Rate': model_metrics['win_rate'],
        'Avg Profit': model_metrics['avg_profit'],
        'Final Capital': model_metrics['final_capital']
    })
    
    # 모델 특화 지표 추가
    if 'model_accuracy' in model_metrics:
        comparison_data[1]['Model Accuracy'] = model_metrics['model_accuracy']
        comparison_data[0]['Model Accuracy'] = np.nan
    
    df_comparison = pd.DataFrame(comparison_data)
    
    return df_comparison

def generate_comparison_report(rule_results: Dict[str, Any], 
                             model_results: Dict[str, Any]) -> str:
    """
    전략 비교 리포트 생성
    
    Args:
        rule_results: 룰 기반 결과
        model_results: 모델 기반 결과
        
    Returns:
        비교 리포트 문자열
    """
    comparison_df = compare_strategies(rule_results, model_results)
    
    report = f"""
📊 SectorFlow Lite - 전략 비교 리포트
{'='*60}

🏆 전략 성과 비교
{comparison_df.to_string(index=False, float_format='%.3f')}

📈 상세 분석
"""
    
    # 수익률 비교
    rule_return = rule_results['metrics']['total_return']
    model_return = model_results['metrics']['total_return']
    return_diff = model_return - rule_return
    
    report += f"""
💰 수익률 비교
- 룰 기반: {rule_return*100:.2f}%
- 모델 기반: {model_return*100:.2f}%
- 차이: {return_diff*100:+.2f}% {'(모델 우세)' if return_diff > 0 else '(룰 기반 우세)'}
"""
    
    # 위험 지표 비교
    rule_dd = rule_results['metrics']['max_drawdown']
    model_dd = model_results['metrics']['max_drawdown']
    
    report += f"""
⚠️ 위험 지표 비교
- 룰 기반 최대 낙폭: {rule_dd*100:.2f}%
- 모델 기반 최대 낙폭: {model_dd*100:.2f}%
- 차이: {(model_dd-rule_dd)*100:+.2f}% {'(모델이 더 안전)' if model_dd < rule_dd else '(룰 기반이 더 안전)'}
"""
    
    # 거래 빈도 비교
    rule_trades = rule_results['metrics']['total_trades']
    model_trades = model_results['metrics']['total_trades']
    
    report += f"""
📊 거래 빈도 비교
- 룰 기반 거래 수: {rule_trades}회
- 모델 기반 거래 수: {model_trades}회
- 차이: {model_trades-rule_trades:+d}회
"""
    
    # 모델 특화 지표
    if 'model_accuracy' in model_results['metrics']:
        model_accuracy = model_results['metrics']['model_accuracy']
        report += f"""
🤖 모델 성능
- 모델 정확도: {model_accuracy:.3f}
- 매수 신호 비율: {model_results['model_info']['signal_counts'].get('BUY', 0) / len(model_results['model_info']['predictions']):.1%}
"""
    
    report += f"\n{'='*60}\n"
    
    return report

def main():
    """테스트용 메인 함수"""
    print("🚀 SectorFlow Lite - Backtest Module 테스트")
    print("=" * 50)
    
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # 가격 데이터 생성
    price = 100
    prices = [price]
    for _ in range(99):
        price *= (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    # 샘플 데이터프레임 생성
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # High, Low 조정
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    
    # 피처 계산 (간단한 버전)
    df['trading_value'] = df['close'] * df['volume']
    df['trading_value_mean'] = df['trading_value'].rolling(window=20).mean()
    df['trading_value_std'] = df['trading_value'].rolling(window=20).std()
    df['z20'] = (df['trading_value'] - df['trading_value_mean']) / df['trading_value_std']
    df['z20'] = df['z20'].replace([np.inf, -np.inf], np.nan)
    
    # RS 계산
    df['returns'] = df['close'].pct_change()
    positive_returns = df['returns'].where(df['returns'] > 0, 0)
    negative_returns = df['returns'].where(df['returns'] < 0, 0).abs()
    df['avg_gain'] = positive_returns.rolling(window=20).mean()
    df['avg_loss'] = negative_returns.rolling(window=20).mean()
    df['rs_4w'] = df['avg_gain'] / df['avg_loss']
    df['rs_4w'] = df['rs_4w'].replace([np.inf, -np.inf], np.nan)
    
    # 1. 룰 기반 신호 생성
    df['trading_signal'] = 'HOLD'
    buy_condition = (df['z20'] >= 1.0) & (df['rs_4w'] > 1.0)
    df.loc[buy_condition, 'trading_signal'] = 'BUY'
    
    print("📊 샘플 데이터 생성 완료")
    print(f"   - 기간: {df['date'].min()} ~ {df['date'].max()}")
    print(f"   - 데이터 수: {len(df)}개")
    print(f"   - BUY 신호: {df['trading_signal'].value_counts().get('BUY', 0)}개")
    
    # 2. 룰 기반 백테스트
    print("\n🔧 룰 기반 백테스트 실행...")
    rule_results = run_backtest(df)
    
    # 3. 모델 기반 백테스트 (가상의 모델 예측)
    print("\n🤖 모델 기반 백테스트 실행...")
    # 가상의 모델 예측 생성 (실제로는 훈련된 모델에서 가져옴)
    np.random.seed(42)
    model_predictions = np.random.beta(2, 5, len(df))  # 0-1 범위의 베타 분포
    model_predictions = np.clip(model_predictions, 0, 1)
    
    model_results = run_model_backtest(df, model_predictions, threshold=0.5)
    
    # 4. 전략 비교
    print("\n📊 전략 비교 분석...")
    comparison_report = generate_comparison_report(rule_results, model_results)
    print(comparison_report)
    
    # 5. 거래 분석
    print("📊 거래 분석:")
    for strategy_name, results in [("룰 기반", rule_results), ("모델 기반", model_results)]:
        trade_analysis = analyze_trades(results['trades'])
        if 'error' not in trade_analysis:
            print(f"\n{strategy_name}:")
            print(f"   - 거래 쌍 수: {trade_analysis['total_pairs']}개")
            print(f"   - 수익 거래: {trade_analysis['profitable_pairs']}개")
            print(f"   - 손실 거래: {trade_analysis['losing_pairs']}개")
            print(f"   - 평균 보유일: {trade_analysis['avg_holding_days']:.1f}일")
            print(f"   - 평균 수익률: {trade_analysis['avg_return_pct']:.2f}%")
    
    print("\n✅ Backtest Module 테스트 완료!")
    return {
        'rule_results': rule_results,
        'model_results': model_results,
        'comparison_report': comparison_report
    }

if __name__ == "__main__":
    results = main()

