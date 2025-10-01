#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SectorFlow Lite - GRU/LSTM Models Module
딥러닝 모델 (GRU, LSTM) 구현

Functions:
- create_gru_model: GRU 모델 생성
- create_lstm_model: LSTM 모델 생성
- create_attention_model: Attention 메커니즘 포함 모델
- compile_model: 모델 컴파일
- get_model_summary: 모델 요약 정보
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras가 설치되어 있지 않을 수 있으므로 try-except로 처리
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        GRU, LSTM, Dense, Dropout, BatchNormalization, 
        Input, Attention, MultiHeadAttention, LayerNormalization,
        GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.utils import plot_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow가 설치되지 않았습니다. pip install tensorflow로 설치하세요.")

def create_gru_model(input_shape: Tuple[int, int],
                     hidden_sizes: list = [64, 32],
                     dropout_rate: float = 0.2,
                     recurrent_dropout: float = 0.2,
                     use_batch_norm: bool = True,
                     use_attention: bool = False) -> Model:
    """
    GRU 모델 생성
    
    Args:
        input_shape: 입력 형태 (timesteps, features)
        hidden_sizes: 은닉층 크기 리스트
        dropout_rate: 드롭아웃 비율
        recurrent_dropout: 순환 드롭아웃 비율
        use_batch_norm: 배치 정규화 사용 여부
        use_attention: 어텐션 메커니즘 사용 여부
        
    Returns:
        컴파일된 GRU 모델
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    print("🔧 GRU 모델 생성 중...")
    
    model = Sequential()
    
    # 입력층
    model.add(Input(shape=input_shape))
    
    # GRU 레이어들
    for i, hidden_size in enumerate(hidden_sizes):
        return_sequences = i < len(hidden_sizes) - 1 or use_attention
        
        model.add(GRU(
            hidden_size,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            name=f'gru_{i+1}'
        ))
        
        if use_batch_norm:
            model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
    
    # 어텐션 메커니즘 (선택사항)
    if use_attention:
        model.add(Attention(name='attention'))
    
    # 드롭아웃
    model.add(Dropout(dropout_rate, name='dropout_final'))
    
    # 출력층
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    print("✅ GRU 모델 생성 완료!")
    return model

def create_lstm_model(input_shape: Tuple[int, int],
                      hidden_sizes: list = [64, 32],
                      dropout_rate: float = 0.2,
                      recurrent_dropout: float = 0.2,
                      use_batch_norm: bool = True,
                      use_attention: bool = False) -> Model:
    """
    LSTM 모델 생성
    
    Args:
        input_shape: 입력 형태 (timesteps, features)
        hidden_sizes: 은닉층 크기 리스트
        dropout_rate: 드롭아웃 비율
        recurrent_dropout: 순환 드롭아웃 비율
        use_batch_norm: 배치 정규화 사용 여부
        use_attention: 어텐션 메커니즘 사용 여부
        
    Returns:
        컴파일된 LSTM 모델
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    print("🔧 LSTM 모델 생성 중...")
    
    model = Sequential()
    
    # 입력층
    model.add(Input(shape=input_shape))
    
    # LSTM 레이어들
    for i, hidden_size in enumerate(hidden_sizes):
        return_sequences = i < len(hidden_sizes) - 1 or use_attention
        
        model.add(LSTM(
            hidden_size,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            name=f'lstm_{i+1}'
        ))
        
        if use_batch_norm:
            model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
    
    # 어텐션 메커니즘 (선택사항)
    if use_attention:
        model.add(Attention(name='attention'))
    
    # 드롭아웃
    model.add(Dropout(dropout_rate, name='dropout_final'))
    
    # 출력층
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    print("✅ LSTM 모델 생성 완료!")
    return model

def create_attention_model(input_shape: Tuple[int, int],
                          hidden_sizes: list = [64, 32],
                          dropout_rate: float = 0.2,
                          num_heads: int = 4) -> Model:
    """
    Multi-Head Attention 모델 생성
    
    Args:
        input_shape: 입력 형태 (timesteps, features)
        hidden_sizes: 은닉층 크기 리스트
        dropout_rate: 드롭아웃 비율
        num_heads: 어텐션 헤드 수
        
    Returns:
        컴파일된 Attention 모델
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    print("🔧 Multi-Head Attention 모델 생성 중...")
    
    # 입력
    inputs = Input(shape=input_shape, name='input')
    
    # 임베딩 레이어 (선택사항)
    x = Dense(hidden_sizes[0], activation='relu', name='embedding')(inputs)
    x = LayerNormalization(name='layer_norm_1')(x)
    
    # Multi-Head Attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_sizes[0] // num_heads,
        name='multi_head_attention'
    )(x, x)
    
    # 잔차 연결 및 정규화
    x = Concatenate(name='concat')([x, attention_output])
    x = LayerNormalization(name='layer_norm_2')(x)
    
    # 추가 Dense 레이어들
    for i, hidden_size in enumerate(hidden_sizes[1:], 1):
        x = Dense(hidden_size, activation='relu', name=f'dense_{i}')(x)
        x = Dropout(dropout_rate, name=f'dropout_{i}')(x)
        x = BatchNormalization(name=f'batch_norm_{i}')(x)
    
    # Global pooling
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # 출력층
    outputs = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='attention_model')
    
    print("✅ Multi-Head Attention 모델 생성 완료!")
    return model

def create_hybrid_model(input_shape: Tuple[int, int],
                        gru_hidden_sizes: list = [64, 32],
                        lstm_hidden_sizes: list = [64, 32],
                        dropout_rate: float = 0.2) -> Model:
    """
    GRU + LSTM 하이브리드 모델 생성
    
    Args:
        input_shape: 입력 형태 (timesteps, features)
        gru_hidden_sizes: GRU 은닉층 크기 리스트
        lstm_hidden_sizes: LSTM 은닉층 크기 리스트
        dropout_rate: 드롭아웃 비율
        
    Returns:
        컴파일된 하이브리드 모델
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    print("🔧 GRU + LSTM 하이브리드 모델 생성 중...")
    
    # 입력
    inputs = Input(shape=input_shape, name='input')
    
    # GRU 브랜치
    gru_branch = inputs
    for i, hidden_size in enumerate(gru_hidden_sizes):
        gru_branch = GRU(
            hidden_size,
            return_sequences=True,
            dropout=dropout_rate,
            name=f'gru_{i+1}'
        )(gru_branch)
        gru_branch = BatchNormalization(name=f'gru_batch_norm_{i+1}')(gru_branch)
    
    # LSTM 브랜치
    lstm_branch = inputs
    for i, hidden_size in enumerate(lstm_hidden_sizes):
        lstm_branch = LSTM(
            hidden_size,
            return_sequences=True,
            dropout=dropout_rate,
            name=f'lstm_{i+1}'
        )(lstm_branch)
        lstm_branch = BatchNormalization(name=f'lstm_batch_norm_{i+1}')(lstm_branch)
    
    # 브랜치 결합
    combined = Concatenate(name='concat_branches')([gru_branch, lstm_branch])
    
    # 추가 처리
    combined = Dense(64, activation='relu', name='dense_combined')(combined)
    combined = Dropout(dropout_rate, name='dropout_combined')(combined)
    combined = BatchNormalization(name='batch_norm_combined')(combined)
    
    # Global pooling
    combined = GlobalAveragePooling1D(name='global_avg_pool')(combined)
    
    # 출력층
    outputs = Dense(1, activation='sigmoid', name='output')(combined)
    
    model = Model(inputs=inputs, outputs=outputs, name='hybrid_model')
    
    print("✅ 하이브리드 모델 생성 완료!")
    return model

def compile_model(model: Model,
                  learning_rate: float = 0.001,
                  optimizer: str = 'adam',
                  loss: str = 'binary_crossentropy',
                  metrics: list = None) -> Model:
    """
    모델 컴파일
    
    Args:
        model: 컴파일할 모델
        learning_rate: 학습률
        optimizer: 옵티마이저
        loss: 손실 함수
        metrics: 평가 지표
        
    Returns:
        컴파일된 모델
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall']
    
    # 옵티마이저 설정
    if optimizer.lower() == 'adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = optimizer
    
    # 모델 컴파일
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )
    
    print(f"✅ 모델 컴파일 완료 (학습률: {learning_rate})")
    return model

def get_model_summary(model: Model) -> str:
    """
    모델 요약 정보 생성
    
    Args:
        model: 모델
        
    Returns:
        모델 요약 문자열
    """
    if not TENSORFLOW_AVAILABLE:
        return "TensorFlow가 설치되지 않았습니다."
    
    # 모델 구조 요약
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    
    # 파라미터 수 계산
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    summary = "\n".join(summary_lines)
    summary += f"\n\n총 파라미터 수: {total_params:,}"
    summary += f"\n훈련 가능한 파라미터: {trainable_params:,}"
    summary += f"\n훈련 불가능한 파라미터: {non_trainable_params:,}"
    
    return summary

def create_callbacks(patience: int = 10,
                    min_delta: float = 0.001,
                    factor: float = 0.5,
                    min_lr: float = 1e-7,
                    save_path: str = None) -> list:
    """
    훈련 콜백 생성
    
    Args:
        patience: 조기 종료 patience
        min_delta: 최소 개선량
        factor: 학습률 감소 비율
        min_lr: 최소 학습률
        save_path: 모델 저장 경로
        
    Returns:
        콜백 리스트
    """
    if not TENSORFLOW_AVAILABLE:
        return []
    
    callbacks = []
    
    # 조기 종료
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # 학습률 감소
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=factor,
        patience=patience//2,
        min_lr=min_lr,
        verbose=1
    )
    callbacks.append(lr_scheduler)
    
    # 모델 저장 (경로가 제공된 경우)
    if save_path:
        model_checkpoint = ModelCheckpoint(
            filepath=save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
    
    return callbacks

def create_model_factory(model_type: str = 'gru',
                        input_shape: Tuple[int, int] = (30, 7),
                        **kwargs) -> Model:
    """
    모델 팩토리 함수
    
    Args:
        model_type: 모델 타입 ('gru', 'lstm', 'attention', 'hybrid')
        input_shape: 입력 형태
        **kwargs: 추가 파라미터
        
    Returns:
        생성된 모델
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow가 설치되지 않았습니다.")
    
    model_type = model_type.lower()
    
    if model_type == 'gru':
        model = create_gru_model(input_shape, **kwargs)
    elif model_type == 'lstm':
        model = create_lstm_model(input_shape, **kwargs)
    elif model_type == 'attention':
        model = create_attention_model(input_shape, **kwargs)
    elif model_type == 'hybrid':
        model = create_hybrid_model(input_shape, **kwargs)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    # 모델 컴파일
    model = compile_model(model, **kwargs)
    
    return model

def main():
    """테스트용 메인 함수"""
    print("🚀 SectorFlow Lite - GRU/LSTM Models Module 테스트")
    print("=" * 60)
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow가 설치되지 않았습니다.")
        return None
    
    # 테스트용 입력 형태
    input_shape = (30, 7)  # 30일, 7개 피처
    
    # 다양한 모델 생성 및 테스트
    models = {}
    
    # 1. GRU 모델
    try:
        gru_model = create_model_factory('gru', input_shape, hidden_sizes=[64, 32])
        models['GRU'] = gru_model
        print("✅ GRU 모델 생성 완료")
    except Exception as e:
        print(f"❌ GRU 모델 생성 실패: {e}")
    
    # 2. LSTM 모델
    try:
        lstm_model = create_model_factory('lstm', input_shape, hidden_sizes=[64, 32])
        models['LSTM'] = lstm_model
        print("✅ LSTM 모델 생성 완료")
    except Exception as e:
        print(f"❌ LSTM 모델 생성 실패: {e}")
    
    # 3. Attention 모델
    try:
        attention_model = create_model_factory('attention', input_shape, hidden_sizes=[64, 32])
        models['Attention'] = attention_model
        print("✅ Attention 모델 생성 완료")
    except Exception as e:
        print(f"❌ Attention 모델 생성 실패: {e}")
    
    # 4. 하이브리드 모델
    try:
        hybrid_model = create_model_factory('hybrid', input_shape)
        models['Hybrid'] = hybrid_model
        print("✅ 하이브리드 모델 생성 완료")
    except Exception as e:
        print(f"❌ 하이브리드 모델 생성 실패: {e}")
    
    # 모델 요약 출력
    for name, model in models.items():
        print(f"\n📊 {name} 모델 요약:")
        print(get_model_summary(model))
    
    print(f"\n✅ 총 {len(models)}개 모델 생성 완료!")
    return models

if __name__ == "__main__":
    models = main()
