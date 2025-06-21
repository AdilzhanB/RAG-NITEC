"""
Продвинутая система обратной связи с машинным обучением и аналитикой
"""

import json
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import hashlib
import sqlite3
import aiosqlite

from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
from transformers import pipeline

from config import *

class FeedbackType(Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    RATING = "rating"
    CORRECTION = "correction"
    SUGGESTION = "suggestion"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"

class ValidationStatus(Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    SUSPICIOUS = "suspicious"
    PROCESSED = "processed"

class FeedbackQuality(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPAM = "spam"

@dataclass
class EnhancedFeedback:
    feedback_id: str
    interaction_id: str
    user_id: str
    session_id: str
    feedback_type: FeedbackType
    rating: Optional[int] = None
    correction: Optional[str] = None
    suggestion: Optional[str] = None
    explanation: Optional[str] = None
    timestamp: datetime = None
    validation_status: ValidationStatus = ValidationStatus.PENDING
    quality_score: float = 0.0
    confidence_score: float = 0.0
    anomaly_score: float = 0.0
    semantic_features: Optional[List[float]] = None
    metadata: Dict[str, Any] = None

@dataclass
class UserBehaviorProfile:
    user_id: str
    total_feedback: int = 0
    feedback_distribution: Dict[str, int] = None
    average_rating: float = 3.0
    rating_variance: float = 1.0
    feedback_frequency: float = 0.0  # feedbacks per day
    avg_response_time: float = 0.0   # seconds
    consistency_score: float = 1.0
    trustworthiness: float = 1.0
    expertise_domains: Dict[str, float] = None
    first_feedback: Optional[datetime] = None
    last_feedback: Optional[datetime] = None

class AnomalyDetector:
    """Детектор аномалий в обратной связи"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # 10% аномалий
            random_state=42
        )
        self.is_fitted = False
        self.feature_history = deque(maxlen=1000)
        
    def extract_features(self, feedback: EnhancedFeedback, 
                        user_profile: UserBehaviorProfile) -> np.ndarray:
        """Извлечение признаков для детекции аномалий"""
        features = []
        
        # Временные признаки
        if feedback.timestamp:
            hour = feedback.timestamp.hour
            day_of_week = feedback.timestamp.weekday()
            features.extend([hour / 24.0, day_of_week / 7.0])
        else:
            features.extend([0.5, 0.5])
        
        # Признаки пользователя
        features.extend([
            min(1.0, user_profile.total_feedback / 100.0),  # Нормализованное количество
            user_profile.average_rating / 5.0,
            min(1.0, user_profile.rating_variance / 4.0),
            min(1.0, user_profile.feedback_frequency / 10.0),
            user_profile.consistency_score,
            user_profile.trustworthiness
        ])
        
        # Признаки обратной связи
        feedback_type_encoding = {
            FeedbackType.LIKE: 0.2,
            FeedbackType.DISLIKE: 0.4,
            FeedbackType.RATING: 0.6,
            FeedbackType.CORRECTION: 0.8,
            FeedbackType.SUGGESTION: 1.0,
            FeedbackType.BUG_REPORT: 0.3,
            FeedbackType.FEATURE_REQUEST: 0.7
        }
        
        features.append(feedback_type_encoding.get(feedback.feedback_type, 0.5))
        
        # Рейтинг (если есть)
        if feedback.rating:
            features.append(feedback.rating / 5.0)
        else:
            features.append(0.5)  # Нейтральное значение
        
        # Длина текстовых полей
        correction_length = len(feedback.correction or '') / 1000.0
        suggestion_length = len(feedback.suggestion or '') / 1000.0
        explanation_length = len(feedback.explanation or '') / 1000.0
        
        features.extend([
            min(1.0, correction_length),
            min(1.0, suggestion_length),
            min(1.0, explanation_length)
        ])
        
        return np.array(features)
    def fit_or_update(self, features_batch: List[np.ndarray]):
        """Обучение или обновление детектора"""
        if not features_batch:
            return
        
        # Добавление в историю
        self.feature_history.extend(features_batch)
        
        if len(self.feature_history) >= 50:  # Минимум для обучения
            features_array = np.array(list(self.feature_history))
            self.isolation_forest.fit(features_array)
            self.is_fitted = True
    
    def detect_anomaly(self, features: np.ndarray) -> Tuple[bool, float]:
        """Детекция аномалии"""
        if not self.is_fitted:
            return False, 0.5  # Нейтральная оценка
        
        features_reshaped = features.reshape(1, -1)
        
        # Предсказание аномалии (-1 = аномалия, 1 = норма)
        prediction = self.isolation_forest.predict(features_reshaped)[0]
        
        # Оценка аномальности
        anomaly_score = self.isolation_forest.decision_function(features_reshaped)[0]
        
        # Нормализация оценки в диапазон [0, 1]
        normalized_score = (anomaly_score + 0.5) / 1.0
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        is_anomaly = prediction == -1
        
        return is_anomaly, normalized_score

class SemanticAnalyzer:
    """Семантический анализатор обратной связи"""
    
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.is_fitted = False
        
    def analyze_feedback_text(self, text: str) -> Dict[str, Any]:
        """Анализ текста обратной связи"""
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'sentiment_score': 0.5,
                'toxicity': 0.0,
                'coherence': 0.5,
                'information_density': 0.0
            }
        
        # Анализ тональности
        sentiment_results = self.sentiment_analyzer(text)
        sentiment_data = max(sentiment_results[0], key=lambda x: x['score'])
        
        # Анализ токсичности (простая эвристика)
        toxicity_score = self._calculate_toxicity(text)
        
        # Анализ связности
        coherence_score = self._calculate_coherence(text)
        
        # Плотность информации
        info_density = self._calculate_information_density(text)
        
        return {
            'sentiment': sentiment_data['label'].lower(),
            'sentiment_score': sentiment_data['score'],
            'toxicity': toxicity_score,
            'coherence': coherence_score,
            'information_density': info_density
        }
    
    def _calculate_toxicity(self, text: str) -> float:
        """Простая оценка токсичности"""
        toxic_words = [
            'дурак', 'идиот', 'тупой', 'плохо', 'ужасно', 'отвратительно',
            'stupid', 'bad', 'terrible', 'awful', 'hate'
        ]
        
        text_lower = text.lower()
        toxic_count = sum(1 for word in toxic_words if word in text_lower)
        
        # Нормализация
        words_count = len(text.split())
        toxicity = toxic_count / max(1, words_count)
        
        return min(1.0, toxicity * 5)  # Усиление для обнаружения
    
    def _calculate_coherence(self, text: str) -> float:
        """Оценка связности текста"""
        sentences = text.split('.')
        
        if len(sentences) <= 1:
            return 0.5
        
        # Простая метрика: соотношение длинных предложений
        long_sentences = [s for s in sentences if len(s.strip()) > 20]
        coherence = len(long_sentences) / len(sentences)
        
        return min(1.0, coherence * 1.5)
    
    def _calculate_information_density(self, text: str) -> float:
        """Плотность информации в тексте"""
        words = text.split()
        
        if not words:
            return 0.0
        
        # Уникальные слова
        unique_words = set(word.lower() for word in words)
        uniqueness = len(unique_words) / len(words)
        
        # Средняя длина слов
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_score = min(1.0, avg_word_length / 8)
        
        # Комбинированная оценка
        density = (uniqueness * 0.7 + length_score * 0.3)
        
        return density

class FeedbackValidator:
    """Продвинутый валидатор обратной связи"""
    
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.semantic_analyzer = SemanticAnalyzer()
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        
    async def validate_feedback(self, feedback: EnhancedFeedback) -> Tuple[ValidationStatus, float, Dict[str, Any]]:
        """Комплексная валидация обратной связи"""
        
        # Получение или создание профиля пользователя
        user_profile = await self._get_or_create_user_profile(feedback.user_id)
        
        # Извлечение признаков
        features = self.anomaly_detector.extract_features(feedback, user_profile)
        
        # Детекция аномалий
        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(features)
        feedback.anomaly_score = anomaly_score
        
        # Семантический анализ
        semantic_analysis = {}
        if feedback.correction:
            semantic_analysis['correction'] = self.semantic_analyzer.analyze_feedback_text(feedback.correction)
        if feedback.suggestion:
            semantic_analysis['suggestion'] = self.semantic_analyzer.analyze_feedback_text(feedback.suggestion)
        if feedback.explanation:
            semantic_analysis['explanation'] = self.semantic_analyzer.analyze_feedback_text(feedback.explanation)
        
        # Валидационные проверки
        validation_results = await self._run_validation_checks(feedback, user_profile, semantic_analysis)
        
        # Определение статуса валидации
        validation_status = self._determine_validation_status(
            validation_results, is_anomaly, semantic_analysis
        )
        
        # Вычисление общей оценки качества
        quality_score = self._calculate_quality_score(validation_results, semantic_analysis)
        
        # Обновление профиля пользователя
        await self._update_user_profile(user_profile, feedback, quality_score)
        
        return validation_status, quality_score, {
            'validation_checks': validation_results,
            'anomaly_detected': is_anomaly,
            'anomaly_score': anomaly_score,
            'semantic_analysis': semantic_analysis,
            'user_profile_updated': True
        }
    
    async def _get_or_create_user_profile(self, user_id: str) -> UserBehaviorProfile:
        """Получение или создание профиля пользователя"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserBehaviorProfile(
                user_id=user_id,
                feedback_distribution={},
                expertise_domains={}
            )
        
        return self.user_profiles[user_id]
    
    async def _run_validation_checks(self, feedback: EnhancedFeedback, 
                                   user_profile: UserBehaviorProfile,
                                   semantic_analysis: Dict[str, Any]) -> Dict[str, bool]:
        """Выполнение валидационных проверок"""
        checks = {}
        
        # 1. Проверка типа обратной связи
        checks['valid_feedback_type'] = self._check_feedback_type(feedback)
        
        # 2. Проверка рейтинга
        checks['valid_rating'] = self._check_rating(feedback)
        
        # 3. Проверка на спам
        checks['not_spam'] = await self._check_spam(feedback, user_profile)
        
        # 4. Проверка консистентности пользователя
        checks['user_consistent'] = self._check_user_consistency(feedback, user_profile)
        
        # 5. Проверка качества текста
        checks['quality_text'] = self._check_text_quality(feedback, semantic_analysis)
        
        # 6. Проверка временных паттернов
        checks['temporal_valid'] = self._check_temporal_patterns(feedback, user_profile)
        
        # 7. Проверка на дубликаты
        checks['not_duplicate'] = await self._check_duplicates(feedback)
        
        return checks
    
    def _check_feedback_type(self, feedback: EnhancedFeedback) -> bool:
        """Проверка валидности типа обратной связи"""
        required_fields = {
            FeedbackType.RATING: ['rating'],
            FeedbackType.CORRECTION: ['correction'],
            FeedbackType.SUGGESTION: ['suggestion'],
            FeedbackType.BUG_REPORT: ['explanation'],
            FeedbackType.FEATURE_REQUEST: ['suggestion']
        }
        
        if feedback.feedback_type in required_fields:
            for field in required_fields[feedback.feedback_type]:
                if not getattr(feedback, field):
                    return False
        
        return True
    
    def _check_rating(self, feedback: EnhancedFeedback) -> bool:
        """Проверка валидности рейтинга"""
        if feedback.feedback_type == FeedbackType.RATING:
            return feedback.rating is not None and 1 <= feedback.rating <= 5
        return True
    
    async def _check_spam(self, feedback: EnhancedFeedback, 
                         user_profile: UserBehaviorProfile) -> bool:
        """Проверка на спам"""
        # Проверка частоты отправки
        if user_profile.feedback_frequency > 50:  # Более 50 отзывов в день
            return False
        
        # Проверка повторяющегося контента
        if feedback.correction and len(feedback.correction) < 10:
            return False
        
        # Проверка на случайные клики (слишком быстрый ответ)
        if hasattr(feedback, 'response_time') and feedback.response_time < 2:  # Менее 2 секунд
            return False
        
        return True
    
    def _check_user_consistency(self, feedback: EnhancedFeedback,
                              user_profile: UserBehaviorProfile) -> bool:
        """Проверка консистентности пользователя"""
        if user_profile.total_feedback < 5:
            return True  # Новые пользователи получают преимущество
        
        # Проверка согласованности рейтингов
        if feedback.rating and user_profile.rating_variance > 3.0:
            rating_diff = abs(feedback.rating - user_profile.average_rating)
            if rating_diff > 2 * user_profile.rating_variance:
                return False
        
        return user_profile.consistency_score > 0.3
    
    def _check_text_quality(self, feedback: EnhancedFeedback,
                           semantic_analysis: Dict[str, Any]) -> bool:
        """Проверка качества текста"""
        for field_analysis in semantic_analysis.values():
            # Проверка на токсичность
            if field_analysis.get('toxicity', 0) > 0.5:
                return False
            
            # Проверка связности
            if field_analysis.get('coherence', 0) < 0.2:
                return False
            
            # Проверка на негативность (для конструктивной критики)
            if (field_analysis.get('sentiment') == 'negative' and 
                field_analysis.get('information_density', 0) < 0.3):
                return False
        
        return True
    
    def _check_temporal_patterns(self, feedback: EnhancedFeedback,
                               user_profile: UserBehaviorProfile) -> bool:
        """Проверка временных паттернов"""
        if not feedback.timestamp:
            return True
        
        # Проверка на подозрительное время (например, 3-5 утра)
        hour = feedback.timestamp.hour
        if 3 <= hour <= 5:
            return user_profile.trustworthiness > 0.7
        
        return True
    
    async def _check_duplicates(self, feedback: EnhancedFeedback) -> bool:
        """Проверка на дубликаты"""
        # Простая проверка - в реальности нужна более сложная логика
        # с использованием семантического сравнения
        return True
    
    def _determine_validation_status(self, validation_results: Dict[str, bool],
                                   is_anomaly: bool, 
                                   semantic_analysis: Dict[str, Any]) -> ValidationStatus:
        """Определение статуса валидации"""
        passed_checks = sum(validation_results.values())
        total_checks = len(validation_results)
        
        pass_rate = passed_checks / total_checks
        
        # Критерии для статусов
        if is_anomaly or pass_rate < 0.5:
            return ValidationStatus.SUSPICIOUS
        elif pass_rate < 0.7:
            return ValidationStatus.REJECTED
        elif pass_rate >= 0.9:
            return ValidationStatus.VALIDATED
        else:
            return ValidationStatus.PENDING
    
    def _calculate_quality_score(self, validation_results: Dict[str, bool],
                               semantic_analysis: Dict[str, Any]) -> float:
        """Вычисление оценки качества"""
        # Базовая оценка на основе валидационных проверок
        base_score = sum(validation_results.values()) / len(validation_results)
        
        # Бонусы за качество семантического анализа
        semantic_bonus = 0.0
        analysis_count = 0
        
        for field_analysis in semantic_analysis.values():
            analysis_count += 1
            
            # Бонус за высокую плотность информации
            info_density = field_analysis.get('information_density', 0)
            semantic_bonus += info_density * 0.1
            
            # Бонус за связность
            coherence = field_analysis.get('coherence', 0)
            semantic_bonus += coherence * 0.05
            
            # Штраф за токсичность
            toxicity = field_analysis.get('toxicity', 0)
            semantic_bonus -= toxicity * 0.2
        
        if analysis_count > 0:
            semantic_bonus /= analysis_count
        
        # Финальная оценка
        quality_score = base_score + semantic_bonus
        
        return max(0.0, min(1.0, quality_score))
    
    async def _update_user_profile(self, user_profile: UserBehaviorProfile,
                                 feedback: EnhancedFeedback, quality_score: float):
        """Обновление профиля пользователя"""
        current_time = feedback.timestamp or datetime.now()
        
        # Обновление базовых метрик
        user_profile.total_feedback += 1
        user_profile.last_feedback = current_time
        
        if user_profile.first_feedback is None:
            user_profile.first_feedback = current_time
        
        # Обновление распределения типов обратной связи
        if user_profile.feedback_distribution is None:
            user_profile.feedback_distribution = {}
        
        feedback_type = feedback.feedback_type.value
        user_profile.feedback_distribution[feedback_type] = \
            user_profile.feedback_distribution.get(feedback_type, 0) + 1
        
        # Обновление рейтингов
        if feedback.rating:
            old_avg = user_profile.average_rating
            n = user_profile.total_feedback
            
            # Обновление среднего
            user_profile.average_rating = (old_avg * (n - 1) + feedback.rating) / n
            
            # Обновление дисперсии
            if n > 1:
                user_profile.rating_variance = (
                    user_profile.rating_variance * (n - 2) + 
                    (feedback.rating - user_profile.average_rating) ** 2
                ) / (n - 1)
        
        # Обновление частоты
        if user_profile.first_feedback and user_profile.last_feedback:
            days_active = (user_profile.last_feedback - user_profile.first_feedback).days + 1
            user_profile.feedback_frequency = user_profile.total_feedback / days_active
        
        # Обновление оценки доверия
        trust_adjustment = (quality_score - 0.5) * 0.1  # Малые изменения
        user_profile.trustworthiness = max(0.1, min(1.0, 
            user_profile.trustworthiness + trust_adjustment))
        
        # Обновление консистентности
        if user_profile.total_feedback > 5:
            consistency_factor = min(1.0, quality_score / 0.7)
            user_profile.consistency_score = (
                user_profile.consistency_score * 0.9 + consistency_factor * 0.1
            )

class AdvancedFeedbackSystem:
    """Продвинутая система обратной связи"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.validator = FeedbackValidator()
        self.feedback_storage: List[EnhancedFeedback] = []
        self.training_queue: deque = deque(maxlen=1000)
        
        # База данных SQLite для персистентности
        self.db_path = FEEDBACK_DIR / "feedback.db"
        asyncio.create_task(self._initialize_database())
        
        self.logger.info("Advanced Feedback System инициализирована")
    
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('AdvancedFeedback')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def _initialize_database(self):
        """Инициализация базы данных"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    interaction_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    feedback_type TEXT,
                    rating INTEGER,
                    correction TEXT,
                    suggestion TEXT,
                    explanation TEXT,
                    timestamp TEXT,
                    validation_status TEXT,
                    quality_score REAL,
                    confidence_score REAL,
                    anomaly_score REAL,
                    metadata TEXT
                )
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    total_feedback INTEGER,
                    average_rating REAL,
                    trustworthiness REAL,
                    consistency_score REAL,
                    feedback_frequency REAL,
                    last_updated TEXT,
                    profile_data TEXT
                )
            ''')
            
            await db.commit()
    
    async def process_advanced_feedback(self, interaction: Any, 
                                      feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка продвинутой обратной связи"""
        try:
            # Создание объекта обратной связи
            feedback = EnhancedFeedback(
                feedback_id=f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(feedback_data)) % 10000:04d}",
                interaction_id=interaction.interaction_id,
                user_id=interaction.user_id,
                session_id=interaction.session_id,
                feedback_type=FeedbackType(feedback_data['feedback_type']),
                rating=feedback_data.get('rating'),
                correction=feedback_data.get('correction'),
                suggestion=feedback_data.get('suggestion'),
                explanation=feedback_data.get('explanation'),
                timestamp=datetime.now(),
                metadata={
                    'original_query': interaction.query,
                    'response_quality': interaction.response_quality.value,
                    'user_agent': feedback_data.get('user_agent', 'unknown'),
                    'response_time': feedback_data.get('response_time', 0)
                }
            )
            
            # Валидация обратной связи
            validation_status, quality_score, validation_details = await self.validator.validate_feedback(feedback)
            
            feedback.validation_status = validation_status
            feedback.quality_score = quality_score
            feedback.confidence_score = self._calculate_confidence(validation_details)
            
            # Сохранение обратной связи
            await self._save_feedback(feedback)
            
            # Добавление в очередь обучения (если валидна)
            if validation_status in [ValidationStatus.VALIDATED, ValidationStatus.PENDING]:
                await self._add_to_training_queue(feedback, interaction)
            
            # Обновление детектора аномалий
            await self._update_anomaly_detector(feedback)
            
            result = {
                'success': True,
                'feedback_id': feedback.feedback_id,
                'validation_status': validation_status.value,
                'quality_score': quality_score,
                'confidence_score': feedback.confidence_score,
                'validation_details': validation_details
            }
            
            self.logger.info(f"Обработана обратная связь {feedback.feedback_id} со статусом {validation_status.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки обратной связи: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'validation_status': ValidationStatus.REJECTED.value
            }
    
    def _calculate_confidence(self, validation_details: Dict[str, Any]) -> float:
        """Вычисление уверенности в обратной связи"""
        # Базовая уверенность на основе валидационных проверок
        validation_checks = validation_details.get('validation_checks', {})
        passed_ratio = sum(validation_checks.values()) / len(validation_checks) if validation_checks else 0.5
        
        # Модификация на основе аномальности
        anomaly_score = validation_details.get('anomaly_score', 0.5)
        anomaly_penalty = (1 - anomaly_score) * 0.3
        
        # Модификация на основе семантического анализа
        semantic_bonus = 0.0
        semantic_analysis = validation_details.get('semantic_analysis', {})
        
        for analysis in semantic_analysis.values():
            info_density = analysis.get('information_density', 0)
            coherence = analysis.get('coherence', 0)
            semantic_bonus += (info_density + coherence) * 0.1
        
        # Финальная уверенность
        confidence = passed_ratio - anomaly_penalty + semantic_bonus
        
        return max(0.0, min(1.0, confidence))
    
    async def _save_feedback(self, feedback: EnhancedFeedback):
        """Сохранение обратной связи в базу данных"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.feedback_id,
                feedback.interaction_id,
                feedback.user_id,
                feedback.session_id,
                feedback.feedback_type.value,
                feedback.rating,
                feedback.correction,
                feedback.suggestion,
                feedback.explanation,
                feedback.timestamp.isoformat() if feedback.timestamp else None,
                feedback.validation_status.value,
                feedback.quality_score,
                feedback.confidence_score,
                feedback.anomaly_score,
                json.dumps(feedback.metadata) if feedback.metadata else None
            ))
            await db.commit()
        
        # Также сохранение в памяти
        self.feedback_storage.append(feedback)
    
    async def _add_to_training_queue(self, feedback: EnhancedFeedback, interaction: Any):
        """Добавление в очередь обучения"""
        training_sample = {
            'feedback_id': feedback.feedback_id,
            'query': interaction.query,
            'response': interaction.generated_response,
            'intent_type': interaction.intent.intent_type.value,
            'domain': interaction.intent.domain,
            'feedback_type': feedback.feedback_type.value,
            'rating': feedback.rating,
            'quality_score': feedback.quality_score,
            'feedback_positive': self._is_positive_feedback(feedback),
            'timestamp': feedback.timestamp.isoformat()
        }
        
        self.training_queue.append(training_sample)
    
    def _is_positive_feedback(self, feedback: EnhancedFeedback) -> bool:
        """Определение позитивности обратной связи"""
        if feedback.feedback_type == FeedbackType.LIKE:
            return True
        elif feedback.feedback_type == FeedbackType.DISLIKE:
            return False
        elif feedback.feedback_type == FeedbackType.RATING:
            return feedback.rating >= 4
        elif feedback.feedback_type in [FeedbackType.CORRECTION, FeedbackType.SUGGESTION]:
            return feedback.quality_score > 0.6  # Конструктивная критика
        else:
            return feedback.quality_score > 0.5
    
    async def _update_anomaly_detector(self, feedback: EnhancedFeedback):
        """Обновление детектора аномалий"""
        user_profile = await self.validator._get_or_create_user_profile(feedback.user_id)
        features = self.validator.anomaly_detector.extract_features(feedback, user_profile)
        
        # Обновление детектора каждые 10 новых примеров
        if len(self.feedback_storage) % 10 == 0:
            recent_features = []
            for recent_feedback in self.feedback_storage[-50:]:  # Последние 50
                if recent_feedback.validation_status == ValidationStatus.VALIDATED:
                    user_prof = await self.validator._get_or_create_user_profile(recent_feedback.user_id)
                    feat = self.validator.anomaly_detector.extract_features(recent_feedback, user_prof)
                    recent_features.append(feat)
            
            if recent_features:
                self.validator.anomaly_detector.fit_or_update(recent_features)
    
    async def get_training_data(self, min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """Получение данных для обучения"""
        training_data = []
        
        for sample in self.training_queue:
            # Получение соответствующей обратной связи
            feedback_id = sample['feedback_id']
            feedback = next((f for f in self.feedback_storage if f.feedback_id == feedback_id), None)
            
            if (feedback and 
                feedback.validation_status == ValidationStatus.VALIDATED and
                feedback.confidence_score >= min_confidence):
                training_data.append(sample)
        
        return training_data
    
    async def mark_feedback_used(self, training_data: List[Dict[str, Any]]):
        """Отметка обратной связи как использованной"""
        used_ids = {sample['feedback_id'] for sample in training_data}
        
        async with aiosqlite.connect(self.db_path) as db:
            for feedback_id in used_ids:
                await db.execute(
                    'UPDATE feedback SET validation_status = ? WHERE feedback_id = ?',
                    (ValidationStatus.PROCESSED.value, feedback_id)
                )
            await db.commit()
        
        # Удаление из очереди обучения
        self.training_queue = deque([
            sample for sample in self.training_queue 
            if sample['feedback_id'] not in used_ids
        ], maxlen=1000)
    
    async def get_advanced_stats(self) -> Dict[str, Any]:
        """Получение расширенной статистики"""
        # Статистика из базы данных
        async with aiosqlite.connect(self.db_path) as db:
            # Общая статистика
            cursor = await db.execute('SELECT COUNT(*) FROM feedback')
            total_feedback = (await cursor.fetchone())[0]
            
            # Статистика по статусам валидации
            cursor = await db.execute('''
                SELECT validation_status, COUNT(*) 
                FROM feedback 
                GROUP BY validation_status
            ''')
            status_distribution = dict(await cursor.fetchall())
            
            # Статистика по типам обратной связи
            cursor = await db.execute('''
                SELECT feedback_type, COUNT(*) 
                FROM feedback 
                GROUP BY feedback_type
            ''')
            type_distribution = dict(await cursor.fetchall())
            
            # Средние метрики
            cursor = await db.execute('''
                SELECT AVG(quality_score), AVG(confidence_score), AVG(anomaly_score)
                FROM feedback 
                WHERE validation_status = ?
            ''', (ValidationStatus.VALIDATED.value,))
            avg_metrics = await cursor.fetchone()
            
            # Статистика пользователей
            cursor = await db.execute('SELECT COUNT(DISTINCT user_id) FROM feedback')
            unique_users = (await cursor.fetchone())[0]
        
        # Статистика обучения
        training_queue_size = len(self.training_queue)
        
        # Статистика детектора аномалий
        anomaly_detector_trained = self.validator.anomaly_detector.is_fitted
        
        # Профили пользователей
        user_profiles_count = len(self.validator.user_profiles)
        avg_user_trustworthiness = np.mean([
            profile.trustworthiness 
            for profile in self.validator.user_profiles.values()
        ]) if self.validator.user_profiles else 0
        
        return {
            'total_feedback': total_feedback,
            'unique_users': unique_users,
            'status_distribution': status_distribution,
            'type_distribution': type_distribution,
            'average_metrics': {
                'quality_score': avg_metrics[0] if avg_metrics[0] else 0,
                'confidence_score': avg_metrics[1] if avg_metrics[1] else 0,
                'anomaly_score': avg_metrics[2] if avg_metrics[2] else 0
            },
            'training_queue_size': training_queue_size,
            'anomaly_detector_trained': anomaly_detector_trained,
            'user_profiles': {
                'total_profiles': user_profiles_count,
                'average_trustworthiness': avg_user_trustworthiness
            },
            'validation_rate': (
                status_distribution.get(ValidationStatus.VALIDATED.value, 0) / 
                max(1, total_feedback)
            )
        }
    
    async def get_user_feedback_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Получение истории обратной связи пользователя"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                SELECT * FROM feedback 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            history = []
            for row in rows:
                feedback_dict = dict(zip(columns, row))
                if feedback_dict['metadata']:
                    feedback_dict['metadata'] = json.loads(feedback_dict['metadata'])
                history.append(feedback_dict)
            
            return history
    
    async def analyze_feedback_trends(self, days: int = 30) -> Dict[str, Any]:
        """Анализ трендов обратной связи"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Тренд по дням
            cursor = await db.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM feedback 
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (cutoff_date,))
            
            daily_trend = await cursor.fetchall()
            
            # Тренд качества
            cursor = await db.execute('''
                SELECT DATE(timestamp) as date, AVG(quality_score) as avg_quality
                FROM feedback 
                WHERE timestamp >= ? AND validation_status = ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (cutoff_date, ValidationStatus.VALIDATED.value))
            
            quality_trend = await cursor.fetchall()
            
            # Распределение по часам
            cursor = await db.execute('''
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                FROM feedback 
                WHERE timestamp >= ?
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''', (cutoff_date,))
            
            hourly_distribution = await cursor.fetchall()
        
        return {
            'daily_feedback_count': [{'date': row[0], 'count': row[1]} for row in daily_trend],
            'daily_quality_trend': [{'date': row[0], 'quality': row[1]} for row in quality_trend],
            'hourly_distribution': [{'hour': int(row[0]), 'count': row[1]} for row in hourly_distribution],
            'analysis_period_days': days
        }

# Глобальный экземпляр
_feedback_system_instance = None

def get_feedback_system() -> AdvancedFeedbackSystem:
    """Получение singleton экземпляра системы обратной связи"""
    global _feedback_system_instance
    if _feedback_system_instance is None:
        _feedback_system_instance = AdvancedFeedbackSystem()
    return _feedback_system_instance