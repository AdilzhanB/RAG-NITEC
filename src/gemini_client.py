import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import time

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config import GEMINI_API_KEY

class GenerationStrategy(Enum):
    FACTUAL = "factual"
    CREATIVE = "creative" 
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"

@dataclass
class GenerationContext:
    strategy: GenerationStrategy
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    safety_settings: Dict[str, Any]

@dataclass
class ResponseMetrics:
    generation_time: float
    token_count: int
    safety_scores: Dict[str, float]
    quality_indicators: Dict[str, float]
    confidence_level: float

class PromptOptimizer:
    """Оптимизатор промптов на основе обратной связи"""
    
    def __init__(self):
        self.prompt_templates = {
            GenerationStrategy.FACTUAL: self._load_factual_templates(),
            GenerationStrategy.ANALYTICAL: self._load_analytical_templates(),
            GenerationStrategy.CONVERSATIONAL: self._load_conversational_templates(),
            GenerationStrategy.TECHNICAL: self._load_technical_templates()
        }
        self.performance_history = {}
        
    def _load_factual_templates(self) -> List[str]:
        """Шаблоны для фактических запросов"""
        return [
            """Ты - эксперт-консультант по государственным услугам Казахстана.

КОНТЕКСТ:
{context}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {query}

ИНСТРУКЦИИ:
1. Предоставь точную и актуальную информацию
2. Используй только данные из контекста
3. Структурируй ответ четко и логично
4. Указывай источники информации
5. Если информации недостаточно, честно об этом скажи

ОТВЕТ:""",

            """Как AI-ассистент государственных услуг, отвечаю на основе официальных данных.

ДОСТУПНАЯ ИНФОРМАЦИЯ:
{context}

ВОПРОС: {query}

Мой ответ основан исключительно на предоставленных данных:""",

            """СИСТЕМА: Государственный AI-консультант Казахстана
ЗАДАЧА: Предоставить точную информацию по запросу

ДАННЫЕ:
{context}

ЗАПРОС: {query}

ТРЕБОВАНИЯ:
- Только проверенная информация
- Ссылки на источники
- Структурированный ответ

РЕЗУЛЬТАТ:"""
        ]
    
    def _load_analytical_templates(self) -> List[str]:
        """Шаблоны для аналитических запросов"""
        return [
            """Ты - аналитик государственных процессов с глубоким пониманием казахстанского контекста.

ИСХОДНЫЕ ДАННЫЕ:
{context}

АНАЛИТИЧЕСКИЙ ЗАПРОС: {query}

МЕТОДОЛОГИЯ АНАЛИЗА:
1. Изучи предоставленные данные
2. Выяви ключевые факторы и взаимосвязи
3. Проанализируй причины и следствия
4. Предоставь обоснованные выводы
5. Укажи ограничения анализа

АНАЛИТИЧЕСКИЙ ОТЧЕТ:""",

            """АНАЛИТИЧЕСКАЯ СИСТЕМА: Анализ государственных процессов РК

ВХОДНЫЕ ДАННЫЕ:
{context}

ЗАПРОС НА АНАЛИЗ: {query}

Провожу комплексный анализ с учетом:
- Нормативно-правовой базы
- Практической реализации
- Потенциальных рисков
- Рекомендаций по улучшению

РЕЗУЛЬТАТЫ АНАЛИЗА:"""
        ]
    
    def _load_conversational_templates(self) -> List[str]:
        """Шаблоны для разговорных запросов"""
        return [
            """Привет! Я - ваш AI-помощник по государственным услугам Казахстана. 
Готов помочь разобраться в любых вопросах.

ЧТО Я ЗНАЮ:
{context}

ВАШ ВОПРОС: {query}

Постараюсь объяснить простыми словами:""",

            """Здравствуйте! Рад помочь с вашим вопросом о государственных услугах.

ДОСТУПНАЯ ИНФОРМАЦИЯ:
{context}

ВЫ СПРАШИВАЕТЕ: {query}

Отвечу максимально понятно и полезно:"""
        ]
    
    def _load_technical_templates(self) -> List[str]:
        """Шаблоны для технических запросов"""
        return [
            """ТЕХНИЧЕСКАЯ ДОКУМЕНТАЦИЯ: Государственные цифровые системы РК

СПРАВОЧНЫЕ МАТЕРИАЛЫ:
{context}

ТЕХНИЧЕСКИЙ ЗАПРОС: {query}

ТЕХНИЧЕСКИЕ СПЕЦИФИКАЦИИ:
- Архитектура системы
- Протоколы взаимодействия
- Требования безопасности
- Процедуры интеграции

ТЕХНИЧЕСКОЕ РЕШЕНИЕ:""",

            """СИСТЕМА: Техническая поддержка цифровых госуслуг

БАЗА ЗНАНИЙ:
{context}

ПРОБЛЕМА/ВОПРОС: {query}

ДИАГНОСТИКА И РЕШЕНИЕ:"""
        ]
    
    def select_optimal_template(self, strategy: GenerationStrategy, 
                              intent_data: Any, feedback_history: List[Dict] = None) -> str:
        """Выбор оптимального шаблона на основе истории производительности"""
        
        templates = self.prompt_templates.get(strategy, self.prompt_templates[GenerationStrategy.FACTUAL])
        
        if not feedback_history:
            return templates[0]  # Базовый шаблон
        
        # Анализ производительности шаблонов
        template_scores = {}
        
        for i, template in enumerate(templates):
            template_hash = hashlib.md5(template.encode()).hexdigest()[:8]
            
            # Поиск в истории обратной связи
            relevant_feedback = [
                fb for fb in feedback_history 
                if fb.get('template_id') == template_hash
            ]
            
            if relevant_feedback:
                avg_rating = sum(fb.get('rating', 3) for fb in relevant_feedback) / len(relevant_feedback)
                template_scores[i] = avg_rating
            else:
                template_scores[i] = 3.0  # Нейтральная оценка
        
        # Выбор лучшего шаблона
        best_template_idx = max(template_scores, key=template_scores.get)
        return templates[best_template_idx]
    
    def optimize_prompt(self, template: str, query: str, context: str, 
                       intent_data: Any, session_history: List[Dict] = None) -> str:
        """Оптимизация промпта на основе контекста и истории"""
        
        # Базовая подстановка
        optimized_prompt = template.format(query=query, context=context)
        
        # Добавление контекста сессии
        if session_history:
            session_context = self._build_session_context(session_history)
            if session_context:
                optimized_prompt = f"{session_context}\n\n{optimized_prompt}"
        
        # Добавление специфичных для интента инструкций
        intent_instructions = self._get_intent_specific_instructions(intent_data)
        if intent_instructions:
            optimized_prompt = f"{optimized_prompt}\n\nДОПОЛНИТЕЛЬНО: {intent_instructions}"
        
        return optimized_prompt
    
    def _build_session_context(self, session_history: List[Dict]) -> str:
        """Построение контекста сессии"""
        if not session_history:
            return ""
        
        context_parts = ["КОНТЕКСТ БЕСЕДЫ:"]
        
        for entry in session_history[-3:]:  # Последние 3 взаимодействия
            query = entry.get('query', '')
            topic = entry.get('topic', '')
            
            if query and topic:
                context_parts.append(f"- Ранее обсуждали: {topic} (запрос: {query[:50]}...)")
        
        return "\n".join(context_parts) if len(context_parts) > 1 else ""
    
    def _get_intent_specific_instructions(self, intent_data: Any) -> str:
        """Получение инструкций, специфичных для интента"""
        if not intent_data:
            return ""
        
        intent_type = getattr(intent_data, 'intent_type', None)
        if not intent_type:
            return ""
        
        instructions = {
            'procedural': "Предоставь пошаговую инструкцию с указанием сроков и необходимых документов.",
            'comparative': "Сравни различные варианты, укажи преимущества и недостатки каждого.",
            'analytical': "Проведи глубокий анализ с выявлением причинно-следственных связей.",
            'troubleshooting': "Предложи конкретные решения проблемы с альтернативными вариантами.",
            'factual': "Предоставь проверенные факты с указанием источников."
        }
        
        return instructions.get(intent_type.value, "")

class ResponseAnalyzer:
    """Анализатор качества сгенерированных ответов"""
    
    def __init__(self):
        self.quality_thresholds = {
            'min_length': 50,
            'max_length': 3000,
            'min_sentences': 2,
            'max_repetition_ratio': 0.3
        }
    
    def analyze_response(self, response: str, query: str, context: str) -> ResponseMetrics:
        """Анализ качества ответа"""
        
        start_time = time.time()
        
        # Базовые метрики
        word_count = len(response.split())
        char_count = len(response)
        sentence_count = len([s for s in response.split('.') if s.strip()])
        
        # Качественные индикаторы
        quality_indicators = {
            'length_appropriate': self._check_length_appropriateness(response),
            'structure_quality': self._assess_structure(response),
            'content_relevance': self._assess_relevance(response, query),
            'information_density': self._calculate_information_density(response),
            'readability': self._assess_readability(response),
            'factual_consistency': self._check_factual_consistency(response, context),
            'completeness': self._assess_completeness(response, query)
        }
        
        # Общий уровень уверенности
        confidence_level = sum(quality_indicators.values()) / len(quality_indicators)
        
        return ResponseMetrics(
            generation_time=time.time() - start_time,
            token_count=word_count,
            safety_scores={},  # Будет заполнено из Gemini API
            quality_indicators=quality_indicators,
            confidence_level=confidence_level
        )
    
    def _check_length_appropriateness(self, response: str) -> float:
        """Проверка соответствия длины ответа"""
        length = len(response)
        
        if self.quality_thresholds['min_length'] <= length <= self.quality_thresholds['max_length']:
            return 1.0
        elif length < self.quality_thresholds['min_length']:
            return length / self.quality_thresholds['min_length']
        else:
            excess_ratio = (length - self.quality_thresholds['max_length']) / self.quality_thresholds['max_length']
            return max(0.0, 1.0 - excess_ratio)
    
    def _assess_structure(self, response: str) -> float:
        """Оценка структуры ответа"""
        # Наличие заголовков, списков, абзацев
        structure_indicators = [
            bool(response.count('\n\n')),  # Абзацы
            bool(response.count('.')),     # Предложения
            bool(response.count(':')),     # Структурированность
            len(response.split('.')) >= 2  # Минимум предложений
        ]
        
        return sum(structure_indicators) / len(structure_indicators)
    
    def _assess_relevance(self, response: str, query: str) -> float:
        """Оценка релевантности ответа запросу"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Пересечение ключевых слов
        common_words = query_words & response_words
        
        if not query_words:
            return 0.5
        
        relevance_ratio = len(common_words) / len(query_words)
        return min(1.0, relevance_ratio * 2)  # Усиление до 100%
    
    def _calculate_information_density(self, response: str) -> float:
        """Вычисление плотности информации"""
        words = response.split()
        
        if not words:
            return 0.0
        
        # Уникальные слова
        unique_words = set(word.lower() for word in words)
        uniqueness_ratio = len(unique_words) / len(words)
        
        # Средняя длина слов
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_score = min(1.0, avg_word_length / 6)  # Нормализация к 6 символам
        
        return (uniqueness_ratio * 0.7 + length_score * 0.3)
    
    def _assess_readability(self, response: str) -> float:
        """Оценка читаемости"""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Средняя длина предложений
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Оптимальная длина предложения 10-20 слов
        if 10 <= avg_sentence_length <= 20:
            readability = 1.0
        elif avg_sentence_length < 10:
            readability = avg_sentence_length / 10
        else:
            readability = max(0.0, 1.0 - (avg_sentence_length - 20) / 20)
        
        return readability
    
    def _check_factual_consistency(self, response: str, context: str) -> float:
        """Проверка фактической согласованности с контекстом"""
        if not context:
            return 0.5
        
        # Простая проверка на наличие общих терминов
        context_terms = set(context.lower().split())
        response_terms = set(response.lower().split())
        
        common_terms = context_terms & response_terms
        
        if not context_terms:
            return 0.5
        
        consistency = len(common_terms) / len(context_terms)
        return min(1.0, consistency * 3)  # Усиление
    
    def _assess_completeness(self, response: str, query: str) -> float:
        """Оценка полноты ответа"""
        # Ключевые слова вопроса
        question_words = ['что', 'как', 'где', 'когда', 'почему', 'кто', 'какой']
        query_lower = query.lower()
        
        question_type = None
        for word in question_words:
            if word in query_lower:
                question_type = word
                break
        
        if not question_type:
            return 0.7  # Нейтральная оценка
        
        # Проверка соответствия типу вопроса
        response_lower = response.lower()
        
        completeness_checks = {
            'что': any(word in response_lower for word in ['это', 'является', 'представляет']),
            'как': any(word in response_lower for word in ['процедура', 'способ', 'метод', 'шаги']),
            'где': any(word in response_lower for word in ['адрес', 'место', 'офис', 'сайт']),
            'когда': any(word in response_lower for word in ['срок', 'время', 'дата', 'период']),
            'почему': any(word in response_lower for word in ['причина', 'потому', 'поскольку']),
            'кто': any(word in response_lower for word in ['орган', 'ведомство', 'должностное']),
            'какой': any(word in response_lower for word in ['характеристика', 'особенность', 'тип'])
        }
        
        return 1.0 if completeness_checks.get(question_type, False) else 0.4

class AdvancedGeminiClient:
    """Продвинутый клиент для Gemini API"""
    
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY не установлен")
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Инициализация моделей
        self.models = {
            'pro': genai.GenerativeModel('gemini-1.5-pro'),
            'flash': genai.GenerativeModel('gemini-1.5-flash')
        }
        
        # Компоненты
        self.prompt_optimizer = PromptOptimizer()
        self.response_analyzer = ResponseAnalyzer()
        
        # История и кэширование
        self.generation_history = []
        self.response_cache = {}
        self.feedback_history = []
        
        # Настройки безопасности
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        self.logger = self._setup_logging()
        self.logger.info("Advanced Gemini Client инициализирован")
    
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('AdvancedGemini')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _determine_generation_strategy(self, intent_data: Any) -> GenerationStrategy:
        """Определение стратегии генерации на основе интента"""
        if not intent_data:
            return GenerationStrategy.CONVERSATIONAL
        
        intent_type = getattr(intent_data, 'intent_type', None)
        if not intent_type:
            return GenerationStrategy.CONVERSATIONAL
        
        strategy_mapping = {
            'factual': GenerationStrategy.FACTUAL,
            'procedural': GenerationStrategy.TECHNICAL,
            'comparative': GenerationStrategy.ANALYTICAL,
            'analytical': GenerationStrategy.ANALYTICAL,
            'creative': GenerationStrategy.CREATIVE,
            'troubleshooting': GenerationStrategy.TECHNICAL
        }
        
        return strategy_mapping.get(intent_type.value, GenerationStrategy.CONVERSATIONAL)
    
    def _get_generation_context(self, strategy: GenerationStrategy, 
                               complexity: float = 0.5) -> GenerationContext:
        """Получение контекста генерации"""
        
        base_configs = {
            GenerationStrategy.FACTUAL: {
                'temperature': 0.1,
                'top_p': 0.8,
                'top_k': 20,
                'max_tokens': 1500
            },
            GenerationStrategy.ANALYTICAL: {
                'temperature': 0.3,
                'top_p': 0.9,
                'top_k': 40,
                'max_tokens': 2000
            },
            GenerationStrategy.CONVERSATIONAL: {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'max_tokens': 1200
            },
            GenerationStrategy.TECHNICAL: {
                'temperature': 0.2,
                'top_p': 0.8,
                'top_k': 30,
                'max_tokens': 1800
            },
            GenerationStrategy.CREATIVE: {
                'temperature': 0.8,
                'top_p': 0.95,
                'top_k': 50,
                'max_tokens': 1600
            }
        }
        
        config = base_configs.get(strategy, base_configs[GenerationStrategy.CONVERSATIONAL])
        
        # Адаптация к сложности
        if complexity > 0.7:  # Высокая сложность
            config['max_tokens'] = int(config['max_tokens'] * 1.3)
            config['temperature'] = max(0.1, config['temperature'] - 0.1)
        elif complexity < 0.3:  # Низкая сложность
            config['max_tokens'] = int(config['max_tokens'] * 0.8)
        
        return GenerationContext(
            strategy=strategy,
            temperature=config['temperature'],
            top_p=config['top_p'],
            top_k=config['top_k'],
            max_tokens=config['max_tokens'],
            safety_settings=self.safety_settings
        )
    
    async def generate_advanced_response(self, query: str, processed_query: str, 
                                       context: str, intent: Any, 
                                       session_history: List[Dict] = None) -> str:
        """Продвинутая генерация ответа с адаптивными настройками"""
        
        try:
            # Определение стратегии
            strategy = self._determine_generation_strategy(intent)
            
            # Получение контекста генерации
            complexity = getattr(intent, 'complexity_score', 0.5)
            gen_context = self._get_generation_context(strategy, complexity)
            
            # Проверка кэша
            cache_key = self._generate_cache_key(query, context, strategy)
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                if self._is_cache_valid(cached_response):
                    self.logger.info(f"Возвращен кэшированный ответ для ключа {cache_key[:8]}")
                    return cached_response['response']
            
            # Оптимизация промпта
            template = self.prompt_optimizer.select_optimal_template(
                strategy, intent, self.feedback_history
            )
            
            optimized_prompt = self.prompt_optimizer.optimize_prompt(
                template, query, context, intent, session_history
            )
            
            # Выбор модели на основе сложности
            model_key = 'pro' if complexity > 0.6 else 'flash'
            model = self.models[model_key]
            
            """
Продвинутый клиент для Google Gemini API (продолжение)
"""

            # Настройка генерации
            generation_config = genai.types.GenerationConfig(
                temperature=gen_context.temperature,
                top_p=gen_context.top_p,
                top_k=gen_context.top_k,
                max_output_tokens=gen_context.max_tokens,
            )
            
            # Генерация ответа
            start_time = time.time()
            
            response = await self._generate_with_retry(
                model, optimized_prompt, generation_config, gen_context.safety_settings
            )
            
            generation_time = time.time() - start_time
            
            # Анализ качества ответа
            response_metrics = self.response_analyzer.analyze_response(
                response, query, context
            )
            response_metrics.generation_time = generation_time
            
            # Постобработка ответа
            processed_response = await self._postprocess_response(
                response, query, context, intent, response_metrics
            )
            
            # Сохранение в историю и кэш
            await self._save_generation_record(
                query, processed_query, context, intent, processed_response, 
                response_metrics, strategy, model_key
            )
            
            # Кэширование
            self.response_cache[cache_key] = {
                'response': processed_response,
                'timestamp': datetime.now(),
                'metrics': response_metrics,
                'strategy': strategy
            }
            
            self.logger.info(
                f"Сгенерирован ответ ({model_key}) за {generation_time:.2f}с, "
                f"качество: {response_metrics.confidence_level:.2f}"
            )
            
            return processed_response
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации ответа: {str(e)}")
            return await self._generate_fallback_response(query, context)
    
    async def _generate_with_retry(self, model, prompt: str, config, safety_settings, 
                                 max_retries: int = 3) -> str:
        """Генерация с повторными попытками"""
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=config,
                    safety_settings=safety_settings
                )
                
                if response.text:
                    return response.text
                else:
                    raise ValueError("Пустой ответ от модели")
                    
            except Exception as e:
                self.logger.warning(f"Попытка {attempt + 1} неудачна: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Уменьшение температуры для следующей попытки
                    config.temperature = max(0.1, config.temperature * 0.8)
                    await asyncio.sleep(1)  # Небольшая задержка
                else:
                    raise e
        
        raise Exception("Превышено максимальное количество попыток генерации")
    
    async def _postprocess_response(self, response: str, query: str, context: str,
                                  intent: Any, metrics: ResponseMetrics) -> str:
        """Постобработка сгенерированного ответа"""
        
        # Базовая очистка
        processed = response.strip()
        
        # Удаление артефактов генерации
        processed = self._remove_generation_artifacts(processed)
        
        # Проверка и коррекция фактов
        processed = await self._verify_and_correct_facts(processed, context)
        
        # Улучшение структуры
        processed = self._improve_structure(processed, intent)
        
        # Добавление источников и заключения
        processed = self._add_sources_and_conclusion(processed, context)
        
        # Проверка качества после обработки
        if metrics.confidence_level < 0.5:
            processed = await self._enhance_low_quality_response(processed, query, context)
        
        return processed
    
    def _remove_generation_artifacts(self, response: str) -> str:
        """Удаление артефактов генерации"""
        # Удаление повторяющихся фраз
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line_clean = line.strip().lower()
            if line_clean and line_clean not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line_clean)
            elif not line_clean:  # Пустые строки оставляем
                cleaned_lines.append(line)
        
        # Удаление лишних пробелов и переносов
        cleaned = '\n'.join(cleaned_lines)
        cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))
        
        # Удаление множественных переносов
        while '\n\n\n' in cleaned:
            cleaned = cleaned.replace('\n\n\n', '\n\n')
        
        return cleaned
    
    async def _verify_and_correct_facts(self, response: str, context: str) -> str:
        """Верификация и коррекция фактов"""
        if not context:
            return response
        
        # Простая проверка на наличие фактов, не упомянутых в контексте
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Список потенциально проблематичных утверждений
        warning_phrases = [
            'всегда', 'никогда', 'абсолютно', 'точно известно',
            'гарантированно', 'обязательно', 'исключительно'
        ]
        
        # Поиск и смягчение категоричных утверждений
        for phrase in warning_phrases:
            if phrase in response_lower:
                # Добавление предостережения в конец
                if "Рекомендуется уточнить" not in response:
                    response += "\n\n*Рекомендуется уточнить актуальную информацию в соответствующих государственных органах."
                break
        
        return response
    
    def _improve_structure(self, response: str, intent: Any) -> str:
        """Улучшение структуры ответа"""
        if not intent:
            return response
        
        intent_type = getattr(intent, 'intent_type', None)
        if not intent_type:
            return response
        
        # Структурирование в зависимости от типа интента
        if intent_type.value == 'procedural':
            return self._structure_procedural_response(response)
        elif intent_type.value == 'comparative':
            return self._structure_comparative_response(response)
        elif intent_type.value == 'analytical':
            return self._structure_analytical_response(response)
        else:
            return self._structure_general_response(response)
    
    def _structure_procedural_response(self, response: str) -> str:
        """Структурирование процедурного ответа"""
        # Поиск пошаговых инструкций
        if 'шаг' not in response.lower() and ('.' in response or ':' in response):
            # Попытка автоматического структурирования
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            
            if len(sentences) > 2:
                structured = "**Пошаговая инструкция:**\n\n"
                for i, sentence in enumerate(sentences[:8], 1):  # Максимум 8 шагов
                    if len(sentence) > 20:  # Только содержательные предложения
                        structured += f"{i}. {sentence}.\n"
                
                # Добавление оставшейся части ответа
                remaining = '. '.join(sentences[8:])
                if remaining:
                    structured += f"\n**Дополнительная информация:**\n{remaining}."
                
                return structured
        
        return response
    
    def _structure_comparative_response(self, response: str) -> str:
        """Структурирование сравнительного ответа"""
        # Добавление заголовков для сравнения
        if 'сравнение' in response.lower() or 'различие' in response.lower():
            if '**' not in response:  # Если нет структуры
                # Простое добавление заголовка
                response = f"**Сравнительный анализ:**\n\n{response}"
        
        return response
    
    def _structure_analytical_response(self, response: str) -> str:
        """Структурирование аналитического ответа"""
        # Добавление аналитической структуры
        if 'анализ' in response.lower() and '**' not in response:
            response = f"**Аналитический обзор:**\n\n{response}"
        
        return response
    
    def _structure_general_response(self, response: str) -> str:
        """Общее структурирование ответа"""
        # Базовое улучшение читаемости
        paragraphs = response.split('\n\n')
        
        if len(paragraphs) > 3:
            # Добавление подзаголовков для длинных ответов
            structured_paragraphs = []
            
            for i, paragraph in enumerate(paragraphs):
                if i == 0:
                    structured_paragraphs.append(paragraph)
                elif len(paragraph) > 100:
                    # Попытка создать подзаголовок
                    first_sentence = paragraph.split('.')[0]
                    if len(first_sentence) < 80:
                        structured_paragraphs.append(f"**{first_sentence.strip()}**\n\n{paragraph}")
                    else:
                        structured_paragraphs.append(paragraph)
                else:
                    structured_paragraphs.append(paragraph)
            
            return '\n\n'.join(structured_paragraphs)
        
        return response
    
    def _add_sources_and_conclusion(self, response: str, context: str) -> str:
        """Добавление источников и заключения"""
        if not context:
            return response
        
        # Извлечение источников из контекста
        sources = []
        for line in context.split('\n'):
            if line.startswith('[Источник'):
                source_info = line.split(':', 1)
                if len(source_info) > 1:
                    sources.append(source_info[1].split('|')[0].strip())
        
        # Добавление источников в конец ответа
        if sources and "Источники:" not in response:
            sources_text = "\n\n**Источники информации:**\n"
            for i, source in enumerate(set(sources[:3]), 1):  # Максимум 3 уникальных источника
                sources_text += f"{i}. {source}\n"
            
            response += sources_text
        
        return response
    
    async def _enhance_low_quality_response(self, response: str, query: str, 
                                          context: str) -> str:
        """Улучшение ответа низкого качества"""
        
        # Добавление предупреждения о неполноте информации
        if len(response) < 100:
            response += "\n\n*Примечание: Информация может быть неполной. Рекомендуется обратиться к официальным источникам для получения более детальной информации."
        
        # Добавление контактной информации для дополнительной помощи
        if "контакт" not in response.lower() and "обращ" not in response.lower():
            response += "\n\n*Для получения дополнительной помощи вы можете обратиться в ЦОН или воспользоваться порталом egov.kz."
        
        return response
    
    def _generate_cache_key(self, query: str, context: str, strategy: GenerationStrategy) -> str:
        """Генерация ключа для кэширования"""
        combined = f"{query}|{context[:200]}|{strategy.value}"  # Первые 200 символов контекста
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_response: Dict[str, Any], 
                       max_age_hours: int = 24) -> bool:
        """Проверка актуальности кэшированного ответа"""
        timestamp = cached_response.get('timestamp')
        if not timestamp:
            return False
        
        age = datetime.now() - timestamp
        return age.total_seconds() / 3600 < max_age_hours
    
    async def _save_generation_record(self, query: str, processed_query: str, 
                                    context: str, intent: Any, response: str,
                                    metrics: ResponseMetrics, strategy: GenerationStrategy,
                                    model_key: str):
        """Сохранение записи о генерации"""
        
        record = {
            'timestamp': datetime.now(),
            'query': query,
            'processed_query': processed_query,
            'context_length': len(context),
            'intent_type': intent.intent_type.value if intent else 'unknown',
            'intent_confidence': intent.confidence if intent else 0.0,
            'strategy': strategy.value,
            'model_used': model_key,
            'response_length': len(response),
            'generation_time': metrics.generation_time,
            'quality_score': metrics.confidence_level,
            'token_count': metrics.token_count
        }
        
        self.generation_history.append(record)
        
        # Ограничение размера истории
        if len(self.generation_history) > 1000:
            self.generation_history = self.generation_history[-800:]  # Оставляем последние 800
    
    async def _generate_fallback_response(self, query: str, context: str) -> str:
        """Генерация резервного ответа при ошибках"""
        
        fallback_responses = [
            f"Извините, у меня возникли технические сложности с обработкой вашего запроса '{query}'. "
            f"Попробуйте переформулировать вопрос или обратитесь позже.",
            
            f"К сожалению, не удалось полностью обработать ваш запрос '{query}'. "
            f"Рекомендую обратиться в ЦОН по телефону 1414 или на портал egov.kz.",
            
            f"Произошла ошибка при генерации ответа на ваш вопрос '{query}'. "
            f"Пожалуйста, попробуйте задать вопрос по-другому или воспользуйтесь официальными источниками."
        ]
        
        # Выбор случайного резервного ответа
        import random
        return random.choice(fallback_responses)
    
    async def learn_from_feedback(self, interaction_id: str, feedback_data: Dict[str, Any]):
        """Обучение на основе обратной связи"""
        
        # Поиск соответствующей записи генерации
        matching_record = None
        for record in self.generation_history:
            if record.get('interaction_id') == interaction_id:
                matching_record = record
                break
        
        if not matching_record:
            self.logger.warning(f"Не найдена запись генерации для взаимодействия {interaction_id}")
            return
        
        # Создание записи обратной связи
        feedback_record = {
            'interaction_id': interaction_id,
            'timestamp': datetime.now(),
            'feedback_type': feedback_data.get('feedback_type'),
            'rating': feedback_data.get('rating'),
            'positive': self._is_positive_feedback(feedback_data),
            'strategy_used': matching_record.get('strategy'),
            'model_used': matching_record.get('model_used'),
            'generation_time': matching_record.get('generation_time'),
            'quality_score': matching_record.get('quality_score'),
            'template_id': feedback_data.get('template_id')  # Если есть
        }
        
        self.feedback_history.append(feedback_record)
        
        # Ограничение размера истории обратной связи
        if len(self.feedback_history) > 500:
            self.feedback_history = self.feedback_history[-400:]
        
        # Адаптивное обучение
        await self._adaptive_learning_update(feedback_record)
        
        self.logger.info(f"Получена обратная связь для взаимодействия {interaction_id}")
    
    def _is_positive_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Определение позитивности обратной связи"""
        feedback_type = feedback_data.get('feedback_type')
        rating = feedback_data.get('rating')
        
        if feedback_type == 'like':
            return True
        elif feedback_type == 'dislike':
            return False
        elif feedback_type == 'rating' and rating:
            return rating >= 4
        elif feedback_type in ['correction', 'suggestion']:
            # Конструктивная критика считается нейтральной/слабо позитивной
            return True
        
        return False
    
    async def _adaptive_learning_update(self, feedback_record: Dict[str, Any]):
        """Адаптивное обновление на основе обратной связи"""
        
        strategy = feedback_record.get('strategy_used')
        model_used = feedback_record.get('model_used')
        is_positive = feedback_record.get('positive', False)
        
        # Простое обновление предпочтений стратегий
        if not hasattr(self, 'strategy_performance'):
            self.strategy_performance = {s.value: {'positive': 0, 'total': 0} for s in GenerationStrategy}
        
        if strategy in self.strategy_performance:
            self.strategy_performance[strategy]['total'] += 1
            if is_positive:
                self.strategy_performance[strategy]['positive'] += 1
        
        # Обновление предпочтений моделей
        if not hasattr(self, 'model_performance'):
            self.model_performance = {'pro': {'positive': 0, 'total': 0}, 'flash': {'positive': 0, 'total': 0}}
        
        if model_used in self.model_performance:
            self.model_performance[model_used]['total'] += 1
            if is_positive:
                self.model_performance[model_used]['positive'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности"""
        
        # Общая статистика генерации
        total_generations = len(self.generation_history)
        
        if total_generations == 0:
            return {'error': 'Нет данных о генерации'}
        
        # Средние метрики
        avg_generation_time = sum(r.get('generation_time', 0) for r in self.generation_history) / total_generations
        avg_quality_score = sum(r.get('quality_score', 0) for r in self.generation_history) / total_generations
        avg_response_length = sum(r.get('response_length', 0) for r in self.generation_history) / total_generations
        
        # Статистика по стратегиям
        strategy_stats = {}
        for record in self.generation_history:
            strategy = record.get('strategy', 'unknown')
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'count': 0, 'avg_quality': 0, 'avg_time': 0}
            
            strategy_stats[strategy]['count'] += 1
            strategy_stats[strategy]['avg_quality'] += record.get('quality_score', 0)
            strategy_stats[strategy]['avg_time'] += record.get('generation_time', 0)
        
        # Нормализация средних значений
        for strategy in strategy_stats:
            count = strategy_stats[strategy]['count']
            strategy_stats[strategy]['avg_quality'] /= count
            strategy_stats[strategy]['avg_time'] /= count
        
        # Статистика обратной связи
        feedback_stats = {
            'total_feedback': len(self.feedback_history),
            'positive_feedback': sum(1 for f in self.feedback_history if f.get('positive', False)),
            'feedback_rate': len(self.feedback_history) / total_generations if total_generations > 0 else 0
        }
        
        # Производительность стратегий (если есть обратная связь)
        strategy_performance = getattr(self, 'strategy_performance', {})
        model_performance = getattr(self, 'model_performance', {})
        
        return {
            'generation_stats': {
                'total_generations': total_generations,
                'avg_generation_time': round(avg_generation_time, 3),
                'avg_quality_score': round(avg_quality_score, 3),
                'avg_response_length': round(avg_response_length, 1)
            },
            'strategy_stats': strategy_stats,
            'feedback_stats': feedback_stats,
            'strategy_performance': strategy_performance,
            'model_performance': model_performance,
            'cache_stats': {
                'cache_size': len(self.response_cache),
                'cache_hit_potential': sum(1 for entry in self.response_cache.values() 
                                         if self._is_cache_valid(entry))
            }
        }
    
    async def optimize_generation_settings(self) -> Dict[str, Any]:
        """Оптимизация настроек генерации на основе обратной связи"""
        
        if len(self.feedback_history) < 10:
            return {'message': 'Недостаточно данных для оптимизации'}
        
        # Анализ лучших настроек
        positive_feedback = [f for f in self.feedback_history if f.get('positive', False)]
        
        if not positive_feedback:
            return {'message': 'Нет позитивной обратной связи для анализа'}
        
        # Статистика по стратегиям с позитивной обратной связью
        best_strategies = {}
        for feedback in positive_feedback:
            strategy = feedback.get('strategy_used')
            if strategy:
                if strategy not in best_strategies:
                    best_strategies[strategy] = 0
                best_strategies[strategy] += 1
        
        # Рекомендации по оптимизации
        recommendations = []
        
        # Лучшая стратегия
        if best_strategies:
            best_strategy = max(best_strategies, key=best_strategies.get)
            recommendations.append(f"Наиболее успешная стратегия: {best_strategy}")
        
        # Анализ времени генерации
        fast_positive = [f for f in positive_feedback if f.get('generation_time', 0) < 2.0]
        if len(fast_positive) / len(positive_feedback) > 0.7:
            recommendations.append("Быстрая генерация коррелирует с положительной обратной связью")
        
        # Анализ качества
        high_quality = [f for f in positive_feedback if f.get('quality_score', 0) > 0.8]
        if len(high_quality) / len(positive_feedback) > 0.6:
            recommendations.append("Высокие оценки качества коррелируют с положительной обратной связью")
        
        return {
            'analyzed_feedback_count': len(positive_feedback),
            'best_strategies': best_strategies,
            'recommendations': recommendations,
            'optimization_applied': True
        }
    
    async def clear_cache(self, max_age_hours: int = 24):
        """Очистка устаревшего кэша"""
        initial_size = len(self.response_cache)
        
        current_time = datetime.now()
        fresh_cache = {}
        
        for key, value in self.response_cache.items():
            timestamp = value.get('timestamp')
            if timestamp and (current_time - timestamp).total_seconds() / 3600 < max_age_hours:
                fresh_cache[key] = value
        
        self.response_cache = fresh_cache
        cleared_count = initial_size - len(fresh_cache)
        
        self.logger.info(f"Очищено {cleared_count} устаревших записей из кэша")
        
        return {
            'initial_cache_size': initial_size,
            'cleared_entries': cleared_count,
            'current_cache_size': len(fresh_cache)
        }

_gemini_client_instance = None

def get_gemini_client() -> AdvancedGeminiClient:
    global _gemini_client_instance
    if _gemini_client_instance is None:
        _gemini_client_instance = AdvancedGeminiClient()
    return _gemini_client_instance