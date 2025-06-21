"""
Продвинутый RAG конвейер с адаптивным обучением и интеллектуальной маршрутизацией
"""

import json
import uuid
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from config import *
from src.data_loader import SimpleDataLoader
from src.gemini_client import AdvancedGeminiClient
from src.web_search import IntelligentWebSearch
from src.feedback_system import AdvancedFeedbackSystem

class QueryType(Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    TROUBLESHOOTING = "troubleshooting"

class ResponseQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    FAILED = "failed"

@dataclass
class QueryIntent:
    intent_type: QueryType
    confidence: float
    entities: List[str]
    keywords: List[str]
    complexity_score: float
    domain: str

@dataclass
class ContextScore:
    relevance: float
    freshness: float
    authority: float
    completeness: float
    total_score: float

@dataclass
class EnhancedInteraction:
    interaction_id: str
    user_id: str
    session_id: str
    query: str
    processed_query: str
    intent: QueryIntent
    retrieved_docs: List[Dict[str, Any]]
    context_scores: List[ContextScore]
    generated_response: str
    response_quality: ResponseQuality
    confidence_score: float
    generation_time: float
    feedback: Optional[Dict] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

class QueryClassifier(nn.Module):
    """Нейронная сеть для классификации запросов"""
    
    def __init__(self, embedding_dim: int = 384, num_classes: int = 6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.classifier(x)

class AdaptiveRetriever:
    """Адаптивный ретривер с обучением на основе обратной связи"""
    
    def __init__(self, embedding_model: SentenceTransformer, collection: chromadb.Collection):
        self.embedding_model = embedding_model
        self.collection = collection
        self.query_patterns = defaultdict(list)
        self.performance_history = deque(maxlen=1000)
        self.domain_weights = defaultdict(lambda: 1.0)
        
    async def retrieve_documents(self, query: str, intent: QueryIntent, 
                               top_k: int = 10) -> Tuple[List[Dict], List[ContextScore]]:
        """Адаптивный поиск документов с учетом интента и обратной связи"""
        
        # 1. Расширение запроса на основе интента
        expanded_query = await self._expand_query(query, intent)
        
        # 2. Гибридный поиск (семантический + ключевые слова)
        semantic_results = await self._semantic_search(expanded_query, top_k)
        keyword_results = await self._keyword_search(query, intent.keywords, top_k // 2)
        
        # 3. Объединение и ранжирование результатов
        combined_docs = self._merge_results(semantic_results, keyword_results)
        
        # 4. Адаптивное ранжирование на основе обратной связи
        ranked_docs = await self._adaptive_ranking(combined_docs, intent, query)
        
        # 5. Вычисление контекстных оценок
        context_scores = await self._calculate_context_scores(ranked_docs, intent)
        
        return ranked_docs[:top_k], context_scores
    
    async def _expand_query(self, query: str, intent: QueryIntent) -> str:
        """Расширение запроса на основе интента и домена"""
        expansions = []
        
        # Расширение по интенту
        intent_expansions = {
            QueryType.FACTUAL: ["что такое", "определение", "информация"],
            QueryType.PROCEDURAL: ["как", "процедура", "шаги", "инструкция"],
            QueryType.COMPARATIVE: ["сравнение", "различия", "сходства"],
            QueryType.ANALYTICAL: ["анализ", "причины", "последствия"],
            QueryType.TROUBLESHOOTING: ["проблема", "решение", "устранение"]
        }
        
        if intent.intent_type in intent_expansions:
            expansions.extend(intent_expansions[intent.intent_type])
        
        # Добавление доменных терминов
        if intent.domain == "government":
            expansions.extend(["государственный", "услуга", "орган", "процедура"])
        elif intent.domain == "technical":
            expansions.extend(["система", "технология", "конфигурация"])
        
        # Объединение с оригинальным запросом
        expanded = f"{query} {' '.join(expansions[:3])}"
        return expanded
    
    async def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Семантический поиск"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        docs = []
        for i in range(len(results['documents'][0])):
            docs.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'semantic_score': 1 - results['distances'][0][i],
                'search_type': 'semantic'
            })
        
        return docs
    
    async def _keyword_search(self, query: str, keywords: List[str], top_k: int) -> List[Dict]:
        """Поиск по ключевым словам с TF-IDF"""
        # Получение всех документов (в реальности нужна оптимизация)
        all_docs = self.collection.get(include=['documents', 'metadatas'])
        
        if not all_docs['documents']:
            return []
        
        # TF-IDF векторизация
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        doc_vectors = vectorizer.fit_transform(all_docs['documents'])
        
        # Векторизация запроса
        query_vector = vectorizer.transform([query])
        
        # Вычисление схожести
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        # Получение топ результатов
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        docs = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Минимальный порог
                docs.append({
                    'content': all_docs['documents'][idx],
                    'metadata': all_docs['metadatas'][idx],
                    'keyword_score': float(similarities[idx]),
                    'search_type': 'keyword'
                })
        
        return docs
    
    def _merge_results(self, semantic_docs: List[Dict], keyword_docs: List[Dict]) -> List[Dict]:
        """Объединение результатов семантического и ключевого поиска"""
        # Создание индекса для избежания дубликатов
        seen_content = set()
        merged_docs = []
        
        # Добавление семантических результатов
        for doc in semantic_docs:
            content_hash = hashlib.md5(doc['content'].encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                merged_docs.append(doc)
        
        # Добавление ключевых результатов
        for doc in keyword_docs:
            content_hash = hashlib.md5(doc['content'].encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                merged_docs.append(doc)
        
        return merged_docs
    
    async def _adaptive_ranking(self, docs: List[Dict], intent: QueryIntent, query: str) -> List[Dict]:
        """Адаптивное ранжирование на основе обратной связи"""
        
        for doc in docs:
            # Базовый скор
            base_score = doc.get('semantic_score', 0) * 0.7 + doc.get('keyword_score', 0) * 0.3
            
            # Адаптация на основе домена
            domain_weight = self.domain_weights.get(intent.domain, 1.0)
            
            # Адаптация на основе типа интента
            intent_boost = self._get_intent_boost(doc, intent)
            
            # Адаптация на основе исторической производительности
            historical_boost = self._get_historical_boost(doc, query)
            
            # Финальный скор
            doc['final_score'] = base_score * domain_weight * intent_boost * historical_boost
        
        # Сортировка по финальному скору
        return sorted(docs, key=lambda x: x.get('final_score', 0), reverse=True)
    
    def _get_intent_boost(self, doc: Dict, intent: QueryIntent) -> float:
        """Буст на основе соответствия интенту"""
        doc_type = doc.get('metadata', {}).get('type', '')
        
        intent_preferences = {
            QueryType.FACTUAL: ['definition', 'encyclopedia', 'reference'],
            QueryType.PROCEDURAL: ['manual', 'guide', 'instruction'],
            QueryType.COMPARATIVE: ['comparison', 'analysis', 'review'],
            QueryType.ANALYTICAL: ['research', 'study', 'analysis'],
            QueryType.TROUBLESHOOTING: ['faq', 'support', 'troubleshoot']
        }
        
        preferred_types = intent_preferences.get(intent.intent_type, [])
        return 1.2 if any(ptype in doc_type.lower() for ptype in preferred_types) else 1.0
    
    def _get_historical_boost(self, doc: Dict, query: str) -> float:
        """Буст на основе исторической производительности"""
        # Простая эвристика - в реальности нужна более сложная логика
        source = doc.get('metadata', {}).get('source', '')
        
        # Анализ производительности по источникам
        source_performance = defaultdict(lambda: {'positive': 0, 'total': 0})
        
        for interaction in self.performance_history:
            if interaction.get('source') == source:
                source_performance[source]['total'] += 1
                if interaction.get('feedback_positive', False):
                    source_performance[source]['positive'] += 1
        
        if source_performance[source]['total'] > 5:
            success_rate = source_performance[source]['positive'] / source_performance[source]['total']
            return 0.8 + 0.4 * success_rate  # Диапазон 0.8-1.2
        
        return 1.0
    
    async def _calculate_context_scores(self, docs: List[Dict], intent: QueryIntent) -> List[ContextScore]:
        """Вычисление детальных оценок контекста"""
        scores = []
        
        for doc in docs:
            relevance = doc.get('final_score', 0)
            
            # Оценка свежести
            freshness = self._calculate_freshness(doc)
            
            # Оценка авторитетности
            authority = self._calculate_authority(doc)
            
            # Оценка полноты
            completeness = self._calculate_completeness(doc, intent)
            
            # Общая оценка
            total_score = (relevance * 0.4 + freshness * 0.2 + 
                          authority * 0.2 + completeness * 0.2)
            
            scores.append(ContextScore(
                relevance=relevance,
                freshness=freshness,
                authority=authority,
                completeness=completeness,
                total_score=total_score
            ))
        
        return scores
    
    def _calculate_freshness(self, doc: Dict) -> float:
        """Оценка свежести документа"""
        metadata = doc.get('metadata', {})
        
        # Если есть дата создания/обновления
        if 'date' in metadata:
            try:
                doc_date = datetime.fromisoformat(metadata['date'])
                days_old = (datetime.now() - doc_date).days
                
                # Экспоненциальное затухание
                freshness = np.exp(-days_old / 365)  # Полураспад = 1 год
                return max(0.1, min(1.0, freshness))
            except:
                pass
        
        # Если дата неизвестна, средняя оценка
        return 0.5
    
    def _calculate_authority(self, doc: Dict) -> float:
        """Оценка авторитетности источника"""
        metadata = doc.get('metadata', {})
        source = metadata.get('source', '').lower()
        
        # Авторитетные источники для казахстанского контекста
        authority_scores = {
            'gov.kz': 0.95,
            'egov.kz': 0.9,
            'zakon.kz': 0.85,
            'adilet.zan.kz': 0.9,
            'mdai.gov.kz': 0.85,
            'official': 0.8,
            'government': 0.8
        }
        
        for pattern, score in authority_scores.items():
            if pattern in source:
                return score
        
        # Базовая оценка для неизвестных источников
        return 0.6
    
    def _calculate_completeness(self, doc: Dict, intent: QueryIntent) -> float:
        """Оценка полноты информации"""
        content = doc.get('content', '')
        content_length = len(content)
        
        # Базовая оценка на основе длины
        length_score = min(1.0, content_length / 1000)  # Нормализация к 1000 символам
        
        # Адаптация к типу интента
        intent_requirements = {
            QueryType.FACTUAL: 500,      # Краткие факты
            QueryType.PROCEDURAL: 1500,  # Детальные инструкции
            QueryType.COMPARATIVE: 1000, # Средний объем
            QueryType.ANALYTICAL: 2000,  # Подробный анализ
            QueryType.TROUBLESHOOTING: 800  # Конкретные решения
        }
        
        required_length = intent_requirements.get(intent.intent_type, 1000)
        completeness = min(1.0, content_length / required_length)
        
        return max(0.2, completeness)
    
    def update_performance(self, interaction: EnhancedInteraction, feedback_positive: bool):
        """Обновление метрик производительности"""
        performance_record = {
            'query': interaction.query,
            'intent': interaction.intent.intent_type.value,
            'source': interaction.retrieved_docs[0].get('metadata', {}).get('source', '') if interaction.retrieved_docs else '',
            'feedback_positive': feedback_positive,
            'timestamp': interaction.timestamp
        }
        
        self.performance_history.append(performance_record)
        
        # Обновление весов доменов
        if feedback_positive:
            self.domain_weights[interaction.intent.domain] *= 1.01
        else:
            self.domain_weights[interaction.intent.domain] *= 0.99
        
        # Ограничение весов
        for domain in self.domain_weights:
            self.domain_weights[domain] = max(0.5, min(2.0, self.domain_weights[domain]))

class AdvancedRAGPipeline:
    """Продвинутый RAG конвейер с адаптивным обучением"""
    
    def __init__(self):
        # Базовые компоненты
        self.logger = self._setup_logging()
        self.data_loader = SimpleDataLoader()
        self.gemini_client = AdvancedGeminiClient()
        self.web_search = IntelligentWebSearch()
        self.feedback_system = AdvancedFeedbackSystem()
        
        # ML компоненты
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.query_classifier = QueryClassifier()
        
        # Векторная база данных
        self.chroma_client = chromadb.PersistentClient(path=str(MODELS_DIR / "chroma"))
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_documents_advanced",
            metadata={"description": "Advanced RAG documents with adaptive learning"}
        )
        
        # Адаптивный ретривер
        self.retriever = AdaptiveRetriever(self.embedding_model, self.collection)
        
        # Текстовый сплиттер
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        
        # Кэши и состояние
        self.interactions_cache = {}
        self.session_contexts = defaultdict(list)
        self.user_profiles = defaultdict(dict)
        self.performance_metrics = defaultdict(float)
        
        # Обучение классификатора запросов
        self._initialize_query_classifier()
        
        self.logger.info("Advanced RAG Pipeline инициализирован")
    
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('AdvancedRAG')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_query_classifier(self):
        """Инициализация классификатора запросов"""
        # В реальности здесь должна быть загрузка предобученной модели
        # Пока используем случайные веса
        self.query_classifier.load_state_dict(
            torch.load(MODELS_DIR / "query_classifier.pth", map_location='cpu')
            if (MODELS_DIR / "query_classifier.pth").exists()
            else self.query_classifier.state_dict()
        )
        self.query_classifier.eval()
    
    async def process_query(self, query: str, user_id: str, 
                          session_id: str = None) -> Dict[str, Any]:
        """Продвинутая обработка пользовательского запроса"""
        start_time = datetime.now()
        interaction_id = str(uuid.uuid4())
        
        if not session_id:
            session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d')}"
        
        try:
            # 1. Анализ интента запроса
            intent = await self._analyze_query_intent(query, user_id, session_id)
            
            # 2. Обработка и расширение запроса
            processed_query = await self._preprocess_query(query, intent, session_id)
            
            # 3. Адаптивный поиск документов
            docs, context_scores = await self.retriever.retrieve_documents(
                processed_query, intent, top_k=TOP_K_DOCUMENTS
            )
            
            # 4. Расширение контекста веб-поиском (если необходимо)
            if await self._needs_web_enhancement(docs, intent):
                web_docs = await self.web_search.intelligent_search(
                    query, intent, max_results=3
                )
                docs.extend(web_docs)
            
            # 5. Контекстно-зависимая генерация ответа
            response = await self._generate_contextual_response(
                query, processed_query, docs, intent, session_id
            )
            
            # 6. Оценка качества ответа
            quality = await self._assess_response_quality(query, response, docs)
            
            # 7. Вычисление метрик уверенности
            confidence = self._calculate_confidence(docs, context_scores, quality)
            
            # 8. Создание расширенного взаимодействия
            interaction = EnhancedInteraction(
                interaction_id=interaction_id,
                user_id=user_id,
                session_id=session_id,
                query=query,
                processed_query=processed_query,
                intent=intent,
                retrieved_docs=docs,
                context_scores=context_scores,
                generated_response=response,
                response_quality=quality,
                confidence_score=confidence,
                generation_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(),
                metadata={
                    'model_version': '1.0',
                    'user_agent': 'advanced_rag',
                    'processing_steps': 8
                }
            )
            
            # 9. Сохранение взаимодействия
            await self._save_interaction(interaction)
            
            # 10. Обновление пользовательского профиля
            await self._update_user_profile(user_id, interaction)
            
            # 11. Формирование ответа
            return {
                'response': response,
                'interaction_id': interaction_id,
                'confidence': confidence,
                'intent': intent.intent_type.value,
                'sources': self._extract_sources(docs),
                'processing_time': interaction.generation_time,
                'quality_assessment': quality.value,
                'suggestions': await self._generate_suggestions(intent, docs),
                'follow_up_questions': await self._generate_follow_up_questions(query, intent)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса {interaction_id}: {str(e)}")
            return {
                'response': f"Извините, произошла ошибка при обработке запроса: {str(e)}",
                'interaction_id': interaction_id,
                'error': str(e),
                'confidence': 0.0
            }
    
    async def _analyze_query_intent(self, query: str, user_id: str, 
                                  session_id: str) -> QueryIntent:
        """Анализ интента пользовательского запроса"""
        
        # 1. Векторизация запроса
        query_embedding = self.embedding_model.encode([query])
        
        # 2. Классификация типа запроса
        with torch.no_grad():
            query_tensor = torch.FloatTensor(query_embedding)
            intent_probs = self.query_classifier(query_tensor)
            intent_type_idx = torch.argmax(intent_probs, dim=1).item()
            confidence = float(torch.max(intent_probs))
        
        intent_types = list(QueryType)
        intent_type = intent_types[intent_type_idx]
        
        # 3. Извлечение сущностей и ключевых слов
        entities = await self._extract_entities(query)
        keywords = await self._extract_keywords(query)
        
        # 4. Оценка сложности
        complexity_score = self._calculate_complexity(query, entities, keywords)
        
        # 5. Определение домена
        domain = await self._identify_domain(query, entities)
        
        return QueryIntent(
            intent_type=intent_type,
            confidence=confidence,
            entities=entities,
            keywords=keywords,
            complexity_score=complexity_score,
            domain=domain
        )
    
    async def _extract_entities(self, query: str) -> List[str]:
        """Извлечение именованных сущностей"""
        # Простая эвристика для извлечения сущностей
        # В реальности здесь должна быть NER модель
        
        kazakh_entities = [
            'Казахстан', 'Алматы', 'Астана', 'Нур-Султан',
            'егов', 'цон', 'акимат', 'министерство'
        ]
        
        entities = []
        query_lower = query.lower()
        
        for entity in kazakh_entities:
            if entity.lower() in query_lower:
                entities.append(entity)
        
        # Поиск номеров, дат, организаций
        import re
        
        # Номера документов
        doc_numbers = re.findall(r'\b\d{12}\b|\b\d{8}\b', query)
        entities.extend(doc_numbers)
        
        # Даты
        dates = re.findall(r'\d{1,2}\.\d{1,2}\.\d{4}|\d{4}-\d{1,2}-\d{1,2}', query)
        entities.extend(dates)
        
        return entities
    
    async def _extract_keywords(self, query: str) -> List[str]:
        """Извлечение ключевых слов"""
        # Простое извлечение ключевых слов
        stopwords = {
            'и', 'или', 'но', 'а', 'в', 'на', 'с', 'по', 'для', 'от', 'до',
            'как', 'что', 'где', 'когда', 'почему', 'кто', 'какой', 'какая'
        }
        
        words = query.lower().split()
        keywords = [word for word in words if len(word) > 3 and word not in stopwords]
        
        return keywords[:10]  # Топ-10 ключевых слов
    
    def _calculate_complexity(self, query: str, entities: List[str], 
                            keywords: List[str]) -> float:
        """Вычисление сложности запроса"""
        # Базовые метрики
        length_score = min(1.0, len(query) / 200)
        entities_score = min(1.0, len(entities) / 5)
        keywords_score = min(1.0, len(keywords) / 10)
        
        # Сложные паттерны
        complex_patterns = [
            'сравни', 'различия', 'анализ', 'причины', 'последствия',
            'почему', 'как происходит', 'что будет если'
        ]
        
        pattern_score = sum(1 for pattern in complex_patterns 
                          if pattern in query.lower()) / len(complex_patterns)
        
        # Общая сложность
        complexity = (length_score * 0.3 + entities_score * 0.2 + 
                     keywords_score * 0.2 + pattern_score * 0.3)
        
        return complexity
    
    async def _identify_domain(self, query: str, entities: List[str]) -> str:
        """Определение домена запроса"""
        domain_keywords = {
            'government': [
                'государственный', 'услуга', 'документ', 'справка', 'заявление',
                'цон', 'егов', 'акимат', 'министерство', 'ведомство'
            ],
            'legal': [
                'закон', 'право', 'юридический', 'кодекс', 'статья', 'норма'
            ],
            'technical': [
                'система', 'технология', 'программа', 'сайт', 'приложение',
                'интернет', 'компьютер', 'цифровой'
            ],
            'business': [
                'бизнес', 'предприятие', 'налог', 'лицензия', 'регистрация'
            ],
            'social': [
                'социальный', 'пенсия', 'пособие', 'льгота', 'медицина'
            ]
        }
        
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[domain] = score
        
        # Возвращаем домен с максимальным скором
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'general'
    
    async def _preprocess_query(self, query: str, intent: QueryIntent, 
                              session_id: str) -> str:
        """Обработка и расширение запроса"""
        
        # 1. Базовая очистка
        processed = query.strip()
        
        # 2. Контекстуализация на основе сессии
        session_context = self.session_contexts.get(session_id, [])
        if session_context:
            # Добавление контекста из предыдущих запросов
            recent_topics = [ctx.get('topic') for ctx in session_context[-3:]]
            context_keywords = ' '.join(filter(None, recent_topics))
            if context_keywords:
                processed = f"{processed} {context_keywords}"
        
        # 3. Расширение на основе интента
        if intent.intent_type == QueryType.PROCEDURAL:
            processed = f"процедура инструкция {processed}"
        elif intent.intent_type == QueryType.COMPARATIVE:
            processed = f"сравнение различия {processed}"
        elif intent.intent_type == QueryType.TROUBLESHOOTING:
            processed = f"проблема решение {processed}"
        
        # 4. Добавление доменной специфики
        if intent.domain == 'government':
            processed = f"государственная услуга {processed}"
        
        return processed
    
    async def _needs_web_enhancement(self, docs: List[Dict], intent: QueryIntent) -> bool:
        """Определение необходимости веб-поиска"""
        
        # Критерии для веб-поиска
        criteria = {
            'insufficient_docs': len(docs) < 3,
            'low_quality_docs': sum(doc.get('final_score', 0) for doc in docs) < 2.0,
            'recent_info_needed': intent.intent_type in [QueryType.FACTUAL, QueryType.ANALYTICAL],
            'high_complexity': intent.complexity_score > 0.7
        }
        
        # Веб-поиск нужен если выполняется 2+ критерия
        return sum(criteria.values()) >= 2
    
    async def _generate_contextual_response(self, original_query: str, 
                                          processed_query: str, docs: List[Dict], 
                                          intent: QueryIntent, session_id: str) -> str:
        """Контекстно-зависимая генерация ответа"""
        
        # Подготовка контекста
        context = self._prepare_enhanced_context(docs, intent)
        
        # Получение истории сессии
        session_history = self.session_contexts.get(session_id, [])
        
        # Генерация с учетом интента и контекста
        response = await self.gemini_client.generate_advanced_response(
            query=original_query,
            processed_query=processed_query,
            context=context,
            intent=intent,
            session_history=session_history[-3:]  # Последние 3 взаимодействия
        )
        
        return response
    
    def _prepare_enhanced_context(self, docs: List[Dict], intent: QueryIntent) -> str:
        """Подготовка расширенного контекста"""
        context_parts = []
        
        for i, doc in enumerate(docs[:5]):
            source = doc.get('metadata', {}).get('source', f'Источник {i+1}')
            title = doc.get('metadata', {}).get('title', f'Документ {i+1}')
            content = doc['content']
            score = doc.get('final_score', 0)
            
            # Адаптация длины в зависимости от интента
            max_length = {
                QueryType.FACTUAL: 300,
                QueryType.PROCEDURAL: 500,
                QueryType.COMPARATIVE: 400,
                QueryType.ANALYTICAL: 600,
                QueryType.TROUBLESHOOTING: 350
            }.get(intent.intent_type, 400)
            
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            context_parts.append(
                f"[Источник {i+1}: {source} | Заголовок: {title} | Релевантность: {score:.2f}]\n{content}"
            )
        
        return "\n\n".join(context_parts)
    
    async def _assess_response_quality(self, query: str, response: str, 
                                     docs: List[Dict]) -> ResponseQuality:
        """Оценка качества сгенерированного ответа"""
        
        # Метрики качества
        length_appropriate = 50 <= len(response) <= 2000
        contains_facts = any(doc.get('final_score', 0) > 0.7 for doc in docs)
        coherent = len(response.split('.')) >= 2  # Минимальная связность
        
        # Проверка на галлюцинации (базовая)
        hallucination_indicators = [
            'точно знаю', 'абсолютно уверен', 'всегда так',
            'никогда не бывает', 'невозможно'
        ]
        has_hallucinations = any(indicator in response.lower() 
                               for indicator in hallucination_indicators)
        
        # Оценка полноты
        query_keywords = set(query.lower().split())
        response_keywords = set(response.lower().split())
        keyword_coverage = len(query_keywords & response_keywords) / len(query_keywords)
        
        # Итоговая оценка
        quality_score = 0
        
        if length_appropriate:
            quality_score += 1
        if contains_facts:
            quality_score += 2
        if coherent:
            quality_score += 1
        if not has_hallucinations:
            quality_score += 1
        if keyword_coverage > 0.3:
            quality_score += 1
        
        # Маппинг скора к качеству
        if quality_score >= 5:
            return ResponseQuality.EXCELLENT
        elif quality_score >= 4:
            return ResponseQuality.GOOD
        elif quality_score >= 3:
            return ResponseQuality.AVERAGE
        elif quality_score >= 2:
            return ResponseQuality.POOR
        else:
            return ResponseQuality.FAILED
    
    def _calculate_confidence(self, docs: List[Dict], context_scores: List[ContextScore], 
                            quality: ResponseQuality) -> float:
        """Вычисление общей уверенности в ответе"""
        
        # Уверенность на основе документов
        doc_confidence = np.mean([doc.get('final_score', 0) for doc in docs]) if docs else 0
        
        # Уверенность на основе контекстных оценок
        context_confidence = np.mean([score.total_score for score in context_scores]) if context_scores else 0
        
        # Уверенность на основе качества
        quality_confidence = {
            ResponseQuality.EXCELLENT: 0.95,
            ResponseQuality.GOOD: 0.8,
            ResponseQuality.AVERAGE: 0.6,
            ResponseQuality.POOR: 0.4,
            ResponseQuality.FAILED: 0.1
        }[quality]
        
        # Взвешенная комбинация
        total_confidence = (doc_confidence * 0.4 + 
                          context_confidence * 0.3 + 
                          quality_confidence * 0.3)
        
        return max(0.0, min(1.0, total_confidence))
    
    async def _save_interaction(self, interaction: EnhancedInteraction):
        """Сохранение расширенного взаимодействия"""
        # Сохранение в кэш
        self.interactions_cache[interaction.interaction_id] = interaction
        
        # Добавление в контекст сессии
        session_context = {
            'interaction_id': interaction.interaction_id,
            'query': interaction.query,
            'intent': interaction.intent.intent_type.value,
            'topic': interaction.intent.domain,
            'timestamp': interaction.timestamp.isoformat()
        }
        
        self.session_contexts[interaction.session_id].append(session_context)
        
        # Ограничение размера контекста сессии
        if len(self.session_contexts[interaction.session_id]) > 20:
            self.session_contexts[interaction.session_id] = \
                self.session_contexts[interaction.session_id][-20:]
        
        # Логирование
        self.logger.info(
            f"Сохранено взаимодействие {interaction.interaction_id} "
            f"для пользователя {interaction.user_id}"
        )
    
    async def _update_user_profile(self, user_id: str, interaction: EnhancedInteraction):
        """Обновление профиля пользователя"""
        profile = self.user_profiles[user_id]
        
        # Обновление статистики
        profile['total_queries'] = profile.get('total_queries', 0) + 1
        profile['last_activity'] = interaction.timestamp.isoformat()
        
        # Обновление предпочтений по интентам
        intent_prefs = profile.setdefault('intent_preferences', {})
        intent_type = interaction.intent.intent_type.value
        intent_prefs[intent_type] = intent_prefs.get(intent_type, 0) + 1
        
        # Обновление предпочтений по доменам
        domain_prefs = profile.setdefault('domain_preferences', {})
        domain = interaction.intent.domain
        domain_prefs[domain] = domain_prefs.get(domain, 0) + 1
        
        # Средние метрики
        profile['avg_confidence'] = (
            profile.get('avg_confidence', 0.5) * 0.9 + 
            interaction.confidence_score * 0.1
        )
        
        # Сложность запросов
        profile['avg_complexity'] = (
            profile.get('avg_complexity', 0.5) * 0.9 + 
            interaction.intent.complexity_score * 0.1
        )
    
    def _extract_sources(self, docs: List[Dict]) -> List[str]:
        """Извлечение источников с дополнительной информацией"""
        sources = []
        
        for doc in docs:
            metadata = doc.get('metadata', {})
            source = metadata.get('source', 'Неизвестный источник')
            title = metadata.get('title', '')
            
            if title:
                sources.append(f"{source} - {title}")
            else:
                sources.append(source)
        
        return list(dict.fromkeys(sources))  # Удаление дубликатов с сохранением порядка
    
    async def _generate_suggestions(self, intent: QueryIntent, docs: List[Dict]) -> List[str]:
        """Генерация предложений для пользователя"""
        suggestions = []
        
        # Предложения на основе интента
        if intent.intent_type == QueryType.PROCEDURAL:
            suggestions.append("Хотите узнать о сроках выполнения процедуры?")
            suggestions.append("Нужна информация о необходимых документах?")
        elif intent.intent_type == QueryType.FACTUAL:
            suggestions.append("Хотите узнать более подробную информацию?")
            suggestions.append("Интересуют связанные вопросы?")
        
        # Предложения на основе доступных документов
        if docs:
            related_topics = set()
            for doc in docs[:3]:
                metadata = doc.get('metadata', {})
                category = metadata.get('category', '')
                if category:
                    related_topics.add(category)
            
            for topic in list(related_topics)[:2]:
                suggestions.append(f"Узнать больше о: {topic}")
        
        return suggestions[:3]  # Максимум 3 предложения
    
    async def _generate_follow_up_questions(self, query: str, intent: QueryIntent) -> List[str]:
        """Генерация уточняющих вопросов"""
        follow_ups = []
        
        # Базовые уточняющие вопросы по интенту
        intent_follow_ups = {
            QueryType.FACTUAL: [
                "Хотите узнать историю этого вопроса?",
                "Интересуют актуальные изменения?"
            ],
            QueryType.PROCEDURAL: [
                "Нужна информация о сроках?",
                "Хотите знать об исключениях?"
            ],
            QueryType.COMPARATIVE: [
                "Интересуют преимущества и недостатки?",
                "Хотите сравнить с альтернативами?"
            ]
        }
        
        follow_ups.extend(intent_follow_ups.get(intent.intent_type, []))
        
        # Доменные уточнения
        if intent.domain == 'government':
            follow_ups.append("Нужна помощь с подачей документов?")
        
        return follow_ups[:2]  # Максимум 2 вопроса
    
    async def submit_feedback(self, interaction_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Продвинутая обработка обратной связи"""
        if interaction_id not in self.interactions_cache:
            return {'error': 'Взаимодействие не найдено'}
        
        interaction = self.interactions_cache[interaction_id]
        
        # Сохранение обратной связи
        result = await self.feedback_system.process_advanced_feedback(
            interaction, feedback_data
        )
        
        if result.get('validated', False):
            # Обновление ретривера
            feedback_positive = feedback_data.get('rating', 0) > 3 or \
                              feedback_data.get('feedback_type') == 'like'
            
            self.retriever.update_performance(interaction, feedback_positive)
            
            # Адаптивное обучение
            await self._trigger_adaptive_learning()
        
        return result
    
    async def _trigger_adaptive_learning(self):
        """Запуск адаптивного обучения"""
        # Получение обратной связи для обучения
        training_feedback = await self.feedback_system.get_training_data()
        
        if len(training_feedback) >= MIN_FEEDBACK_FOR_TRAINING:
            self.logger.info(f"Запуск адаптивного обучения с {len(training_feedback)} примерами")
            
            # Обучение классификатора запросов
            await self._retrain_query_classifier(training_feedback)
            
            # Обновление метрик производительности
            await self._update_performance_metrics(training_feedback)
            
            # Отметка обратной связи как использованной
            await self.feedback_system.mark_feedback_used(training_feedback)
    
    async def _retrain_query_classifier(self, training_data: List[Dict]):
        """Переобучение классификатора запросов"""
        # Здесь должна быть логика переобучения
        # Пока просто логируем
        self.logger.info(f"Переобучение классификатора с {len(training_data)} примерами")
        
        # Сохранение обновленной модели
        torch.save(
            self.query_classifier.state_dict(),
            MODELS_DIR / "query_classifier.pth"
        )
    
    async def _update_performance_metrics(self, training_data: List[Dict]):
        """Обновление метрик производительности"""
        for data in training_data:
            intent_type = data.get('intent_type', 'unknown')
            feedback_positive = data.get('feedback_positive', False)
            
            # Обновление метрик по типам интентов
            current_metric = self.performance_metrics.get(intent_type, 0.5)
            
            if feedback_positive:
                self.performance_metrics[intent_type] = current_metric * 0.95 + 0.05
            else:
                self.performance_metrics[intent_type] = current_metric * 0.95
    
    def load_and_index_documents(self):
        """Загрузка и индексация документов с расширенными метаданными"""
        self.logger.info("Загрузка документов...")
        
        # Загрузка документов
        all_documents = []
        
        # Локальные документы
        local_docs = self.data_loader.load_documents_from_folder()
        all_documents.extend(local_docs)
        
        # Примерные данные
        sample_docs = self.data_loader.load_sample_kazakh_gov_data()
        all_documents.extend(sample_docs)
        
        if all_documents:
            self._index_documents_advanced(all_documents)
            self.logger.info(f"Проиндексировано документов: {len(all_documents)}")
        else:
            self.logger.warning("Документы не найдены")
    
    def _index_documents_advanced(self, documents: List[Dict[str, Any]]):
        """Продвинутая индексация документов"""
        texts = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(documents):
            # Разбиение на чанки
            chunks = self.text_splitter.split_text(doc['content'])
            
            for j, chunk in enumerate(chunks):
                chunk_id = f"doc_{i}_chunk_{j}_{uuid.uuid4().hex[:8]}"
                
                # Расширенные метаданные
                enhanced_metadata = {
                    **doc['metadata'],
                    'chunk_id': chunk_id,
                    'doc_index': i,
                    'chunk_index': j,
                    'chunk_length': len(chunk),
                    'index_timestamp': datetime.now().isoformat(),
                    'quality_score': self._assess_chunk_quality(chunk),
                    'estimated_domain': self._estimate_chunk_domain(chunk)
                }
                
                texts.append(chunk)
                metadatas.append(enhanced_metadata)
                ids.append(chunk_id)
        
        # Генерация эмбеддингов батчами
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts).tolist()
            all_embeddings.extend(batch_embeddings)
        
        # Добавление в ChromaDB
        self.collection.add(
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        self.logger.info(f"Добавлено чанков в векторную базу: {len(texts)}")
    
    def _assess_chunk_quality(self, chunk: str) -> float:
        """Оценка качества чанка"""
        # Простые метрики качества
        length_score = min(1.0, len(chunk) / 500)  # Оптимальная длина 500 символов
        
        # Плотность информации (соотношение букв к пробелам)
        alpha_chars = sum(c.isalpha() for c in chunk)
        info_density = alpha_chars / len(chunk) if chunk else 0
        
        # Наличие структуры (точки, запятые)
        structure_score = min(1.0, (chunk.count('.') + chunk.count(',')) / 10)
        
        # Общая оценка
        quality = (length_score * 0.4 + info_density * 0.4 + structure_score * 0.2)
        return max(0.1, min(1.0, quality))
    
    def _estimate_chunk_domain(self, chunk: str) -> str:
        """Оценка домена чанка"""
        domain_keywords = {
            'government': ['государственный', 'услуга', 'орган', 'министерство'],
            'legal': ['закон', 'право', 'статья', 'кодекс'],
            'technical': ['система', 'технология', 'программа'],
            'business': ['бизнес', 'предприятие', 'налог'],
            'social': ['социальный', 'пенсия', 'медицина']
        }
        
        chunk_lower = chunk.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in chunk_lower)
            domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'general'
    
    async def get_advanced_stats(self) -> Dict[str, Any]:
        """Получение расширенной статистики системы"""
        # Базовые метрики
        total_docs = self.collection.count()
        total_interactions = len(self.interactions_cache)
        
        # Метрики производительности
        recent_interactions = [
            interaction for interaction in self.interactions_cache.values()
            if interaction.timestamp > datetime.now() - timedelta(days=7)
        ]
        
        # Средние метрики
        avg_confidence = np.mean([
            i.confidence_score for i in recent_interactions
        ]) if recent_interactions else 0
        
        avg_processing_time = np.mean([
            i.generation_time for i in recent_interactions
        ]) if recent_interactions else 0
        
        # Распределение по интентам
        intent_distribution = defaultdict(int)
        for interaction in recent_interactions:
            intent_distribution[interaction.intent.intent_type.value] += 1
        
        # Распределение по качеству
        quality_distribution = defaultdict(int)
        for interaction in recent_interactions:
            quality_distribution[interaction.response_quality.value] += 1
        
        # Активные пользователи
        active_users = len(set(i.user_id for i in recent_interactions))
        
        # Метрики обратной связи
        feedback_stats = await self.feedback_system.get_advanced_stats()
        
        return {
            'document_metrics': {
                'total_documents': total_docs,
                'last_indexed': datetime.now().isoformat()
            },
            'interaction_metrics': {
                'total_interactions': total_interactions,
                'recent_interactions': len(recent_interactions),
                'active_users': active_users,
                'avg_confidence': round(avg_confidence, 3),
                'avg_processing_time': round(avg_processing_time, 3)
            },
            'intent_distribution': dict(intent_distribution),
            'quality_distribution': dict(quality_distribution),
            'performance_metrics': dict(self.performance_metrics),
            'feedback_stats': feedback_stats,
            'system_health': {
                'memory_usage': self._get_memory_usage(),
                'cache_size': len(self.interactions_cache),
                'session_count': len(self.session_contexts)
            }
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Получение информации об использовании памяти"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
            'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
            'percent': round(process.memory_percent(), 2)
        }

# Инициализация глобального экземпляра
_pipeline_instance = None

def get_rag_pipeline() -> AdvancedRAGPipeline:
    """Получение singleton экземпляра RAG pipeline"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AdvancedRAGPipeline()
    return _pipeline_instance