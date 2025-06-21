"""
Интеллектуальная система веб-поиска с адаптивными алгоритмами и фильтрацией
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from urllib.parse import quote, urlparse
import time

from bs4 import BeautifulSoup
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config import GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID

class SearchPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ContentType(Enum):
    NEWS = "news"
    OFFICIAL = "official"
    DOCUMENTATION = "documentation"
    FORUM = "forum"
    ACADEMIC = "academic"
    GENERAL = "general"

@dataclass
class SearchQuery:
    original_query: str
    processed_query: str
    keywords: List[str]
    entities: List[str]
    domain_specific_terms: List[str]
    temporal_context: Optional[str] = None
    geographic_context: Optional[str] = None

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str
    content_type: ContentType
    authority_score: float
    relevance_score: float
    freshness_score: float
    quality_score: float
    total_score: float
    metadata: Dict[str, Any]
    extracted_at: datetime

@dataclass
class SearchMetrics:
    total_queries: int
    successful_queries: int
    average_response_time: float
    cache_hit_rate: float
    quality_distribution: Dict[str, int]
    source_distribution: Dict[str, int]

class AuthorityScorer:
    """Оценщик авторитетности источников"""
    
    def __init__(self):
        # Авторитетные домены для казахстанского контекста
        self.authority_domains = {
            # Официальные государственные сайты
            'gov.kz': 1.0,
            'egov.kz': 0.95,
            'zakon.kz': 0.9,
            'adilet.zan.kz': 0.95,
            'mdai.gov.kz': 0.9,
            'akorda.kz': 0.95,
            'nis.gov.kz': 0.85,
            'kgd.gov.kz': 0.9,
            'gov.nu': 0.9,  # Нур-Султан
            'almaty.gov.kz': 0.85,
            
            # Международные организации
            'un.org': 0.85,
            'worldbank.org': 0.8,
            'oecd.org': 0.8,
            'who.int': 0.8,
            
            # Академические и исследовательские
            'edu.kz': 0.75,
            'kaznu.kz': 0.7,
            'nu.edu.kz': 0.75,
            'kimep.kz': 0.7,
            
            # Новостные агентства
            'inform.kz': 0.6,
            'tengrinews.kz': 0.55,
            'kazpravda.kz': 0.6,
            'bna.kz': 0.5,
            
            # Международные новости
            'reuters.com': 0.7,
            'bbc.com': 0.7,
            'cnn.com': 0.65,
            
            # Технические ресурсы
            'github.com': 0.6,
            'stackoverflow.com': 0.6,
            'wikipedia.org': 0.5,
            'medium.com': 0.4
        }
        
        # Паттерны для определения типа контента
        self.content_patterns = {
            ContentType.NEWS: [
                'новости', 'сегодня', 'вчера', 'сообщает', 'заявил',
                'news', 'today', 'yesterday', 'reported', 'announced'
            ],
            ContentType.OFFICIAL: [
                'постановление', 'приказ', 'закон', 'указ', 'регламент',
                'официально', 'министерство', 'ведомство',
                'decree', 'order', 'law', 'regulation', 'official'
            ],
            ContentType.DOCUMENTATION: [
                'инструкция', 'руководство', 'документация', 'справка',
                'manual', 'guide', 'documentation', 'instructions'
            ],
            ContentType.FORUM: [
                'форум', 'обсуждение', 'комментарий', 'ответ',
                'forum', 'discussion', 'comment', 'reply'
            ],
            ContentType.ACADEMIC: [
                'исследование', 'анализ', 'научный', 'статья',
                'research', 'analysis', 'scientific', 'article', 'paper'
            ]
        }
    
    def calculate_authority_score(self, url: str, content: str, 
                                title: str) -> Tuple[float, ContentType]:
        """Вычисление оценки авторитетности и типа контента"""
        
        # Извлечение домена
        domain = self._extract_domain(url)
        
        # Базовая оценка авторитетности по домену
        base_authority = self.authority_domains.get(domain, 0.3)
        
        # Определение типа контента
        content_type = self._classify_content_type(content, title, url)
        
        # Модификация авторитетности на основе типа контента
        type_modifiers = {
            ContentType.OFFICIAL: 1.2,
            ContentType.DOCUMENTATION: 1.1,
            ContentType.ACADEMIC: 1.15,
            ContentType.NEWS: 0.9,
            ContentType.FORUM: 0.7,
            ContentType.GENERAL: 1.0
        }
        
        modified_authority = base_authority * type_modifiers.get(content_type, 1.0)
        
        # Дополнительные факторы
        if self._has_https(url):
            modified_authority *= 1.05
        
        if self._has_professional_indicators(content, title):
            modified_authority *= 1.1
        
        # Ограничение в диапазоне [0, 1]
        final_authority = max(0.0, min(1.0, modified_authority))
        
        return final_authority, content_type
    
    def _extract_domain(self, url: str) -> str:
        """Извлечение домена из URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Удаление www
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
        except:
            return ""
    
    def _classify_content_type(self, content: str, title: str, url: str) -> ContentType:
        """Классификация типа контента"""
        text = f"{title} {content}".lower()
        url_lower = url.lower()
        
        # Подсчет совпадений для каждого типа
        type_scores = {}
        
        for content_type, patterns in self.content_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text)
            
            # Бонус за URL паттерны
            url_bonus = 0
            if content_type == ContentType.NEWS and ('news' in url_lower or 'новости' in url_lower):
                url_bonus = 2
            elif content_type == ContentType.OFFICIAL and any(gov in url_lower for gov in ['gov.', 'official', 'ministry']):
                url_bonus = 3
            elif content_type == ContentType.DOCUMENTATION and ('docs' in url_lower or 'manual' in url_lower):
                url_bonus = 2
            elif content_type == ContentType.FORUM and ('forum' in url_lower or 'discussion' in url_lower):
                url_bonus = 2
            elif content_type == ContentType.ACADEMIC and ('edu' in url_lower or 'research' in url_lower):
                url_bonus = 2
            
            type_scores[content_type] = score + url_bonus
        
        # Возвращение типа с максимальным скором
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        return ContentType.GENERAL
    
    def _has_https(self, url: str) -> bool:
        """Проверка наличия HTTPS"""
        return url.lower().startswith('https://')
    
    def _has_professional_indicators(self, content: str, title: str) -> bool:
        """Проверка на профессиональные индикаторы"""
        text = f"{title} {content}".lower()
        
        professional_indicators = [
            'официальный', 'министерство', 'ведомство', 'департамент',
            'управление', 'комитет', 'агентство', 'служба',
            'official', 'ministry', 'department', 'agency', 'service',
            '©', 'copyright', 'все права защищены'
        ]
        
        return any(indicator in text for indicator in professional_indicators)

class ContentExtractor:
    """Экстрактор содержимого веб-страниц"""
    
    def __init__(self):
        self.session = None
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={'User-Agent': self.user_agent}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def extract_content(self, url: str) -> Tuple[str, str, Dict[str, Any]]:
        """Извлечение содержимого страницы"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return "", "", {}
                
                html = await response.text()
                return self._parse_html(html, url)
                
        except Exception as e:
            return "", "", {'error': str(e)}
    
    def _parse_html(self, html: str, url: str) -> Tuple[str, str, Dict[str, Any]]:
        """Парсинг HTML содержимого"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Удаление ненужных элементов
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Извлечение заголовка
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Извлечение основного контента
            content = ""
            
            # Приоритетные селекторы для основного контента
            content_selectors = [
                'main', 'article', '.content', '#content', '.main-content',
                '.post-content', '.entry-content', '.article-content'
            ]
            
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    content = content_element.get_text().strip()
                    break
            
            # Если основной контент не найден, используем body
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text().strip()
            
            # Очистка контента
            content = self._clean_text(content)
            
            # Извлечение метаданных
            metadata = self._extract_metadata(soup, url)
            
            return title, content, metadata
            
        except Exception as e:
            return "", "", {'parse_error': str(e)}
    
    def _clean_text(self, text: str) -> str:
        """Очистка текста"""
        # Удаление лишних пробелов и переносов
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Удаление повторяющихся символов
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        return text.strip()
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Извлечение метаданных страницы"""
        metadata = {'url': url}
        
        # Meta теги
        description_meta = soup.find('meta', attrs={'name': 'description'})
        if description_meta:
            metadata['description'] = description_meta.get('content', '')
        
        keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_meta:
            metadata['keywords'] = keywords_meta.get('content', '')
        
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            metadata['author'] = author_meta.get('content', '')
        
        # Open Graph теги
        og_title = soup.find('meta', property='og:title')
        if og_title:
            metadata['og_title'] = og_title.get('content', '')
        
        og_description = soup.find('meta', property='og:description')
        if og_description:
            metadata['og_description'] = og_description.get('content', '')
        
        # Дата публикации
        date_selectors = [
            'time[datetime]', '.date', '.published', '.post-date',
            '[datetime]', '.entry-date'
        ]
        
        for selector in date_selectors:
            date_element = soup.select_one(selector)
            if date_element:
                date_text = date_element.get('datetime') or date_element.get_text()
                if date_text:
                    metadata['published_date'] = date_text.strip()
                    break
        
        return metadata

class QueryOptimizer:
    """Оптимизатор поисковых запросов"""
    
    def __init__(self):
        # Словари для расширения запросов
        self.kazakh_synonyms = {
            'государственный': ['правительственный', 'официальный', 'гос'],
            'услуга': ['сервис', 'обслуживание'],
            'документ': ['справка', 'свидетельство', 'удостоверение'],
            'получить': ['оформить', 'взять', 'выдать'],
            'подать': ['отправить', 'предоставить', 'передать'],
            'цон': ['центр обслуживания населения', 'ЦОН'],
            'егов': ['электронное правительство', 'egov']
        }
        
        self.domain_keywords = {
            'government': [
                'государственный', 'министерство', 'ведомство', 'акимат',
                'цон', 'егов', 'услуга', 'документ'
            ],
            'legal': [
                'закон', 'право', 'кодекс', 'статья', 'норма',
                'юридический', 'правовой'
            ],
            'technical': [
                'система', 'технология', 'программа', 'сайт',
                'цифровой', 'электронный', 'IT'
            ]
        }
    
    def optimize_query(self, query: str, intent_data: Any = None) -> SearchQuery:
        """Оптимизация поискового запроса"""
        
        # Базовая обработка
        processed_query = query.strip()
        
        # Извлечение ключевых слов
        keywords = self._extract_keywords(query)
        
        # Извлечение сущностей
        entities = self._extract_entities(query)
        
        # Определение доменных терминов
        domain_terms = self._identify_domain_terms(query, intent_data)
        
        # Расширение запроса синонимами
        expanded_query = self._expand_with_synonyms(processed_query, keywords)
        
        # Добавление контекстуальных терминов
        contextualized_query = self._add_contextual_terms(expanded_query, intent_data)
        
        # Определение временного контекста
        temporal_context = self._extract_temporal_context(query)
        
        # Определение географического контекста
        geographic_context = self._extract_geographic_context(query)
        
        return SearchQuery(
            original_query=query,
            processed_query=contextualized_query,
            keywords=keywords,
            entities=entities,
            domain_specific_terms=domain_terms,
            temporal_context=temporal_context,
            geographic_context=geographic_context
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Извлечение ключевых слов"""
        # Стоп-слова
        stop_words = {
            'и', 'или', 'но', 'а', 'в', 'на', 'с', 'по', 'для', 'от', 'до',
            'как', 'что', 'где', 'когда', 'почему', 'кто', 'какой', 'какая',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'how', 'what', 'where', 'when', 'why', 'who', 'which'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:10]  # Ограничение количества ключевых слов
    
    def _extract_entities(self, query: str) -> List[str]:
        """Извлечение именованных сущностей"""
        entities = []
        
        # Организации и учреждения
        org_patterns = [
            r'\b(?:министерство|ведомство|департамент|управление|комитет|агентство)\s+\w+',
            r'\b(?:ЦОН|цон|егов|ЕГОВ)\b',
            r'\b(?:акимат|мэрия)\s+\w+',
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        # Номера документов
        doc_patterns = [
            r'\b\d{12}\b',  # ИИН
            r'\b\d{9}\b',   # Номер паспорта
            r'\b[A-Z]{2}\d{7}\b',  # Номер документа
        ]
        
        for pattern in doc_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Даты
        date_patterns = [
            r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        return entities
    
    def _identify_domain_terms(self, query: str, intent_data: Any = None) -> List[str]:
        """Определение доменных терминов"""
        domain_terms = []
        query_lower = query.lower()
        
        # Определение домена на основе интента или ключевых слов
        domain = 'general'
        if intent_data and hasattr(intent_data, 'domain'):
            domain = intent_data.domain
        else:
            # Простое определение домена по ключевым словам
            for domain_name, keywords in self.domain_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    domain = domain_name
                    break
        
        # Добавление доменных терминов
        if domain in self.domain_keywords:
            for keyword in self.domain_keywords[domain]:
                if keyword in query_lower:
                    domain_terms.append(keyword)
        
        return domain_terms
    
    def _expand_with_synonyms(self, query: str, keywords: List[str]) -> str:
        """Расширение запроса синонимами"""
        expanded_terms = []
        
        for keyword in keywords:
            if keyword in self.kazakh_synonyms:
                synonyms = self.kazakh_synonyms[keyword]
                expanded_terms.extend(synonyms[:2])  # Максимум 2 синонима
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        
        return query
    
    def _add_contextual_terms(self, query: str, intent_data: Any = None) -> str:
        """Добавление контекстуальных терминов"""
        contextual_terms = []
        
        # Добавление контекста на основе интента
        if intent_data:
            intent_type = getattr(intent_data, 'intent_type', None)
            if intent_type:
                intent_contexts = {
                    'procedural': ['процедура', 'инструкция', 'как получить'],
                    'factual': ['информация', 'что такое', 'определение'],
                    'comparative': ['сравнение', 'различия', 'преимущества'],
                    'troubleshooting': ['проблема', 'решение', 'помощь']
                }
                
                context_terms = intent_contexts.get(intent_type.value, [])
                contextual_terms.extend(context_terms[:1])  # Один контекстуальный термин
        
        # Добавление региональных терминов для Казахстана
        if not any(geo in query.lower() for geo in ['казахстан', 'алматы', 'астана', 'нур-султан']):
            contextual_terms.append('Казахстан')
        
        if contextual_terms:
            return f"{query} {' '.join(contextual_terms)}"
        
        return query
    
    def _extract_temporal_context(self, query: str) -> Optional[str]:
        """Извлечение временного контекста"""
        temporal_indicators = [
            'сегодня', 'вчера', 'завтра', 'сейчас', 'текущий', 'актуальный',
            'новый', 'последний', 'свежий', '2024', '2025',
            'today', 'yesterday', 'tomorrow', 'now', 'current', 'latest'
        ]
        
        query_lower = query.lower()
        for indicator in temporal_indicators:
            if indicator in query_lower:
                return indicator
        
        return None
    
    def _extract_geographic_context(self, query: str) -> Optional[str]:
        """Извлечение географического контекста"""
        geographic_terms = [
            'казахстан', 'алматы', 'астана', 'нур-султан', 'шымкент',
            'караганда', 'актобе', 'тараз', 'павлодар', 'усть-каменогорск',
            'семей', 'атырау', 'костанай', 'кызылорда', 'актау',
            'kazakhstan', 'almaty', 'astana', 'nur-sultan'
        ]
        
        query_lower = query.lower()
        for term in geographic_terms:
            if term in query_lower:
                return term
        
        return 'Казахстан'  # По умолчанию для казахстанского контекста

class IntelligentWebSearch:
    """Интеллектуальная система веб-поиска"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Компоненты системы
        self.authority_scorer = AuthorityScorer()
        self.query_optimizer = QueryOptimizer()
        
        # API ключи
        self.google_api_key = GOOGLE_SEARCH_API_KEY
        self.google_engine_id = GOOGLE_SEARCH_ENGINE_ID
        
        # Кэш результатов
        self.search_cache = {}
        self.cache_ttl = timedelta(hours=6)  # TTL кэша
        
        # Метрики
        self.metrics = SearchMetrics(
            total_queries=0,
            successful_queries=0,
            average_response_time=0.0,
            cache_hit_rate=0.0,
            quality_distribution={},
            source_distribution={}
        )
        
        # История поиска
        self.search_history = []
        
        self.logger.info("Intelligent Web Search инициализирована")
    
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('IntelligentWebSearch')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def intelligent_search(self, query: str, intent_data: Any = None,
                                max_results: int = 5, priority: SearchPriority = SearchPriority.MEDIUM) -> List[Dict[str, Any]]:
        """Интеллектуальный поиск с адаптивными алгоритмами"""
        
        start_time = time.time()
        self.metrics.total_queries += 1
        
        try:
            # Оптимизация запроса
            optimized_query = self.query_optimizer.optimize_query(query, intent_data)
            
            # Проверка кэша
            cache_key = self._generate_cache_key(optimized_query.processed_query, max_results)
            
            if cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self.logger.info(f"Возвращен кэшированный результат для запроса: {query}")
                    self._update_cache_metrics(True)
                    return cached_result['results']
            
            self._update_cache_metrics(False)
            
            # Выполнение поиска
            search_results = await self._execute_search(optimized_query, max_results, priority)
            
            # Извлечение и анализ контента
            enriched_results = await self._enrich_search_results(search_results, optimized_query)
            
            # Ранжирование результатов
            ranked_results = await self._rank_results(enriched_results, optimized_query, intent_data)
            
            # Конвертация в формат RAG системы
            formatted_results = self._format_for_rag(ranked_results[:max_results])
            
            # Кэширование результатов
            self.search_cache[cache_key] = {
                'results': formatted_results,
                'timestamp': datetime.now(),
                'query': optimized_query
            }
            
            # Обновление метрик
            search_time = time.time() - start_time
            self._update_metrics(search_time, len(formatted_results), ranked_results)
            
            # Сохранение в историю
            await self._save_search_record(optimized_query, formatted_results, search_time)
            
            self.metrics.successful_queries += 1
            
            self.logger.info(
                f"Выполнен поиск для запроса '{query}', "
                f"найдено {len(formatted_results)} результатов за {search_time:.2f}с"
            )
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении поиска '{query}': {str(e)}")
            return []
    
    async def _execute_search(self, optimized_query: SearchQuery, 
                            max_results: int, priority: SearchPriority) -> List[Dict[str, Any]]:
        """Выполнение поиска через различные источники"""
        
        search_sources = []
        
        # Google Custom Search (основной источник)
        if self.google_api_key and self.google_engine_id:
            google_results = await self._google_search(optimized_query.processed_query, max_results)
            search_sources.extend(google_results)
        
        # Специализированные источники для казахстанского контекста
        if priority in [SearchPriority.HIGH, SearchPriority.CRITICAL]:
            gov_results = await self._government_sources_search(optimized_query)
            search_sources.extend(gov_results)
        
        # Новостные источники (если есть временной контекст)
        if optimized_query.temporal_context:
            news_results = await self._news_sources_search(optimized_query)
            search_sources.extend(news_results)
        
        return search_sources
    
    async def _google_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Поиск через Google Custom Search API"""
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_engine_id,
                'q': query,
                'num': min(max_results, 10),  # Google ограничивает до 10
                'lr': 'lang_ru|lang_kk|lang_en',  # Языки
                'gl': 'kz',  # Страна
                'safe': 'active'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_google_results(data)
                    else:
                        self.logger.error(f"Google Search API вернул статус {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Ошибка Google Search: {str(e)}")
            return []
    
    def _parse_google_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Парсинг результатов Google Search"""
        results = []
        
        for item in data.get('items', []):
            result = {
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'source': 'google',
                'metadata': {
                    'cacheId': item.get('cacheId'),
                    'formattedUrl': item.get('formattedUrl'),
                    'displayLink': item.get('displayLink')
                }
            }
            
            # Дополнительные метаданные из pagemap
            if 'pagemap' in item:
                pagemap = item['pagemap']
                if 'metatags' in pagemap and pagemap['metatags']:
                    metatags = pagemap['metatags'][0]
                    result['metadata'].update({
                        'description': metatags.get('description', ''),
                        'keywords': metatags.get('keywords', ''),
                        'author': metatags.get('author', ''),
                        'published': metatags.get('article:published_time', '')
                    })
            
            results.append(result)
        
        return results
    
    async def _government_sources_search(self, optimized_query: SearchQuery) -> List[Dict[str, Any]]:
        """Поиск по государственным источникам"""
        gov_sites = [
            'site:gov.kz',
            'site:egov.kz',
            'site:zakon.kz',
            'site:adilet.zan.kz'
        ]
        
        results = []
        
        for site in gov_sites:
            site_query = f"{optimized_query.processed_query} {site}"
            
            try:
                site_results = await self._google_search(site_query, 3)
                for result in site_results:
                    result['source'] = 'government'
                    result['priority_boost'] = 1.5  # Буст для официальных источников
                
                results.extend(site_results)
            except Exception as e:
                self.logger.warning(f"Ошибка поиска по {site}: {str(e)}")
        
        return results
    
    async def _news_sources_search(self, optimized_query: SearchQuery) -> List[Dict[str, Any]]:
        """Поиск по новостным источникам"""
        news_sites = [
            'site:inform.kz',
            'site:tengrinews.kz',
            'site:kazpravda.kz'
        ]
        
        results = []
        
        # Добавление временных фильтров
        time_filter = ""
        if optimized_query.temporal_context:
            if 'сегодня' in optimized_query.temporal_context or 'today' in optimized_query.temporal_context:
                time_filter = " after:1d"
            elif 'вчера' in optimized_query.temporal_context or 'yesterday' in optimized_query.temporal_context:
                time_filter = " after:2d"
        """
Интеллектуальная система веб-поиска (продолжение)
"""

        for site in news_sites:
            site_query = f"{optimized_query.processed_query} {site}{time_filter}"
            
            try:
                site_results = await self._google_search(site_query, 2)
                for result in site_results:
                    result['source'] = 'news'
                    result['freshness_boost'] = 1.2  # Буст для свежих новостей
                
                results.extend(site_results)
            except Exception as e:
                self.logger.warning(f"Ошибка поиска новостей по {site}: {str(e)}")
        
        return results
    
    async def _enrich_search_results(self, search_results: List[Dict[str, Any]], 
                                   optimized_query: SearchQuery) -> List[SearchResult]:
        """Обогащение результатов поиска дополнительным контентом"""
        
        enriched_results = []
        
        async with ContentExtractor() as extractor:
            for result in search_results:
                try:
                    # Извлечение полного контента страницы
                    title, content, metadata = await extractor.extract_content(result['url'])
                    
                    # Если контент не извлечен, используем snippet
                    if not content:
                        content = result.get('snippet', '')
                    
                    # Если заголовок не извлечен, используем title из поиска
                    if not title:
                        title = result.get('title', '')
                    
                    # Вычисление оценок
                    authority_score, content_type = self.authority_scorer.calculate_authority_score(
                        result['url'], content, title
                    )
                    
                    relevance_score = self._calculate_relevance_score(
                        content, title, optimized_query
                    )
                    
                    freshness_score = self._calculate_freshness_score(
                        metadata, result.get('metadata', {})
                    )
                    
                    quality_score = self._calculate_quality_score(
                        content, title, authority_score
                    )
                    
                    # Применение бустов
                    priority_boost = result.get('priority_boost', 1.0)
                    freshness_boost = result.get('freshness_boost', 1.0)
                    
                    total_score = (
                        relevance_score * 0.4 + 
                        authority_score * 0.3 + 
                        quality_score * 0.2 + 
                        freshness_score * 0.1
                    ) * priority_boost * freshness_boost
                    
                    # Создание обогащенного результата
                    enriched_result = SearchResult(
                        title=title,
                        url=result['url'],
                        snippet=result.get('snippet', ''),
                        content=content[:2000],  # Ограничение размера контента
                        content_type=content_type,
                        authority_score=authority_score,
                        relevance_score=relevance_score,
                        freshness_score=freshness_score,
                        quality_score=quality_score,
                        total_score=total_score,
                        metadata={
                            **metadata,
                            **result.get('metadata', {}),
                            'source_type': result.get('source', 'web'),
                            'extraction_success': bool(content),
                            'processing_time': time.time()
                        },
                        extracted_at=datetime.now()
                    )
                    
                    enriched_results.append(enriched_result)
                    
                except Exception as e:
                    self.logger.warning(f"Ошибка обогащения результата {result['url']}: {str(e)}")
                    
                    # Создание базового результата при ошибке
                    basic_result = SearchResult(
                        title=result.get('title', ''),
                        url=result['url'],
                        snippet=result.get('snippet', ''),
                        content=result.get('snippet', ''),
                        content_type=ContentType.GENERAL,
                        authority_score=0.3,
                        relevance_score=0.5,
                        freshness_score=0.5,
                        quality_score=0.3,
                        total_score=0.4,
                        metadata={'error': str(e)},
                        extracted_at=datetime.now()
                    )
                    
                    enriched_results.append(basic_result)
        
        return enriched_results
    
    def _calculate_relevance_score(self, content: str, title: str, 
                                 optimized_query: SearchQuery) -> float:
        """Вычисление оценки релевантности"""
        
        # Объединение контента для анализа
        full_text = f"{title} {content}".lower()
        
        # Базовая релевантность через TF-IDF
        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            
            # Создание корпуса из запроса и контента
            corpus = [optimized_query.processed_query.lower(), full_text]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Вычисление косинусного сходства
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            base_relevance = similarity
            
        except Exception:
            # Fallback к простому подсчету совпадений
            query_words = set(optimized_query.processed_query.lower().split())
            content_words = set(full_text.split())
            
            common_words = query_words & content_words
            base_relevance = len(common_words) / len(query_words) if query_words else 0
        
        # Бонусы за совпадения ключевых терминов
        keyword_bonus = 0
        for keyword in optimized_query.keywords:
            if keyword.lower() in full_text:
                keyword_bonus += 0.1
        
        # Бонусы за совпадения сущностей
        entity_bonus = 0
        for entity in optimized_query.entities:
            if entity.lower() in full_text:
                entity_bonus += 0.15
        
        # Бонусы за доменные термины
        domain_bonus = 0
        for term in optimized_query.domain_specific_terms:
            if term.lower() in full_text:
                domain_bonus += 0.1
        
        # Итоговая релевантность
        total_relevance = base_relevance + keyword_bonus + entity_bonus + domain_bonus
        
        return max(0.0, min(1.0, total_relevance))
    
    def _calculate_freshness_score(self, page_metadata: Dict[str, Any], 
                                 search_metadata: Dict[str, Any]) -> float:
        """Вычисление оценки свежести контента"""
        
        # Поиск даты публикации
        published_date = None
        
        # Приоритет источников дат
        date_sources = [
            page_metadata.get('published_date'),
            search_metadata.get('published'),
            page_metadata.get('date'),
            search_metadata.get('date')
        ]
        
        for date_source in date_sources:
            if date_source:
                published_date = self._parse_date(date_source)
                if published_date:
                    break
        
        if not published_date:
            return 0.5  # Нейтральная оценка при отсутствии даты
        
        # Вычисление возраста контента
        now = datetime.now()
        age_days = (now - published_date).days
        
        # Оценка свежести с экспоненциальным затуханием
        if age_days <= 1:
            return 1.0  # Очень свежий контент
        elif age_days <= 7:
            return 0.9  # Недельной давности
        elif age_days <= 30:
            return 0.7  # Месячной давности
        elif age_days <= 365:
            return 0.5  # Годичной давности
        else:
            return max(0.1, 0.5 * np.exp(-age_days / 365))  # Экспоненциальное затухание
    
    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Парсинг даты из строки"""
        
        date_formats = [
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%d.%m.%Y',
            '%d/%m/%Y',
            '%B %d, %Y',
            '%d %B %Y'
        ]
        
        # Очистка строки
        date_string = date_string.strip()
        
        for date_format in date_formats:
            try:
                return datetime.strptime(date_string, date_format)
            except ValueError:
                continue
        
        # Попытка извлечения даты регулярными выражениями
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{2}\.\d{2}\.\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_string)
            if match:
                try:
                    if '-' in match.group(1):
                        return datetime.strptime(match.group(1), '%Y-%m-%d')
                    elif '.' in match.group(1):
                        return datetime.strptime(match.group(1), '%d.%m.%Y')
                    elif '/' in match.group(1):
                        return datetime.strptime(match.group(1), '%d/%m/%Y')
                except ValueError:
                    continue
        
        return None
    
    def _calculate_quality_score(self, content: str, title: str, 
                               authority_score: float) -> float:
        """Вычисление оценки качества контента"""
        
        quality_factors = []
        
        # 1. Длина контента
        content_length = len(content)
        if 200 <= content_length <= 5000:
            length_score = 1.0
        elif content_length < 200:
            length_score = content_length / 200
        else:
            length_score = max(0.5, 1.0 - (content_length - 5000) / 10000)
        
        quality_factors.append(length_score)
        
        # 2. Структурированность
        structure_indicators = [
            content.count('\n') > 3,  # Абзацы
            content.count('.') > 5,   # Предложения
            content.count(':') > 0,   # Списки/структура
            len(title) > 0            # Наличие заголовка
        ]
        
        structure_score = sum(structure_indicators) / len(structure_indicators)
        quality_factors.append(structure_score)
        
        # 3. Информационная плотность
        words = content.split()
        if words:
            unique_words = set(word.lower() for word in words)
            density_score = len(unique_words) / len(words)
        else:
            density_score = 0
        
        quality_factors.append(density_score)
        
        # 4. Отсутствие спама и некачественного контента
        spam_indicators = [
            'реклама', 'купить', 'скидка', 'акция', 'продажа',
            'casino', 'bet', 'loan', 'credit', 'viagra'
        ]
        
        content_lower = content.lower()
        spam_count = sum(1 for indicator in spam_indicators if indicator in content_lower)
        spam_penalty = min(1.0, spam_count / 10)
        anti_spam_score = 1.0 - spam_penalty
        
        quality_factors.append(anti_spam_score)
        
        # 5. Авторитетность источника
        quality_factors.append(authority_score)
        
        # Итоговая оценка качества
        base_quality = sum(quality_factors) / len(quality_factors)
        
        return max(0.0, min(1.0, base_quality))
    
    async def _rank_results(self, enriched_results: List[SearchResult], 
                          optimized_query: SearchQuery, intent_data: Any = None) -> List[SearchResult]:
        """Ранжирование результатов поиска"""
        
        # Базовое ранжирование по общему скору
        ranked_results = sorted(enriched_results, key=lambda x: x.total_score, reverse=True)
        
        # Дополнительные правила ранжирования
        
        # 1. Приоритет официальным источникам для государственных запросов
        if intent_data and hasattr(intent_data, 'domain') and intent_data.domain == 'government':
            official_results = [r for r in ranked_results if r.content_type == ContentType.OFFICIAL]
            other_results = [r for r in ranked_results if r.content_type != ContentType.OFFICIAL]
            ranked_results = official_results + other_results
        
        # 2. Приоритет свежему контенту для временных запросов
        if optimized_query.temporal_context:
            fresh_results = sorted(ranked_results, key=lambda x: x.freshness_score, reverse=True)
            ranked_results = fresh_results
        
        # 3. Диверсификация источников
        ranked_results = self._diversify_sources(ranked_results)
        
        # 4. Фильтрация низкокачественных результатов
        ranked_results = [r for r in ranked_results if r.total_score > 0.2]
        
        return ranked_results
    
    def _diversify_sources(self, results: List[SearchResult]) -> List[SearchResult]:
        """Диверсификация источников в результатах"""
        
        diversified = []
        used_domains = set()
        
        # Первый проход - уникальные домены
        for result in results:
            domain = self.authority_scorer._extract_domain(result.url)
            if domain not in used_domains:
                diversified.append(result)
                used_domains.add(domain)
        
        # Второй проход - добавление оставшихся результатов
        for result in results:
            if result not in diversified:
                diversified.append(result)
        
        return diversified
    
    def _format_for_rag(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Форматирование результатов для RAG системы"""
        
        formatted_results = []
        
        for result in search_results:
            formatted_result = {
                'content': result.content,
                'metadata': {
                    'source': result.url,
                    'title': result.title,
                    'type': 'web_search',
                    'content_type': result.content_type.value,
                    'authority_score': round(result.authority_score, 3),
                    'relevance_score': round(result.relevance_score, 3),
                    'quality_score': round(result.quality_score, 3),
                    'total_score': round(result.total_score, 3),
                    'extracted_at': result.extracted_at.isoformat(),
                    'snippet': result.snippet
                },
                'score': result.total_score  # Для совместимости с RAG системой
            }
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _generate_cache_key(self, query: str, max_results: int) -> str:
        """Генерация ключа для кэширования"""
        cache_input = f"{query}|{max_results}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_entry: Dict[str, Any]) -> bool:
        """Проверка актуальности кэшированного результата"""
        timestamp = cached_entry.get('timestamp')
        if not timestamp:
            return False
        
        age = datetime.now() - timestamp
        return age < self.cache_ttl
    
    def _update_cache_metrics(self, cache_hit: bool):
        """Обновление метрик кэша"""
        if not hasattr(self, '_cache_requests'):
            self._cache_requests = 0
            self._cache_hits = 0
        
        self._cache_requests += 1
        if cache_hit:
            self._cache_hits += 1
        
        self.metrics.cache_hit_rate = self._cache_hits / self._cache_requests
    
    def _update_metrics(self, search_time: float, results_count: int, 
                       enriched_results: List[SearchResult]):
        """Обновление общих метрик"""
        
        # Обновление времени ответа
        total_time = self.metrics.average_response_time * (self.metrics.total_queries - 1)
        self.metrics.average_response_time = (total_time + search_time) / self.metrics.total_queries
        
        # Обновление распределения качества
        for result in enriched_results:
            quality_tier = self._get_quality_tier(result.total_score)
            self.metrics.quality_distribution[quality_tier] = \
                self.metrics.quality_distribution.get(quality_tier, 0) + 1
        
        # Обновление распределения источников
        for result in enriched_results:
            source_type = result.metadata.get('source_type', 'unknown')
            self.metrics.source_distribution[source_type] = \
                self.metrics.source_distribution.get(source_type, 0) + 1
    
    def _get_quality_tier(self, score: float) -> str:
        """Определение уровня качества по скору"""
        if score >= 0.8:
            return 'high'
        elif score >= 0.6:
            return 'medium'
        elif score >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    async def _save_search_record(self, optimized_query: SearchQuery, 
                                results: List[Dict[str, Any]], search_time: float):
        """Сохранение записи о поиске"""
        
        record = {
            'timestamp': datetime.now(),
            'original_query': optimized_query.original_query,
            'processed_query': optimized_query.processed_query,
            'keywords': optimized_query.keywords,
            'entities': optimized_query.entities,
            'results_count': len(results),
            'search_time': search_time,
            'avg_quality': sum(r['metadata']['total_score'] for r in results) / len(results) if results else 0,
            'sources_used': list(set(r['metadata']['source'] for r in results))
        }
        
        self.search_history.append(record)
        
        # Ограничение размера истории
        if len(self.search_history) > 1000:
            self.search_history = self.search_history[-800:]
    
    async def search_news_feeds(self, query: str, max_articles: int = 5) -> List[Dict[str, Any]]:
        """Поиск в RSS лентах новостных сайтов"""
        
        news_feeds = [
            'https://inform.kz/rss/',
            'https://tengrinews.kz/rss/',
            'https://kazpravda.kz/rss'
        ]
        
        articles = []
        
        for feed_url in news_feeds:
            try:
                # Парсинг RSS ленты
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:max_articles]:
                    # Проверка релевантности
                    if self._is_article_relevant(entry, query):
                        article = {
                            'content': entry.get('summary', entry.get('description', '')),
                            'metadata': {
                                'source': entry.get('link', ''),
                                'title': entry.get('title', ''),
                                'type': 'news_feed',
                                'published': entry.get('published', ''),
                                'author': entry.get('author', ''),
                                'feed_source': feed_url
                            },
                            'score': 0.8  # Высокий скор для новостей
                        }
                        
                        articles.append(article)
                        
            except Exception as e:
                self.logger.warning(f"Ошибка парсинга RSS ленты {feed_url}: {str(e)}")
        
        return articles[:max_articles]
    
    def _is_article_relevant(self, entry: Any, query: str) -> bool:
        """Проверка релевантности статьи запросу"""
        
        article_text = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
        query_words = set(query.lower().split())
        
        # Простая проверка на пересечение ключевых слов
        article_words = set(article_text.split())
        common_words = query_words & article_words
        
        return len(common_words) >= 1  # Минимум одно общее слово
    
    async def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Получение предложений для автодополнения поиска"""
        
        suggestions = []
        
        # Предложения на основе истории поиска
        for record in self.search_history:
            if partial_query.lower() in record['original_query'].lower():
                suggestions.append(record['original_query'])
        
        # Популярные казахстанские запросы
        popular_queries = [
            'как получить справку о несудимости',
            'регистрация ИП в Казахстане',
            'получение паспорта РК',
            'подача документов через ЦОН',
            'услуги egov.kz',
            'налоговые льготы для бизнеса',
            'социальные выплаты в Казахстане'
        ]
        
        for query in popular_queries:
            if partial_query.lower() in query.lower():
                suggestions.append(query)
        
        # Удаление дубликатов и ограничение количества
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:10]
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Получение аналитики поиска"""
        
        if not self.search_history:
            return {'message': 'Нет данных для аналитики'}
        
        # Базовая статистика
        total_searches = len(self.search_history)
        
        # Средние метрики
        avg_search_time = sum(r['search_time'] for r in self.search_history) / total_searches
        avg_results_count = sum(r['results_count'] for r in self.search_history) / total_searches
        avg_quality = sum(r['avg_quality'] for r in self.search_history) / total_searches
        
        # Популярные ключевые слова
        all_keywords = []
        for record in self.search_history:
            all_keywords.extend(record.get('keywords', []))
        
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        popular_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Временная статистика
        recent_searches = [r for r in self.search_history 
                          if r['timestamp'] > datetime.now() - timedelta(days=7)]
        
        # Источники результатов
        all_sources = []
        for record in self.search_history:
            all_sources.extend(record.get('sources_used', []))
        
        source_domains = {}
        for source in all_sources:
            domain = self.authority_scorer._extract_domain(source)
            source_domains[domain] = source_domains.get(domain, 0) + 1
        
        top_sources = sorted(source_domains.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_searches': total_searches,
            'recent_searches_7d': len(recent_searches),
            'average_metrics': {
                'search_time': round(avg_search_time, 3),
                'results_count': round(avg_results_count, 1),
                'quality_score': round(avg_quality, 3)
            },
            'popular_keywords': [{'keyword': k, 'count': c} for k, c in popular_keywords],
            'top_sources': [{'domain': d, 'count': c} for d, c in top_sources],
            'cache_performance': {
                'hit_rate': round(self.metrics.cache_hit_rate, 3),
                'cache_size': len(self.search_cache)
            },
            'quality_distribution': self.metrics.quality_distribution,
            'source_distribution': self.metrics.source_distribution
        }
    
    async def optimize_search_performance(self) -> Dict[str, Any]:
        """Оптимизация производительности поиска"""
        
        optimizations = []
        
        # Очистка устаревшего кэша
        initial_cache_size = len(self.search_cache)
        current_time = datetime.now()
        fresh_cache = {}
        
        for key, value in self.search_cache.items():
            if self._is_cache_valid(value):
                fresh_cache[key] = value
        
        self.search_cache = fresh_cache
        cleared_cache = initial_cache_size - len(fresh_cache)
        
        if cleared_cache > 0:
            optimizations.append(f"Очищено {cleared_cache} устаревших записей кэша")
        
        # Анализ производительности запросов
        if self.search_history:
            slow_queries = [r for r in self.search_history if r['search_time'] > 5.0]
            if slow_queries:
                optimizations.append(f"Обнаружено {len(slow_queries)} медленных запросов")
        
        # Оптимизация TTL кэша на основе паттернов использования
        if self.metrics.cache_hit_rate < 0.3:
            self.cache_ttl = timedelta(hours=12)  # Увеличение TTL
            optimizations.append("Увеличен TTL кэша для улучшения hit rate")
        elif self.metrics.cache_hit_rate > 0.8:
            self.cache_ttl = timedelta(hours=3)  # Уменьшение TTL для свежести
            optimizations.append("Уменьшен TTL кэша для обеспечения свежести")
        
        # Обновление конфигурации поиска
        if len(self.search_history) > 100:
            # Анализ наиболее эффективных источников
            source_performance = {}
            for record in self.search_history[-100:]:  # Последние 100 записей
                for source in record.get('sources_used', []):
                    domain = self.authority_scorer._extract_domain(source)
                    if domain not in source_performance:
                        source_performance[domain] = {'total': 0, 'avg_quality': 0}
                    
                    source_performance[domain]['total'] += 1
                    source_performance[domain]['avg_quality'] += record['avg_quality']
            
            # Вычисление средних значений
            for domain in source_performance:
                perf = source_performance[domain]
                perf['avg_quality'] /= perf['total']
            
            optimizations.append("Обновлена статистика производительности источников")
        
        return {
            'optimizations_applied': len(optimizations),
            'details': optimizations,
            'current_cache_size': len(self.search_cache),
            'cache_ttl_hours': self.cache_ttl.total_seconds() / 3600,
            'performance_metrics': {
                'avg_response_time': self.metrics.average_response_time,
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'success_rate': self.metrics.successful_queries / max(1, self.metrics.total_queries)
            }
        }
    
    async def clear_search_cache(self, older_than_hours: int = 24) -> Dict[str, Any]:
        """Очистка кэша поиска"""
        
        initial_size = len(self.search_cache)
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        fresh_cache = {}
        for key, value in self.search_cache.items():
            timestamp = value.get('timestamp')
            if timestamp and timestamp > cutoff_time:
                fresh_cache[key] = value
        
        self.search_cache = fresh_cache
        cleared_count = initial_size - len(fresh_cache)
        
        self.logger.info(f"Очищено {cleared_count} записей кэша старше {older_than_hours} часов")
        
        return {
            'initial_cache_size': initial_size,
            'cleared_entries': cleared_count,
            'current_cache_size': len(fresh_cache),
            'cutoff_hours': older_than_hours
        }

_web_search_instance = None

def get_web_search() -> IntelligentWebSearch:
    global _web_search_instance
    if _web_search_instance is None:
        _web_search_instance = IntelligentWebSearch()
    return _web_search_instance