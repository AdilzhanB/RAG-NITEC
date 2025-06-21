import asyncio
import uvicorn
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Импорт продвинутых компонентов
from src.rag_pipeline import get_rag_pipeline, AdvancedRAGPipeline
from src.feedback_system import get_feedback_system, AdvancedFeedbackSystem
from src.gemini_client import get_gemini_client, AdvancedGeminiClient
from src.web_search import get_web_search, IntelligentWebSearch

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Глобальные экземпляры компонентов
rag_pipeline: Optional[AdvancedRAGPipeline] = None
feedback_system: Optional[AdvancedFeedbackSystem] = None
gemini_client: Optional[AdvancedGeminiClient] = None
web_search: Optional[IntelligentWebSearch] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Инициализация при запуске
    global rag_pipeline, feedback_system, gemini_client, web_search
    
    print("🚀 Инициализация продвинутой RAG системы...")
    
    try:
        # Инициализация компонентов
        rag_pipeline = get_rag_pipeline()
        feedback_system = get_feedback_system()
        gemini_client = get_gemini_client()
        web_search = get_web_search()
        
        # Загрузка документов
        print("📚 Загрузка и индексация документов...")
        rag_pipeline.load_and_index_documents()
        
        print("✅ Система готова к работе!")
        
        yield
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {str(e)}")
        raise
    finally:
        # Очистка при завершении
        print("🔄 Завершение работы системы...")
        
        if web_search:
            await web_search.clear_search_cache(older_than_hours=1)
        
        if gemini_client:
            await gemini_client.clear_cache(max_age_hours=1)

# Создание FastAPI приложения
app = FastAPI(
    title="Advanced RAG Chatbot with RLHF",
    description="Продвинутая RAG система с обратной связью и адаптивным обучением",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic модели для API
class QueryRequest(BaseModel):
    query: str = Field(..., description="Пользовательский запрос")
    user_id: str = Field(default="default", description="ID пользователя")
    session_id: Optional[str] = Field(None, description="ID сессии")
    priority: str = Field(default="medium", description="Приоритет запроса")

class FeedbackRequest(BaseModel):
    interaction_id: str = Field(..., description="ID взаимодействия")
    feedback_type: str = Field(..., description="Тип обратной связи")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Оценка (1-5)")
    correction: Optional[str] = Field(None, description="Исправление")
    suggestion: Optional[str] = Field(None, description="Предложение")
    explanation: Optional[str] = Field(None, description="Объяснение")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Поисковый запрос")
    max_results: int = Field(default=5, ge=1, le=20, description="Максимум результатов")
    priority: str = Field(default="medium", description="Приоритет поиска")

# Зависимости
async def get_components():
    """Получение инициализированных компонентов"""
    if not all([rag_pipeline, feedback_system, gemini_client, web_search]):
        raise HTTPException(status_code=503, detail="Система не инициализирована")
    
    return {
        'rag': rag_pipeline,
        'feedback': feedback_system,
        'gemini': gemini_client,
        'search': web_search
    }

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def main_page():
    """Главная страница"""
    return FileResponse("static/index.html")

@app.post("/api/v2/query")
async def process_advanced_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    components: Dict = Depends(get_components)
):
    """Продвинутая обработка пользовательского запроса"""
    try:
        start_time = datetime.now()
        
        # Обработка запроса через продвинутый RAG pipeline
        result = await components['rag'].process_query(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        # Добавление дополнительных метрик
        processing_time = (datetime.now() - start_time).total_seconds()
        result['processing_time'] = processing_time
        result['timestamp'] = datetime.now().isoformat()
        result['api_version'] = "2.0"
        
        # Фоновая задача для обновления аналитики
        background_tasks.add_task(
            update_analytics,
            request.query,
            request.user_id,
            processing_time,
            result.get('confidence', 0)
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")

@app.post("/api/v2/feedback")
async def submit_advanced_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    components: Dict = Depends(get_components)
):
    """Отправка продвинутой обратной связи"""
    try:
        # Получение взаимодействия
        interaction = components['rag'].interactions_cache.get(request.interaction_id)
        if not interaction:
            raise HTTPException(status_code=404, detail="Взаимодействие не найдено")
        
        # Обработка обратной связи
        feedback_data = {
            'feedback_type': request.feedback_type,
            'rating': request.rating,
            'correction': request.correction,
            'suggestion': request.suggestion,
            'explanation': request.explanation,
            'timestamp': datetime.now().isoformat()
        }
        
        result = await components['feedback'].process_advanced_feedback(
            interaction, feedback_data
        )
        
        # Обучение Gemini клиента на основе обратной связи
        background_tasks.add_task(
            components['gemini'].learn_from_feedback,
            request.interaction_id,
            feedback_data
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки обратной связи: {str(e)}")

@app.post("/api/v2/search")
async def intelligent_web_search(
    request: SearchRequest,
    components: Dict = Depends(get_components)
):
    """Интеллектуальный веб-поиск"""
    try:
        from src.web_search import SearchPriority
        
        priority_mapping = {
            'low': SearchPriority.LOW,
            'medium': SearchPriority.MEDIUM,
            'high': SearchPriority.HIGH,
            'critical': SearchPriority.CRITICAL
        }
        
        priority = priority_mapping.get(request.priority, SearchPriority.MEDIUM)
        
        results = await components['search'].intelligent_search(
            query=request.query,
            max_results=request.max_results,
            priority=priority
        )
        
        return {
            'query': request.query,
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка веб-поиска: {str(e)}")

@app.get("/api/v2/analytics")
async def get_system_analytics(components: Dict = Depends(get_components)):
    """Получение расширенной аналитики системы"""
    try:
        # Сбор аналитики от всех компонентов
        rag_stats = components['rag'].get_advanced_stats()
        feedback_stats = await components['feedback'].get_advanced_stats()
        gemini_stats = components['gemini'].get_performance_stats()
        search_stats = components['search'].get_search_analytics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_version': '2.0.0',
            'rag_pipeline': rag_stats,
            'feedback_system': feedback_stats,
            'gemini_client': gemini_stats,
            'web_search': search_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения аналитики: {str(e)}")

@app.get("/api/v2/health")
async def health_check(components: Dict = Depends(get_components)):
    """Проверка состояния системы"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'rag_pipeline': 'active',
                'feedback_system': 'active',
                'gemini_client': 'active',
                'web_search': 'active'
            },
            'uptime': 'calculated_uptime',
            'version': '2.0.0'
        }
        
        return health_status
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        )

@app.post("/api/v2/optimize")
async def optimize_system_performance(
    background_tasks: BackgroundTasks,
    components: Dict = Depends(get_components)
):
    """Оптимизация производительности системы"""
    try:
        optimization_results = {}
        
        # Оптимизация Gemini клиента
        gemini_optimization = await components['gemini'].optimize_generation_settings()
        optimization_results['gemini'] = gemini_optimization
        
        # Оптимизация веб-поиска
        search_optimization = await components['search'].optimize_search_performance()
        optimization_results['web_search'] = search_optimization
        
        # Фоновые задачи оптимизации
        background_tasks.add_task(cleanup_old_data, components)
        
        return {
            'optimization_completed': True,
            'timestamp': datetime.now().isoformat(),
            'results': optimization_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка оптимизации: {str(e)}")

@app.get("/api/v2/suggestions/{partial_query}")
async def get_query_suggestions(
    partial_query: str,
    components: Dict = Depends(get_components)
):
    """Получение предложений для автодополнения"""
    try:
        suggestions = await components['search'].get_search_suggestions(partial_query)
        
        return {
            'partial_query': partial_query,
            'suggestions': suggestions,
            'count': len(suggestions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения предложений: {str(e)}")

@app.get("/api/v2/feedback/trends")
async def get_feedback_trends(
    days: int = 30,
    components: Dict = Depends(get_components)
):
    """Получение трендов обратной связи"""
    try:
        trends = await components['feedback'].analyze_feedback_trends(days)
        return trends
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа трендов: {str(e)}")

@app.get("/api/v2/user/{user_id}/history")
async def get_user_history(
    user_id: str,
    limit: int = 50,
    components: Dict = Depends(get_components)
):
    """Получение истории пользователя"""
    try:
        history = await components['feedback'].get_user_feedback_history(user_id, limit)
        
        return {
            'user_id': user_id,
            'history': history,
            'count': len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения истории: {str(e)}")

# Фоновые задачи
async def update_analytics(query: str, user_id: str, processing_time: float, confidence: float):
    """Обновление аналитики в фоновом режиме"""
    try:
        # Здесь можно добавить логику для сохранения метрик в базу данных
        # или отправки в систему мониторинга
        pass
    except Exception as e:
        logging.error(f"Ошибка обновления аналитики: {str(e)}")

async def cleanup_old_data(components: Dict):
    """Очистка устаревших данных"""
    try:
        # Очистка кэша веб-поиска
        await components['search'].clear_search_cache(older_than_hours=24)
        
        # Очистка кэша Gemini
        await components['gemini'].clear_cache(max_age_hours=24)
        
        logging.info("Очистка устаревших данных завершена")
        
    except Exception as e:
        logging.error(f"Ошибка очистки данных: {str(e)}")

# Обработчики ошибок
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Ресурс не найден",
            "message": "Запрошенный ресурс не существует",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Внутренняя ошибка сервера",
            "message": "Произошла неожиданная ошибка",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    print(" Система включает:")
    print("   - Адаптивный RAG pipeline с RLHF")
    print("   - Интеллектуальную систему обратной связи") 
    print("   - Продвинутый Gemini клиент")
    print("   - Умный веб-поиск")
    print("   - Аналитику и мониторинг")
    print("")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )