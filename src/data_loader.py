import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import docx
from bs4 import BeautifulSoup
from config import DOCUMENTS_DIR

class SimpleDataLoader:
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.docx', '.html']
    
    def load_documents_from_folder(self) -> List[Dict[str, Any]]:
        documents = []
        
        for root, dirs, files in os.walk(DOCUMENTS_DIR):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.supported_formats:
                    try:
                        content = self._extract_content(file_path)
                        if content.strip():
                            documents.append({
                                'content': content,
                                'metadata': {
                                    'source': str(file_path),
                                    'filename': file,
                                    'type': file_path.suffix[1:],
                                    'size': file_path.stat().st_size
                                }
                            })
                    except Exception as e:
                        print(f"Ошибка загрузки {file_path}: {e}")
        
        return documents
    
    def _extract_content(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return file_path.read_text(encoding='utf-8', errors='ignore')
        
        elif suffix == '.pdf':
            return self._extract_pdf(file_path)
        
        elif suffix == '.docx':
            return self._extract_docx(file_path)
        
        elif suffix == '.html':
            return self._extract_html(file_path)
        
        return ""
    
    def _extract_pdf(self, file_path: Path) -> str:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    
    def _extract_docx(self, file_path: Path) -> str:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_html(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            return soup.get_text()
    
    def load_from_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        documents = []
        
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                
                if text.strip():
                    documents.append({
                        'content': text,
                        'metadata': {
                            'source': url,
                            'type': 'web',
                            'title': soup.title.string if soup.title else url
                        }
                    })
            except Exception as e:
                print(f"Ошибка загрузки {url}: {e}")
        
        return documents
    
    def load_sample_kazakh_gov_data(self) -> List[Dict[str, Any]]:
        sample_data = [
            {
                'content': """
                Электронное правительство Республики Казахстан - это система предоставления 
                государственных услуг в электронном виде через единый портал egov.kz. 
                Портал обеспечивает доступ граждан и бизнеса к более чем 700 государственным услугам.
                """,
                'metadata': {
                    'source': 'egov.kz',
                    'type': 'government_service',
                    'category': 'электронные услуги'
                }
            },
            {
                'content': """
                Министерство цифрового развития, инноваций и аэрокосмической промышленности 
                Республики Казахстан осуществляет реализацию государственной политики в сфере 
                информационно-коммуникационных технологий, цифровизации, инноваций.
                """,
                'metadata': {
                    'source': 'mdai.gov.kz',
                    'type': 'ministry_info',
                    'category': 'цифровое развитие'
                }
            },
            {
                'content': """
                Национальная платформа искусственного интеллекта (НПИИ) создается для 
                предоставления доступа к современным AI-инструментам государственным 
                и частным организациям Казахстана. Платформа включает в себя модули 
                для обработки естественного языка, компьютерного зрения и аналитики данных.
                """,
                'metadata': {
                    'source': 'nis.gov.kz',
                    'type': 'ai_platform',
                    'category': 'искусственный интеллект'
                }
            }
        ]
        
        return sample_data

QUICK_DATA_SOURCES = {
    "local_documents": {
        "path": DOCUMENTS_DIR,
        "description": "Локальные документы в папке data/documents/",
        "loader": "load_documents_from_folder"
    },
    
    "kazakh_gov_websites": {
        "urls": [
            "https://egov.kz",
            "https://gov.kz", 
            "https://mdai.gov.kz",
            "https://nis.gov.kz"
        ],
        "description": "Сайты казахстанских государственных органов",
        "loader": "load_from_urls"
    },
    
    "sample_data": {
        "description": "Примерные данные для быстрого тестирования",
        "loader": "load_sample_kazakh_gov_data"
    }
}