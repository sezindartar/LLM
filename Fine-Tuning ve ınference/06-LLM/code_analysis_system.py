"""
Akıllı Kod Analizi ve Otomatik Dokümantasyon Sistemi
Bu sistem, farklı programlama dillerindeki kodları analiz eder,
kalite skorları hesaplar ve otomatik dokümantasyon üretir.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    pipeline
)
import ast
import re
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
from radon.raw import analyze
import subprocess

class IntelligentCodeAnalyzer:
    """Akıllı kod analizi ve dokümantasyon sistemi"""
    
    def __init__(self, workspace_dir: str = "./code_workspace"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(exist_ok=True)
        
        # Model bileşenleri
        self.code_model = None
        self.code_tokenizer = None
        self.doc_generator = None
        
        # Analiz verileri
        self.analysis_database = []
        self.project_reports = []
        
        # Desteklenen diller
        self.supported_languages = {
            'python': {
                'extensions': ['.py'],
                'keywords': ['def', 'class', 'import', 'from', 'if', 'else', 'for', 'while'],
                'complexity_analyzer': self._analyze_python_complexity
            },
            'javascript': {
                'extensions': ['.js', '.jsx'],
                'keywords': ['function', 'const', 'let', 'var', 'if', 'else', 'for', 'while'],
                'complexity_analyzer': self._analyze_js_complexity
            },
            'java': {
                'extensions': ['.java'],
                'keywords': ['public', 'private', 'class', 'interface', 'if', 'else', 'for'],
                'complexity_analyzer': self._analyze_java_complexity
            },
            'cpp': {
                'extensions': ['.cpp', '.hpp', '.h'],
                'keywords': ['#include', 'class', 'namespace', 'if', 'else', 'for', 'while'],
                'complexity_analyzer': self._analyze_cpp_complexity
            }
        }
    
    def initialize_models(self) -> bool:
        """Kod analizi için gerekli modelleri yükler"""
        print("🤖 Kod analizi modelleri yükleniyor...")
        
        try:
            # Ana kod modeli - CodeT5 benzeri
            print("💻 Kod analiz modeli yükleniyor...")
            self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.code_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-medium",  # Kod dokümantasyonu için
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Dokümantasyon üretici
            self.doc_generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                tokenizer=self.code_tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                max_length=512
            )
            
            print("✅ Tüm modeller başarıyla yüklendi!")
            return True
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return False
    
    def detect_language(self, code: str, filename: str = "") -> str:
        """Kod dilini otomatik tespit eder"""
        
        # Dosya uzantısına göre
        for lang, info in self.supported_languages.items():
            for ext in info['extensions']:
                if filename.endswith(ext):
                    return lang
        
        # İçerik analizi ile
        language_scores = {}
        
        for lang, info in self.supported_languages.items():
            score = 0
            keywords = info['keywords']
            
            for keyword in keywords:
                # Keyword'ün kod içinde geçme sayısı
                matches = len(re.findall(rf'\b{keyword}\b', code))
                score += matches
            
            language_scores[lang] = score
        
        # En yüksek skora sahip dil
        if language_scores:
            detected_lang = max(language_scores, key=language_scores.get)
            if language_scores[detected_lang] > 0:
                return detected_lang
        
        return 'unknown'
    
    def analyze_code_structure(self, code: str, language: str) -> Dict[str, Any]:
        """Kod yapısını detaylı analiz eder"""
        
        print(f"🔍 Kod yapısı analiz ediliyor ({language})...")
        
        # Temel metrikler
        lines = code.split('\n')
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        blank_lines = total_lines - code_lines - comment_lines
        
        # Fonksiyon ve sınıf analizi
        functions = self._extract_functions(code, language)
        classes = self._extract_classes(code, language)
        imports = self._extract_imports(code, language)
        
        # Kompleksite analizi
        complexity_info = self._analyze_complexity(code, language)
        
        # Kod kalitesi metrikleri
        quality_metrics = self._calculate_quality_metrics(code, language)
        
        # Stil analizi
        style_analysis = self._analyze_code_style(code, language)
        
        return {
            "language": language,
            "metrics": {
                "total_lines": total_lines,
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "blank_lines": blank_lines,
                "comment_ratio": comment_lines / total_lines if total_lines > 0 else 0
            },
            "structure": {
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "function_count": len(functions),
                "class_count": len(classes),
                "import_count": len(imports)
            },
            "complexity": complexity_info,
            "quality": quality_metrics,
            "style": style_analysis
        }
    
    def _extract_functions(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Fonksiyonları çıkarır ve analiz eder"""
        
        functions = []
        
        if language == 'python':
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_info = {
                            "name": node.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                            "args_count": len(node.args.args),
                            "has_docstring": ast.get_docstring(node) is not None,
                            "docstring": ast.get_docstring(node) or "",
                            "decorators": [d.id for d in node.decorator_list if hasattr(d, 'id')]
                        }
                        functions.append(func_info)
            except:
                pass
        
        elif language == 'javascript':
            # Basit regex ile JS fonksiyon analizi
            js_func_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*{'
            matches = re.finditer(js_func_pattern, code)
            
            for match in matches:
                func_name = match.group(1)
                params = match.group(2)
                param_count = len([p.strip() for p in params.split(',') if p.strip()]) if params.strip() else 0
                
                functions.append({
                    "name": func_name,
                    "args_count": param_count,
                    "has_docstring": False,
                    "docstring": "",
                    "type": "function"
                })
        
        return functions
    
    def _extract_classes(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Sınıfları çıkarır ve analiz eder"""
        
        classes = []
        
        if language == 'python':
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        
                        class_info = {
                            "name": node.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                            "methods": methods,
                            "method_count": len(methods),
                            "has_docstring": ast.get_docstring(node) is not None,
                            "docstring": ast.get_docstring(node) or "",
                            "inheritance": [base.id for base in node.bases if hasattr(base, 'id')]
                        }
                        classes.append(class_info)
            except:
                pass
        
        elif language == 'java':
            # Java sınıf analizi
            java_class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{'
            matches = re.finditer(java_class_pattern, code)
            
            for match in matches:
                class_name = match.group(1)
                parent_class = match.group(2) if match.group(2) else None
                
                classes.append({
                    "name": class_name,
                    "inheritance": [parent_class] if parent_class else [],
                    "has_docstring": False,
                    "type": "class"
                })
        
        return classes
    
    def _extract_imports(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Import/include ifadelerini çıkarır"""
        
        imports = []
        
        if language == 'python':
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append({
                                "module": alias.name,
                                "alias": alias.asname,
                                "type": "import"
                            })
                    elif isinstance(node, ast.ImportFrom):
                        imports.append({
                            "module": node.module,
                            "names": [alias.name for alias in node.names],
                            "type": "from_import"
                        })
            except:
                pass
        
        elif language == 'cpp':
            # C++ include analizi
            include_pattern = r'#include\s*[<"]([^>"]+)[>"]'
            matches = re.finditer(include_pattern, code)
            
            for match in matches:
                imports.append({
                    "module": match.group(1),
                    "type": "include"
                })
        
        return imports
    
    def _analyze_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """Kod karmaşıklığını analiz eder"""
        
        complexity_info = {
            "cyclomatic_complexity": 0,
            "cognitive_complexity": 0,
            "nesting_depth": 0,
            "complexity_score": 0
        }
        
        if language == 'python':
            try:
                # Radon ile Python kompleksite analizi
                cc_results = radon_cc.cc_visit(code)
                if cc_results:
                    complexities = [result.complexity for result in cc_results]
                    complexity_info["cyclomatic_complexity"] = max(complexities) if complexities else 0
                    complexity_info["average_complexity"] = sum(complexities) / len(complexities) if complexities else 0
                
                # Basit nesting depth hesaplama
                lines = code.split('\n')
                max_depth = 0
                current_depth = 0
                
                for line in lines:
                    stripped = line.lstrip()
                    if stripped and not stripped.startswith('#'):
                        indent_level = (len(line) - len(stripped)) // 4
                        max_depth = max(max_depth, indent_level)
                
                complexity_info["nesting_depth"] = max_depth
                
            except Exception as e:
                print(f"Kompleksite analizi hatası: {e}")
        
        # Genel kompleksite skoru hesaplama
        cc = complexity_info["cyclomatic_complexity"]
        nd = complexity_info["nesting_depth"]
        complexity_info["complexity_score"] = min((cc * 10 + nd * 5), 100)
        
        return complexity_info
    
    def _analyze_python_complexity(self, code: str) -> Dict[str, Any]:
        """Python özel kompleksite analizi"""
        return {"language_specific": "python", "analyzed": True}
    
    def _analyze_js_complexity(self, code: str) -> Dict[str, Any]:
        """JavaScript özel kompleksite analizi"""
        return {"language_specific": "javascript", "analyzed": True}
    
    def _analyze_java_complexity(self, code: str) -> Dict[str, Any]:
        """Java özel kompleksite analizi"""
        return {"language_specific": "java", "analyzed": True}
    
    def _analyze_cpp_complexity(self, code: str) -> Dict[str, Any]:
        """C++ özel kompleksite analizi"""
        return {"language_specific": "cpp", "analyzed": True}
    
    def _calculate_quality_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Kod kalitesi metriklerini hesaplar"""
        
        lines = code.split('\n')
        total_lines = len(lines)
        
        # Kod yoğunluğu
        code_density = len([l for l in lines if l.strip() and not l.strip().startswith('#')]) / total_lines if total_lines > 0 else 0
        
        # Ortalama satır uzunluğu
        avg_line_length = np.mean([len(line) for line in lines])
        
        # Uzun satır sayısı (80+ karakter)
        long_lines = len([line for line in lines if len(line) > 80])
        
        # Boş fonksiyon sayısı (sadece pass içeren)
        empty_functions = len(re.findall(r'def\s+\w+\([^)]*\):\s*pass', code))
        
        # TODO/FIXME sayısı
        todos = len(re.findall(r'#\s*(TODO|FIXME|XXX)', code, re.IGNORECASE))
        
        # Kod tekrarı (basit string analizi)
        line_frequencies = Counter([line.strip() for line in lines if line.strip()])
        duplicated_lines = sum(count - 1 for count in line_frequencies.values() if count > 1)
        
        # Genel kalite skoru
        quality_factors = {
            "code_density": code_density * 30,
            "avg_line_length": max(0, 30 - (avg_line_length - 50) / 2) if avg_line_length > 50 else 30,
            "long_lines_penalty": max(0, 20 - long_lines * 2),
            "todos_penalty": max(0, 10 - todos),
            "duplication_penalty": max(0, 10 - duplicated_lines * 0.5)
        }
        
        quality_score = sum(quality_factors.values())
        
        return {
            "code_density": code_density,
            "avg_line_length": avg_line_length,
            "long_lines_count": long_lines,
            "empty_functions": empty_functions,
            "todos_count": todos,
            "duplicated_lines": duplicated_lines,
            "quality_score": min(quality_score, 100),
            "quality_factors": quality_factors
        }
    
    def _analyze_code_style(self, code: str, language: str) -> Dict[str, Any]:
        """Kod stili analiz eder"""
        
        style_issues = []
        style_score = 100
        
        lines = code.split('\n')
        
        # Naming convention kontrolü
        if language == 'python':
            # Snake_case kontrolü
            function_names = re.findall(r'def\s+(\w+)', code)
            for name in function_names:
                if not re.match(r'^[a-z_][a-z0-9_]*$', name):
                    style_issues.append(f"Function name '{name}' doesn't follow snake_case")
                    style_score -= 2
        
        # Indentation kontrolü
        inconsistent_indents = 0
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces % 4 != 0:  # 4 space indentation expected
                    inconsistent_indents += 1
        
        if inconsistent_indents > 0:
            style_issues.append(f"{inconsistent_indents} lines with inconsistent indentation")
            style_score -= inconsistent_indents
        
        # Çok uzun satırlar
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            style_issues.append(f"Lines too long (>120 chars): {long_lines[:5]}")
            style_score -= len(long_lines) * 0.5
        
        return {
            "style_score": max(style_score, 0),
            "style_issues": style_issues,
            "issues_count": len(style_issues)
        }
    
    def generate_documentation(self, code: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Kod için otomatik dokümantasyon üretir"""
        
        print("📖 Otomatik dokümantasyon üretiliyor...")
        
        # Kod özeti oluştur
        language = analysis["language"]
        structure = analysis["structure"]
        
        # Ana bilgiler
        doc_prompt = f"""Bu bir {language} kodu analizi:
        - {structure['function_count']} fonksiyon
        - {structure['class_count']} sınıf
        - {analysis['metrics']['code_lines']} kod satırı
        Kod hakkında açıklayıcı dokümantasyon yazın:"""
        
        try:
            # AI ile dokümantasyon üret
            generated_docs = self.doc_generator(
                doc_prompt,
                max_length=300,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.code_tokenizer.eos_token_id
            )
            
            ai_description = generated_docs[0]['generated_text'][len(doc_prompt):].strip()
            
        except Exception as e:
            ai_description = f"Otomatik açıklama üretilemedi: {e}"
        
        # Manuel dokümantasyon bileşenleri
        function_docs = []
        for func in structure["functions"]:
            func_doc = {
                "name": func["name"],
                "signature": f"{func['name']}() with {func['args_count']} parameters",
                "description": func.get("docstring", "No description available"),
                "has_docstring": func["has_docstring"]
            }
            function_docs.append(func_doc)
        
        class_docs = []
        for cls in structure["classes"]:
            class_doc = {
                "name": cls["name"],
                "description": cls.get("docstring", "No description available"),
                "methods_count": cls.get("method_count", 0),
                "inheritance": cls.get("inheritance", [])
            }
            class_docs.append(class_doc)
        
        # Genel özet
        complexity_level = "Low" if analysis["complexity"]["complexity_score"] < 30 else \
                          "Medium" if analysis["complexity"]["complexity_score"] < 60 else "High"
        
        quality_level = "Excellent" if analysis["quality"]["quality_score"] > 80 else \
                       "Good" if analysis["quality"]["quality_score"] > 60 else \
                       "Needs Improvement"
        
        return {
            "ai_generated_description": ai_description,
            "summary": {
                "language": language,
                "complexity_level": complexity_level,
                "quality_level": quality_level,
                "maintainability": "High" if analysis["quality"]["quality_score"] > 70 else "Medium"
            },
            "functions": function_docs,
            "classes": class_docs,
            "recommendations": self._generate_recommendations(analysis),
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Kod iyileştirme önerileri üretir"""
        
        recommendations = []
        
        # Kalite bazlı öneriler
        quality = analysis["quality"]
        
        if quality["quality_score"] < 60:
            recommendations.append("⚠️ Genel kod kalitesi iyileştirilebilir")
        
        if quality["long_lines_count"] > 5:
            recommendations.append("📏 Çok uzun satırları kısaltmayı düşünün (80-120 karakter)")
        
        if quality["todos_count"] > 3:
            recommendations.append("📝 TODO ve FIXME notlarını gözden geçirin")
        
        if quality["duplicated_lines"] > 10:
            recommendations.append("🔄 Kod tekrarlarını azaltın, fonksiyon extraction kullanın")
        
        # Kompleksite bazlı öneriler
        complexity = analysis["complexity"]
        
        if complexity["complexity_score"] > 70:
            recommendations.append("🧠 Kod karmaşıklığı yüksek, fonksiyonları küçültmeyi düşünün")
        
        if complexity["nesting_depth"] > 4:
            recommendations.append("🔗 İç içe geçmiş kod bloklarını azaltın")
        
        # Yorum bazlı öneriler
        comment_ratio = analysis["metrics"]["comment_ratio"]
        
        if comment_ratio < 0.1:
            recommendations.append("💬 Daha fazla açıklayıcı yorum ekleyin")
        elif comment_ratio > 0.4:
            recommendations.append("📝 Çok fazla yorum var, kodun kendini açıklaması daha iyi")
        
        # Stil bazlı öneriler
        style = analysis["style"]
        
        if style["issues_count"] > 0:
            recommendations.append(f"🎨 {style['issues_count']} stil sorunu düzeltilmeli")
        
        if not recommendations:
            recommendations.append("✅ Kod kalitesi iyi görünüyor!")
        
        return recommendations
    
    def analyze_project_directory(self, directory_path: str) -> Dict[str, Any]:
        """Tüm proje dizinini analiz eder"""
        
        print(f"📁 Proje dizini analiz ediliyor: {directory_path}")
        
        project_path = Path(directory_path)
        if not project_path.exists():
            return {"error": f"Directory not found: {directory_path}"}
        
        # Tüm kod dosyalarını bul
        code_files = []
        for lang_info in self.supported_languages.values():
            for ext in lang_info['extensions']:
                code_files.extend(project_path.rglob(f"*{ext}"))
        
        if not code_files:
            return {"error": "No code files found in directory"}
        
        print(f"🔍 {len(code_files)} kod dosyası bulundu")
        
        # Her dosyayı analiz et
        file_analyses = []
        project_stats = {
            "total_files": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "languages": set(),
            "avg_quality": 0,
            "avg_complexity": 0
        }
        
        for file_path in code_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                # Dosya analizi
                language = self.detect_language(code_content, file_path.name)
                analysis = self.analyze_code_structure(code_content, language)
                documentation = self.generate_documentation(code_content, analysis)
                
                file_analysis = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_size": len(code_content),
                    "analysis": analysis,
                    "documentation": documentation,
                    "analyzed_at": datetime.now().isoformat()
                }
                
                file_analyses.append(file_analysis)
                
                # Proje istatistiklerini güncelle
                project_stats["total_files"] += 1
                project_stats["total_lines"] += analysis["metrics"]["total_lines"]
                project_stats["total_functions"] += analysis["structure"]["function_count"]
                project_stats["total_classes"] += analysis["structure"]["class_count"]
                project_stats["languages"].add(language)
                
                print(f"✅ Analiz tamamlandı: {file_path.name}")
                
            except Exception as e:
                print(f"❌ Dosya analiz hatası {file_path.name}: {e}")
                continue
        
        # Ortalama değerleri hesapla
        if file_analyses:
            project_stats["avg_quality"] = np.mean([
                fa["analysis"]["quality"]["quality_score"] for fa in file_analyses
            ])
            project_stats["avg_complexity"] = np.mean([
                fa["analysis"]["complexity"]["complexity_score"] for fa in file_analyses
            ])
        
        project_stats["languages"] = list(project_stats["languages"])
        
        # Proje raporu
        project_report = {
            "project_path": str(project_path),
            "analyzed_at": datetime.now().isoformat(),
            "statistics": project_stats,
            "file_analyses": file_analyses,
            "project_recommendations": self._generate_project_recommendations(project_stats, file_analyses)
        }
        
        # Veritabanına ekle
        self.project_reports.append(project_report)
        
        return project_report
    
    def _generate_project_recommendations(self, stats: Dict[str, Any], 
                                        file_analyses: List[Dict[str, Any]]) -> List[str]:
        """Proje seviyesinde öneriler üretir"""
        
        recommendations = []
        
        # Kalite bazlı öneriler
        if stats["avg_quality"] < 60:
            recommendations.append("🔧 Proje genelinde kod kalitesi iyileştirilebilir")
        
        if stats["avg_complexity"] > 70:
            recommendations.append("🧠 Proje karmaşıklığı yüksek, refactoring düşünülmeli")
        
        # Dosya bazlı öneriler
        large_files = [fa for fa in file_analyses if fa["file_size"] > 10000]
        if large_files:
            recommendations.append(f"📄 {len(large_files)} büyük dosya var, bölünmeyi düşünün")
        
        # Dil çeşitliliği
        if len(stats["languages"]) > 3:
            recommendations.append("🔀 Çok fazla programlama dili kullanılıyor, tutarlılığı gözden geçirin")
        
        # Fonksiyon dağılımı
        files_with_many_functions = [
            fa for fa in file_analyses 
            if fa["analysis"]["structure"]["function_count"] > 20
        ]
        if files_with_many_functions:
            recommendations.append("⚡ Bazı dosyalarda çok fazla fonksiyon var")
        
        if not recommendations:
            recommendations.append("✅ Proje yapısı genel olarak iyi görünüyor!")
        
        return recommendations
    
    def save_project_report(self, project_report: Dict[str, Any], 
                          output_dir: str = "./reports") -> str:
        """Proje raporunu kaydeder"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = Path(project_report["project_path"]).name
        
        # JSON raporu
        json_file = output_path / f"code_analysis_{project_name}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(project_report, f, ensure_ascii=False, indent=2)
        
        # Markdown raporu
        md_file = output_path / f"code_report_{project_name}_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_report(project_report))
        
        print(f"💾 Proje raporu kaydedildi:")
        print(f"   📊 JSON: {json_file}")
        print(f"   📋 Markdown: {md_file}")
        
        return timestamp
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Markdown formatında rapor üretir"""
        
        stats = report["statistics"]
        
        md_content = f"""# 📊 Kod Analizi Raporu

## 📁 Proje Bilgileri
- **Proje Yolu:** `{report['project_path']}`
- **Analiz Tarihi:** {report['analyzed_at']}
- **Toplam Dosya:** {stats['total_files']}
- **Toplam Satır:** {stats['total_lines']:,}

## 🎯 Genel İstatistikler
- **Programlama Dilleri:** {', '.join(stats['languages'])}
- **Toplam Fonksiyon:** {stats['total_functions']}
- **Toplam Sınıf:** {stats['total_classes']}
- **Ortalama Kalite Skoru:** {stats['avg_quality']:.1f}/100
- **Ortalama Karmaşıklık:** {stats['avg_complexity']:.1f}/100

## 🏆 En İyi ve En Kötü Dosyalar

### ✅ En Kaliteli Dosyalar
"""
        
        # En iyi dosyaları bul
        sorted_files = sorted(
            report["file_analyses"], 
            key=lambda x: x["analysis"]["quality"]["quality_score"], 
            reverse=True
        )
        
        for i, file_analysis in enumerate(sorted_files[:3], 1):
            analysis = file_analysis["analysis"]
            md_content += f"""
{i}. **{file_analysis['file_name']}**
   - Kalite: {analysis['quality']['quality_score']:.1f}/100
   - Karmaşıklık: {analysis['complexity']['complexity_score']:.1f}/100
   - Fonksiyon: {analysis['structure']['function_count']}
"""

        md_content += "\n### ⚠️ İyileştirme Gereken Dosyalar\n"
        
        for i, file_analysis in enumerate(sorted_files[-3:], 1):
            analysis = file_analysis["analysis"]
            md_content += f"""
{i}. **{file_analysis['file_name']}**
   - Kalite: {analysis['quality']['quality_score']:.1f}/100
   - Sorunlar: {analysis['style']['issues_count']} stil sorunu
   - Öneriler: {len(file_analysis['documentation']['recommendations'])} öneri
"""

        md_content += "\n## 🔧 Proje Önerileri\n"
        
        for rec in report["project_recommendations"]:
            md_content += f"- {rec}\n"
        
        md_content += f"""
## 📈 Detaylı Dosya Analizi

| Dosya | Dil | Kalite | Karmaşıklık | Fonksiyon | Sınıf |
|-------|-----|--------|-------------|-----------|-------|
"""
        
        for file_analysis in sorted_files:
            analysis = file_analysis["analysis"]
            md_content += f"| {file_analysis['file_name']} | {analysis['language']} | {analysis['quality']['quality_score']:.1f} | {analysis['complexity']['complexity_score']:.1f} | {analysis['structure']['function_count']} | {analysis['structure']['class_count']} |\n"
        
        return md_content
    
    def interactive_code_studio(self):
        """İnteraktif kod analiz stüdyosu"""
        
        print("\n💻 İNTERAKTİF KOD ANALİZ STÜDYOSU")
        print("=" * 60)
        print("Komutlar:")
        print("  /analyze <dosya_yolu> - Tek dosya analizi")
        print("  /project <dizin_yolu> - Proje analizi")
        print("  /paste - Kod yapıştırarak analiz et")
        print("  /languages - Desteklenen dilleri göster")
        print("  /history - Son analizleri göster")
        print("  /stats - Genel istatistikler")
        print("  /quit - Çıkış")
        print("-" * 60)
        
        session_analyses = []
        
        while True:
            user_input = input("\n🔍 Komut veya kod: ").strip()
            
            if user_input == "/quit":
                print("👋 Kod analiz stüdyosu kapatılıyor...")
                if session_analyses:
                    save_session = input("Bu oturumun analizlerini kaydetmek ister misiniz? (y/n): ")
                    if save_session.lower() in ['y', 'yes', 'evet']:
                        self._save_session_analyses(session_analyses)
                break
            
            elif user_input == "/languages":
                print("\n🌐 Desteklenen Diller:")
                for lang, info in self.supported_languages.items():
                    extensions = ', '.join(info['extensions'])
                    print(f"  📝 {lang.upper()}: {extensions}")
                continue
            
            elif user_input.startswith("/analyze "):
                file_path = user_input[9:].strip()
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    
                    language = self.detect_language(code_content, file_path)
                    analysis = self.analyze_code_structure(code_content, language)
                    documentation = self.generate_documentation(code_content, analysis)
                    
                    result = {
                        "file_path": file_path,
                        "analysis": analysis,
                        "documentation": documentation,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    session_analyses.append(result)
                    self._display_analysis_result(result)
                    
                except Exception as e:
                    print(f"❌ Dosya analiz hatası: {e}")
                continue
            
            elif user_input.startswith("/project "):
                project_path = user_input[9:].strip()
                try:
                    project_report = self.analyze_project_directory(project_path)
                    if "error" not in project_report:
                        self._display_project_summary(project_report)
                        
                        save_report = input("Proje raporunu kaydetmek ister misiniz? (y/n): ")
                        if save_report.lower() in ['y', 'yes', 'evet']:
                            self.save_project_report(project_report)
                    else:
                        print(f"❌ {project_report['error']}")
                        
                except Exception as e:
                    print(f"❌ Proje analiz hatası: {e}")
                continue
            
            elif user_input == "/paste":
                print("📋 Kod yapıştırın (bitirmek için boş satırda 'END' yazın):")
                code_lines = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    code_lines.append(line)
                
                if code_lines:
                    code_content = '\n'.join(code_lines)
                    language = self.detect_language(code_content)
                    analysis = self.analyze_code_structure(code_content, language)
                    documentation = self.generate_documentation(code_content, analysis)
                    
                    result = {
                        "source": "pasted_code",
                        "analysis": analysis,
                        "documentation": documentation,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    session_analyses.append(result)
                    self._display_analysis_result(result)
                else:
                    print("❌ Kod girilmedi!")
                continue
            
            elif user_input == "/history":
                if session_analyses:
                    print(f"\n📚 Bu oturumda {len(session_analyses)} analiz yapıldı:")
                    for i, analysis in enumerate(session_analyses[-5:], 1):
                        source = analysis.get("file_path", analysis.get("source", "unknown"))
                        quality = analysis["analysis"]["quality"]["quality_score"]
                        print(f"  {i}. {source} - Kalite: {quality:.1f}")
                else:
                    print("📚 Henüz analiz yapılmadı!")
                continue
            
            elif user_input == "/stats":
                if session_analyses:
                    self._show_session_stats(session_analyses)
                else:
                    print("📊 İstatistik için analiz gerekli!")
                continue
            
            elif user_input == "":
                continue
            
            else:
                print("❓ Bilinmeyen komut. Yardım için komutları kontrol edin.")
    
    def _display_analysis_result(self, result: Dict[str, Any]):
        """Analiz sonucunu güzel formatta gösterir"""
        
        analysis = result["analysis"]
        documentation = result["documentation"]
        
        print(f"\n📊 KOD ANALİZİ SONUCU")
        print("=" * 50)
        print(f"🔤 Dil: {analysis['language'].upper()}")
        print(f"📏 Toplam Satır: {analysis['metrics']['total_lines']}")
        print(f"⚡ Fonksiyon: {analysis['structure']['function_count']}")
        print(f"🏗️  Sınıf: {analysis['structure']['class_count']}")
        
        print(f"\n🎯 SKORLAR:")
        print(f"  ⭐ Kalite: {analysis['quality']['quality_score']:.1f}/100")
        print(f"  🧠 Karmaşıklık: {analysis['complexity']['complexity_score']:.1f}/100")
        print(f"  🎨 Stil: {analysis['style']['style_score']:.1f}/100")
        
        print(f"\n💡 ÖNERİLER:")
        for rec in documentation['recommendations'][:3]:
            print(f"  • {rec}")
        
        if len(documentation['recommendations']) > 3:
            print(f"  ... ve {len(documentation['recommendations']) - 3} öneri daha")
    
    def _display_project_summary(self, project_report: Dict[str, Any]):
        """Proje özeti gösterir"""
        
        stats = project_report["statistics"]
        
        print(f"\n📁 PROJE ANALİZİ ÖZETİ")
        print("=" * 50)
        print(f"📂 Proje: {Path(project_report['project_path']).name}")
        print(f"📄 Dosya Sayısı: {stats['total_files']}")
        print(f"📏 Toplam Satır: {stats['total_lines']:,}")
        print(f"🌐 Diller: {', '.join(stats['languages'])}")
        
        print(f"\n🎯 GENEL SKORLAR:")
        print(f"  ⭐ Ortalama Kalite: {stats['avg_quality']:.1f}/100")
        print(f"  🧠 Ortalama Karmaşıklık: {stats['avg_complexity']:.1f}/100")
        
        print(f"\n🏆 EN İYİ DOSYALAR:")
        sorted_files = sorted(
            project_report["file_analyses"], 
            key=lambda x: x["analysis"]["quality"]["quality_score"], 
            reverse=True
        )
        
        for i, file_analysis in enumerate(sorted_files[:3], 1):
            quality = file_analysis["analysis"]["quality"]["quality_score"]
            print(f"  {i}. {file_analysis['file_name']} - {quality:.1f}")
    
    def _show_session_stats(self, session_analyses: List[Dict[str, Any]]):
        """Oturum istatistiklerini gösterir"""
        
        print(f"\n📊 OTURUM İSTATİSTİKLERİ")
        print("=" * 40)
        print(f"📈 Toplam Analiz: {len(session_analyses)}")
        
        if session_analyses:
            quality_scores = [a["analysis"]["quality"]["quality_score"] for a in session_analyses]
            complexity_scores = [a["analysis"]["complexity"]["complexity_score"] for a in session_analyses]
            
            print(f"⭐ Ortalama Kalite: {np.mean(quality_scores):.1f}")
            print(f"🧠 Ortalama Karmaşıklık: {np.mean(complexity_scores):.1f}")
            print(f"🏆 En Yüksek Kalite: {max(quality_scores):.1f}")
            print(f"⚠️ En Düşük Kalite: {min(quality_scores):.1f}")
            
            # Dil dağılımı
            languages = [a["analysis"]["language"] for a in session_analyses]
            lang_counts = Counter(languages)
            
            print(f"\n🌐 Dil Dağılımı:")
            for lang, count in lang_counts.most_common():
                print(f"  {lang}: {count}")
    
    def _save_session_analyses(self, session_analyses: List[Dict[str, Any]]):
        """Oturum analizlerini kaydeder"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_analyses_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_analyses, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Oturum analizleri kaydedildi: {filename}")

def main():
    """Ana program - Akıllı kod analizi sistemi"""
    
    print("💻 AKILLI KOD ANALİZİ VE DOKÜMANTASYON SİSTEMİ")
    print("=" * 70)
    
    # Sistem başlatma
    analyzer = IntelligentCodeAnalyzer()
    
    # Modelleri yükle
    if not analyzer.initialize_models():
        print("❌ Modeller yüklenemedi, program sonlandırılıyor.")
        return
    
    print("\n🎯 Ne yapmak istiyorsunuz?")
    print("1. 📁 Proje dizini analizi")
    print("2. 📄 Tek dosya analizi")  
    print("3. 💻 İnteraktif kod stüdyosu")
    print("4. 🧪 Demo analiz çalıştır")
    
    choice = input("\nSeçiminiz (1-4): ").strip()
    
    if choice == "1":
        project_path = input("📁 Proje dizini yolu: ").strip()
        if project_path:
            project_report = analyzer.analyze_project_directory(project_path)
            if "error" not in project_report:
                analyzer._display_project_summary(project_report)
                analyzer.save_project_report(project_report)
            else:
                print(f"❌ {project_report['error']}")
    
    elif choice == "2":
        file_path = input("📄 Dosya yolu: ").strip()
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                language = analyzer.detect_language(code_content, file_path)
                analysis = analyzer.analyze_code_structure(code_content, language)
                documentation = analyzer.generate_documentation(code_content, analysis)
                
                result = {
                    "file_path": file_path,
                    "analysis": analysis,
                    "documentation": documentation
                }
                
                analyzer._display_analysis_result(result)
                
            except Exception as e:
                print(f"❌ Dosya analiz hatası: {e}")
    
    elif choice == "3":
        analyzer.interactive_code_studio()
    
    elif choice == "4":
        print("🧪 Demo kod analizi çalıştırılıyor...")
        
        # Örnek Python kodu
        demo_code = '''
def calculate_fibonacci(n):
    """Fibonacci sayısını hesaplar"""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class MathUtils:
    """Matematik yardımcı sınıfı"""
    
    def __init__(self):
        self.pi = 3.14159
    
    def circle_area(self, radius):
        return self.pi * radius * radius
    
    def factorial(self, n):
        if n <= 1:
            return 1
        return n * self.factorial(n-1)

# TODO: Optimize fibonacci function
# FIXME: Add input validation
'''
        
        language = analyzer.detect_language(demo_code)
        analysis = analyzer.analyze_code_structure(demo_code, language)
        documentation = analyzer.generate_documentation(demo_code, analysis)
        
        result = {
            "source": "demo_code",
            "analysis": analysis,
            "documentation": documentation
        }
        
        analyzer._display_analysis_result(result)
    
    print(f"\n🎉 Program tamamlandı!")

if __name__ == "__main__":
    main()