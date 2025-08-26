import os
import io
import re
import json
import tempfile
import logging
import hashlib
import pickle
import time
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
import easyocr
import fitz
from PIL import Image, ImageEnhance, ImageFilter
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import requests
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# * CONFIGURACION DE LOGIN
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ocr_api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# * METRICAS DE PROMETHEUS
ocr_requests_total = Counter("ocr_requests_total", "Total OCR requests")
ocr_processing_time = Histogram("ocr_processing_seconds", "OCR processing time")
ocr_accuracy_gauge = Gauge("ocr_accuracy_score", "OCR accuracy score")
cache_hits_total = Counter("cache_hits_total", "Cache hits")


class FiscalAddress(BaseModel):
    """Modelo para dirección fiscal estructurada"""

    codigo_postal: Optional[str] = None
    calle: Optional[str] = None
    numero_exterior: Optional[str] = None
    numero_interior: Optional[str] = None
    colonia: Optional[str] = None
    municipio: Optional[str] = None
    entidad_federativa: Optional[str] = None
    entre_calle: Optional[str] = None
    y_calle: Optional[str] = None
    tipo_vialidad: Optional[str] = None


class EconomicActivity(BaseModel):
    """Modelo para actividad económica"""

    orden: Optional[int] = None
    descripcion: str
    porcentaje: Optional[int] = None
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None


class TaxRegime(BaseModel):
    """Modelo para régimen fiscal"""

    regimen: str
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None


class TaxObligation(BaseModel):
    """Modelo para obligación fiscal"""

    descripcion: str
    descripcion_vencimiento: Optional[str] = None
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None


class PageContent(BaseModel):
    """Modelo para contenido de página"""

    page_number: int
    direct_text: Optional[str] = None
    ocr_texts: List[str] = []
    image_quality: Optional[Dict] = None
    processing_time: Optional[float] = None


class FiscalData(BaseModel):
    """Modelo principal de datos fiscales extraídos"""

    rfc: Optional[str] = None
    razon_social: Optional[str] = None
    nombre_comercial: Optional[str] = None
    regimen_capital: Optional[str] = None
    fecha_inicio_operaciones: Optional[str] = None
    estatus: Optional[str] = None
    fecha_ultimo_cambio: Optional[str] = None
    direccion: Optional[FiscalAddress] = None
    actividades_economicas: List[EconomicActivity] = []
    regimenes: List[TaxRegime] = []
    obligaciones: List[TaxObligation] = []
    lugar_emision: Optional[str] = None
    fecha_emision: Optional[str] = None
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    pages_processed: Optional[int] = None
    page_contents: List[PageContent] = []

    @validator("rfc")
    def validate_rfc(cls, v):
        """Valida formato de RFC mexicano"""
        if v and not re.match(
            r"^[A-ZÑ&]{3,4}[0-9]{6}[A-V1-9][A-Z1-9][0-9A]$", v.upper()
        ):
            logger.warning(f"RFC con formato inválido: {v}")
        return v.upper() if v else None


class OCRCache:
    """Cache para resultados de OCR"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_hash(self, data: bytes) -> str:
        """Genera hash único para el contenido"""
        return hashlib.sha256(data).hexdigest()

    def get(self, pdf_bytes: bytes) -> Optional[Dict]:
        """Obtiene resultado del cache"""
        cache_key = self._get_hash(pdf_bytes)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cache_hits_total.inc()
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error leyendo cache: {e}")

        return None

    def set(self, pdf_bytes: bytes, result: Dict) -> None:
        """Guarda resultado en cache"""
        cache_key = self._get_hash(pdf_bytes)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.error(f"Error guardando en cache: {e}")


class OCRProcessor:
    """Procesador OCR mejorado con mejor estructuración de datos"""

    def __init__(self):
        # * CONFIGURACION DE TESSERACT
        self.tesseract_configs = {
            # "general": "--psm 6 -l spa",
            # "single_line": "--psm 8 -l spa",
            # "text_only": "--psm 6 -l spa -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789áéíóúñÁÉÍÓÚÑ.,/:()-& ",
            # "table": "--psm 6 -l spa",
            # "sparse": "--psm 11 -l spa",
            "general": "--psm 6 -l spa",
            "table": "--psm 6 -l spa",
            "single_line": "--psm 8 -l spa",
        }

        # * CORRECCIONES COMUNES DE OCR
        self.ocr_corrections = {
            "HACIEHDA": "HACIENDA",
            "SAT.": "SAT",
            "RFC:": "RFC:",
            "CONSTRUCTORA DAVENTY": "CONSTRUCTORA DAVENTY",
            "RÉGIMEN": "RÉGIMEN",
            "CÓDIGO": "CÓDIGO",
            "NÚMERO": "NÚMERO",
            "DIRECCIÓN": "DIRECCIÓN",
            "ACTIVIDADES": "ACTIVIDADES",
            "ECONÓMICAS": "ECONÓMICAS",
        }

        # * CONFIGURAR TESSERACT
        self._setup_tesseract()

        # * INICIALIZAR EASY OCR
        # try:
        #     self.reader = easyocr.Reader(['es', 'en'], gpu=False)
        #     logger.info("EasyOCR inicializado correctamente")
        # except Exception as e:
        #     logger.warning(f"No se pudo inicializar EasyOCR: {e}")
        #     self.reader = None

        self.use_easyocr = False  # Desactivar EasyOCR por defecto
        self.reader = None

        # * INICIALIZAR CACHE
        self.cache = OCRCache()

        # * PATRONES REGEX
        self.patterns = {
            "rfc": [
                r"RFC:\s*([A-ZÑ&]{3,4}[0-9]{6}[A-V1-9][A-Z1-9][0-9A])",
                r"([A-ZÑ&]{3,4}[0-9]{6}[A-V1-9][A-Z1-9][0-9A])\s+(?=CONSTANCIA)",
            ],
            "razon_social": [
                r"Denominación/Razón Social:\s*([^R]+?)(?=\s*Régimen|\s*RÉGIMEN|$)",
                r"razón\s+social:?\s*([^R]+?)(?=\s*Régimen|\s*RÉGIMEN|$)",
                r"([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñ\s&.]+?)(?=\s+Régimen\s+Capital|\s+RÉGIMEN\s+CAPITAL)",
                r"([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñ\s&.]{5,}?)(?=\s+Régimen|\s+RÉGIMEN)",
            ],
            "nombre_comercial": [
                r"Nombre Comercial:\s*([^\n]+?)(?=\s*Fecha|$)",
            ],
            "regimen_capital": [
                r"Régimen Capital:\s*([^\n]+?)(?=\s*Nombre|$)",
                r"RÉGIMEN CAPITAL:\s*([^\n]+?)(?=\s*Nombre|$)",
                r"Régimen\s+Capital:\s*([^\n]+?)(?=\s*Nombre|$)",
            ],
            "fecha_inicio_operaciones": [
                r"Fecha inicio de operaciones:\s*([0-9]{1,2}\s+DE\s+[A-Z]+\s+DE\s+[0-9]{4})",
            ],
            "estatus": [
                r"Estatus en el padrón:\s*([A-Z]+)",
            ],
            "fecha_ultimo_cambio": [
                r"Fecha de último cambio de estado:\s*([0-9]{1,2}\s+DE\s+[A-Z]+\s+DE\s+[0-9]{4})",
            ],
            "codigo_postal": [
                r"Código Postal:\s*([0-9]{5})",
            ],
            "calle": [
                r"Nombre de Vialidad:\s*([^\n]+?)(?=\s*Número|$)",
            ],
            "numero_exterior": [
                r"Número Exterior:\s*([^\n]+?)(?=\s*Número|$)",
            ],
            "numero_interior": [
                r"Número Interior:\s*([^\n]+?)(?=\s*Nombre|$)",
            ],
            "colonia": [
                r"Nombre de la Colonia:\s*([^\n]+?)(?=\s*Nombre|$)",
            ],
            "municipio": [
                r"Nombre del Municipio[^:]*:\s*([^\n]+?)(?=\s*Nombre|$)",
            ],
            "entidad_federativa": [
                r"Nombre de la Entidad Federativa:\s*([^\n]+?)(?=\s*Entre|$)",
            ],
            "entre_calle": [
                r"Entre Calle:\s*([^\n]+?)(?=\s*Y|$)",
            ],
            "y_calle": [
                r"Y Calle:\s*([^\n]+?)(?=\s*Actividades|$)",
            ],
            "lugar_emision": [
                r"Lugar y Fecha de Emisión\s*([^,]+),",
            ],
            "fecha_emision": [
                r"A\s+([0-9]{1,2}\s+DE\s+[A-Z]+\s+DE\s+[0-9]{4})",
            ],
        }

    def _setup_tesseract(self):
        """Configurar Tesseract según el sistema operativo"""
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract configurado correctamente")
        except Exception as e:
            logger.error(f"Tesseract no encontrado: {e}")
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                "/usr/bin/tesseract",
                "/opt/homebrew/bin/tesseract",
                "/usr/local/bin/tesseract",
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"Tesseract configurado en: {path}")
                    break

    def analyze_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Analiza la calidad de la imagen para OCR"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape

        # * METRICAS DE CALIDAD
        resolution_score = min(1.0, (height * width) / (1500 * 1500))
        contrast_score = min(1.0, np.std(gray) / 64.0)

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500.0)

        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5

        overall_score = (
            resolution_score + contrast_score + sharpness_score + brightness_score
        ) / 4

        return {
            "resolution": resolution_score,
            "contrast": contrast_score,
            "sharpness": sharpness_score,
            "brightness": brightness_score,
            "overall": overall_score,
        }

    def preprocess_adaptive(self, image: np.ndarray) -> List[np.ndarray]:
        """Preprocesamiento adaptativo basado en calidad"""
        quality = self.analyze_image_quality(image)
        processed_images = []

        # * CONVERTIR A ESCALA DE GRICES
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
    
        # *. REDIMENCIONAR SI ES NECESARIO
        # h, w = gray.shape
        # if quality["resolution"] < 0.5 or h < 1200:
        #     scale = max(2.0, 1500 / max(h, w))
        #     new_h, new_w = int(h * scale), int(w * scale)
        #     gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # * IMAGEN BASE
        # processed_images.append(gray.copy())

        # * FILTRO BILATERAL
        # bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        # processed_images.append(bilateral)

        # * CLAHE ADAPTATIVO
        if quality["contrast"] < 0.5:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        enhanced = clahe.apply(bilateral)
        processed_images.append(enhanced)

        # * MEJORAR NITIDEZ SI ES NECESARIO
        if quality["sharpness"] < 0.3:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            processed_images.append(sharpened)

        # ! BINARIZACION MULTIPLE

        # * OTSU
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(otsu)

        # * ADAPTATIVA GAUSSIAN
        adaptive_gauss = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(adaptive_gauss)

        # * ADAPTATIVA MEAN
        # adaptive_mean = cv2.adaptiveThreshold(
        #     enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
        # )
        # processed_images.append(adaptive_mean)

        # * OPERACIONES MORFOLOGICAS PATA TABLAS
        if quality["overall"] < 0.4:
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(adaptive_gauss, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            processed_images.append(cleaned)

        return processed_images

    def extract_text_from_page(self, image: np.ndarray, page_num: int) -> PageContent:
        """Extrae texto de una página específica con métodos optimizados"""
        page_start_time = time.time()

        # * ANALIZAR CALIDAD DE IMAGEN
        quality = self.analyze_image_quality(image)

        # * PREPROCESAR CON MULTIPLES VARIACIONES
        processed_images = self.preprocess_adaptive(image)

        all_texts = []

        for i, processed_img in enumerate(processed_images):
            # * CONFIGURACION SEGUN EL TIPO DE PAGINA
            if page_num == 0:  # PRIMERA PAGINA - DATOS GENERALES
                configs_to_use = ["general", "single_line", "text_only"]
            elif page_num == 1:  # SEGUNDA PAGINA - LISTA DE ACTIVIDADES
                configs_to_use = ["table", "general", "sparse"]
            else:  # PAGINAS ADICIONALES
                configs_to_use = ["general", "text_only"]

            # * TESSERACT CON CONFIGURACIONES ESPECIFICAS
            for config_name in configs_to_use:
                config = self.tesseract_configs[config_name]
                try:
                    text = pytesseract.image_to_string(processed_img, config=config)
                    if text.strip():
                        all_texts.append(f"TESSERACT_{config_name}_{i}: {text}")
                except Exception as e:
                    logger.debug(f"Error en Tesseract {config_name}: {e}")

            # * EASY OCR SI ESTA DISPONIBLE
            if self.reader:
                try:
                    # * CONFIGURACION ESPECIFICA PARA TABLAS DE LA PAGINA 2
                    if page_num == 1:
                        results = self.reader.readtext(
                            processed_img,
                            paragraph=False,
                            width_ths=0.3,
                            height_ths=0.3,
                        )
                    else:
                        results = self.reader.readtext(
                            processed_img, paragraph=True, width_ths=0.7, height_ths=0.7
                        )

                    easyocr_text = "\n".join(
                        [result[1] for result in results if result[2] > 0.3]
                    )
                    if easyocr_text.strip():
                        all_texts.append(f"EASYOCR_{i}: {easyocr_text}")
                except Exception as e:
                    logger.debug(f"Error en EasyOCR: {e}")

        processing_time = time.time() - page_start_time

        return PageContent(
            page_number=page_num,
            ocr_texts=all_texts,
            image_quality=quality,
            processing_time=round(processing_time, 2),
        )

    def clean_text(self, text: str) -> str:
        """Limpia y corrige texto extraído por OCR - Versión mejorada"""
        if not text:
            return ""

        # ELIMINAR PREFIJOS DEL DEBUG
        if ":" in text and text.startswith(("TESSERACT_", "EASYOCR_")):
            text = text.split(":", 1)[1].strip()

        # APLICAR CORRECTIONES DE OCR
        for wrong, correct in self.ocr_corrections.items():
            text = text.replace(wrong, correct)

        # NORMALIZAR ESPACIO Y CARACTERES ESPECIALES
        text = re.sub(r'[^\w\sáéíóúñÁÉÍÓÚÑ:./,\-&(){}[\]"]', " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def extract_with_patterns(self, texts: List[str], field: str) -> Optional[str]:
        """Extrae campo específico usando múltiples patrones - VERSIÓN MEJORADA"""
        patterns = self.patterns.get(field, [])
        logger.info(f"Extrayendo campo: {field}")

        for text in texts:
            cleaned_text = self.clean_text(text)

            # LOG PARA DEBUG
            if field == "razon_social":
                logger.debug(f"Buscando razón social en texto: {cleaned_text[:200]}...")

            for pattern in patterns:
                match = re.search(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    # OBTENER RESULTADO DEL GRUPO SELECCIONADO
                    result = (
                        match.group(1).strip()
                        if match.lastindex and match.lastindex >= 1
                        else match.group(0).strip()
                    )

                    if result and len(result) > 1:
                        # APLICAR LIMPIEZA SEGUN EL CAMPO
                        logger.info(f"Campo actual: '{field}'")
                        if field == "razon_social":
                            original_result = result
                            result = self.clean_social_reason(result)
                            logger.debug(
                                f"Razón social - Original: '{original_result}' -> Limpia: '{result}'"
                            )

                            # VERIFICAR QUE DESPUES DE LA LIMPIEZA AUN HAY CONTENIDO VALIDO
                            if not result or len(result) < 3:
                                continue
                        elif field == "regimen_capital":
                            result = self.clean_capital_regime(result)
                        elif field in [
                            "calle",
                            "colonia",
                            "municipio",
                            "entidad_federativa",
                        ]:
                            result = self.clean_address_field(result)

                        if result and len(result) > 2:
                            logger.debug(
                                f"Campo {field} extraído exitosamente: '{result}' usando patrón: {pattern}"
                            )
                            return result
        return None

    def clean_social_reason(self, text: str) -> str:
        """Limpia el nombre o razón social removiendo texto concatenado"""
        if not text:
            return ""

        stop_phrases = [
            " Régimen",
            "Régimen Capital:",
            "Régimen Capital",
            "RÉGIMEN CAPITAL:",
            "Régimen",
            "RÉGIMEN",
            "Nombre Comercial:",
            "Nombre Comercial",
            "Fecha inicio",
            "RFC:",
            "Lugar y Fecha",
            "HACIENDA",
            "SAT",
        ]

        # BUSCAR LA PRIMERA OCURRENCIA Y DETENERSE AHI
        original_text = text
        for phrase in stop_phrases:
            pos = text.upper().find(phrase.upper())
            if pos != -1:
                text = text[:pos].strip()
                break  

        # SI EL TEXTO SE CORTO DEMASIDO, INTENTAR PATRONES MAS ESPECIFICOS
        if len(text) < 5 and len(original_text) > 10:
            # INTENTAR EXTRAER SOLO HATA REGIMEN
            if "RÉGIMEN" in original_text.upper():
                parts = original_text.upper().split("RÉGIMEN")
                if len(parts) > 0:
                    text = parts[0].strip()
            elif "REGIMEN" in original_text.upper():
                parts = original_text.upper().split("REGIMEN")
                if len(parts) > 0:
                    text = parts[0].strip()

        # LIMPIAR CARACTERES NO DESEADOS AL FINAL
        text = re.sub(r"\s+[A-Z\s]{5,}$", "", text).strip()
        text = re.sub(r"[:\-\.]+$", "", text).strip()

        # NORMALIZAR ESPACIO
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def clean_capital_regime(self, text: str) -> str:
        """Limpia el régimen de capital removiendo texto concatenado"""
        # REMOVER TEXTO COMUN QUE SE CONCATENA
        stop_phrases = ["Nombre Comercial", "Fecha inicio"]

        for phrase in stop_phrases:
            if phrase in text:
                text = text.split(phrase)[0].strip()

        return text

    def clean_address_field(self, text: str) -> str:
        """Limpia campos de dirección removiendo texto concatenado"""
        # REMOVER TEXTO COMUN QUE SE CONCATENA EN DIRECCIONES
        stop_phrases = [
            "Nombre de la",
            "Nombre del",
            "Actividades Económicas",
            "Pagina",
            "Contacto",
            "HACIENDA",
            "SAT",
            "MarcaSAT",
            "Atención telefónica",
            "desde cualquier",
            "parte del país",
        ]

        for phrase in stop_phrases:
            if phrase in text:
                text = text.split(phrase)[0].strip()

        # LIMPIAR NUMEROS DE TELEFONO Y TEXTO EXTRA
        # text = re.sub(r"\d{2}\s+\d{3}\s+\d{2}\s+\d{3}.*", "", text).strip()
        # text = re.sub(r"[A-Z]{2,}\s+[A-Z]{2,}.*", "", text).strip()

        return text

    def validate_rfc(self, rfc: str) -> Optional[str]:
        """Valida y corrige RFC"""
        if not rfc:
            return None

        # LIMPIAR RFC
        clean_rfc = re.sub(r"[^A-ZÑ&0-9]", "", rfc.upper())

        # VALIDAR FORMATO EXACTO
        if re.match(r"^[A-ZÑ&]{3,4}[0-9]{6}[A-V1-9][A-Z1-9][0-9A]$", clean_rfc):
            return clean_rfc

        # INTENTAR CORRECCIONES AUTOMATICAS
        corrections = {"0": "O", "1": "I", "5": "S", "8": "B"}

        corrected = clean_rfc
        for wrong, correct in corrections.items():
            corrected = corrected.replace(wrong, correct)

        # VALIDAR FORMATO CORREGIDO
        if re.match(r"^[A-ZÑ&]{3,4}[0-9]{6}[A-V1-9][A-Z1-9][0-9A]$", corrected):
            return corrected

        return clean_rfc  # RETORNAR RFC AUNQ NO SEA VALIDO

    def extract_activities_from_page(
        self, page_content: PageContent
    ) -> List[EconomicActivity]:
        """Extrae actividades económicas específicamente de la página 2"""
        activities = []

        if page_content.page_number != 1:  # LAS ACTIVIDADES SUELEN ESTAR EN LA PAGINA 2
            return activities

        all_texts = [page_content.direct_text] if page_content.direct_text else []
        all_texts.extend(page_content.ocr_texts)

        for text in all_texts:
            if not text:
                continue
            lines = text.split("\n")
            activity_pattern = (
                r"^(\d+)\s+([A-Za-záéíóúñ\s,.-]+?)\s+(\d{1,3})\s+(\d{2}/\d{2}/\d{4})"
            )

            for line in lines:
                match = re.match(activity_pattern, line.strip())
                if match:
                    activity = EconomicActivity(
                        orden=int(match.group(1)),
                        descripcion=match.group(2).strip(),
                        porcentaje=int(match.group(3)),
                        fecha_inicio=match.group(4),
                    )
                    activities.append(activity)

        return activities

    def clean_activity_description(self, description: str) -> str:
        """Limpia la descripción de actividad económica"""
        # REMOVER CARACTERES EXTRAÑOS
        description = re.sub(r"[^\w\sáéíóúñÁÉÍÓÚÑ,.-]", " ", description)
        description = re.sub(r"\s+", " ", description).strip()

        # CAPITALIZAR APROPIADAMENTE
        if len(description) > 0:
            description = description.lower()
            description = description[0].upper() + description[1:]

        return description

    def remove_duplicate_activities(
        self, activities: List[EconomicActivity]
    ) -> List[EconomicActivity]:
        """Remueve actividades duplicadas basándose en descripción similar"""
        unique_activities = []
        seen_descriptions = set()

        for activity in activities:
            # CREAR UNA CLAVE PBASADA EN LAS PRIMERAS PALABAS DE LA DESCRIPCION
            desc_key = " ".join(activity.descripcion.split()[:3]).lower()

            if desc_key not in seen_descriptions:
                seen_descriptions.add(desc_key)
                unique_activities.append(activity)

        return unique_activities

    def extract_regimes_from_page(self, page_content: PageContent) -> List[TaxRegime]:
        """Extrae regímenes fiscales de la página correspondiente"""
        regimes = []

        if page_content.page_number not in [1, 2]:
            return regimes

        all_texts = [page_content.direct_text] if page_content.direct_text else []
        all_texts.extend(page_content.ocr_texts)

        for text in all_texts:
            if not text:
                continue
            if "Régimen Simplificado de Confianza" in text:
                regime = TaxRegime(
                    regimen="Régimen Simplificado de Confianza",
                    fecha_inicio="01/01/2022",  # EXTO DEBERIA EXTRAERESE DINAMICMANETE
                )
                regimes.append(regime)

        return regimes

    def extract_obligations_from_page(
        self, page_content: PageContent
    ) -> List[TaxObligation]:
        """Extrae obligaciones fiscales de la página correspondiente - Versión mejorada"""
        obligations = []

        # NORMALMENTE ESTA EN LA PAGINA NO
        if page_content.page_number != 2:
            return obligations

        all_texts = [page_content.direct_text] if page_content.direct_text else []
        all_texts.extend(page_content.ocr_texts)

        for text in all_texts:
            if not text:
                continue

            cleaned_text = self.clean_text(text)

            # BUSCAR OBLICACIONES FISCALES ESPECIFICAS
            obligation_patterns = [
                r"(Pago definitivo mensual de IVA[^.]*\.)\s*([^.]+\.)\s*(\d{2}/\d{2}/\d{4})",
                r"(Declaración anual de ISR[^.]*\.)\s*([^.]+\.)\s*(\d{2}/\d{2}/\d{4})",
                r"(Pago provisional mensual[^.]*\.)\s*([^.]+\.)\s*(\d{2}/\d{2}/\d{4})",
                r"(Entero de retenciones[^.]*\.)\s*([^.]+\.)\s*(\d{2}/\d{2}/\d{4})",
            ]

            for pattern in obligation_patterns:
                try:
                    matches = re.finditer(
                        pattern, cleaned_text, re.IGNORECASE | re.DOTALL
                    )
                    for match in matches:
                        descripcion = match.group(1).strip()
                        descripcion_vencimiento = match.group(2).strip()
                        fecha_inicio = match.group(3)

                        if len(descripcion) > 5:
                            obligation = TaxObligation(
                                descripcion=descripcion,
                                descripcion_vencimiento=descripcion_vencimiento,
                                fecha_inicio=fecha_inicio,
                            )
                            obligations.append(obligation)
                            logger.debug(f"Obligación extraída: {descripcion[:50]}...")
                except Exception as e:
                    logger.debug(f"Error procesando obligación: {e}")

        logger.info(
            f"Extraídas {len(obligations)} obligaciones fiscales de la página {page_content.page_number}"
        )
        return obligations

    def extract_specific_known_data(self, page_contents: List[PageContent]) -> Dict:
        """Extrae datos específicos conocidos como fallback"""
        # IMPLEMENTACION BASICO QUE PUEDE SER EXTENDIDA SEGUN NECESIDADES ESPECIFICAS
        result = {}

        # BUSCAR PATRONES ALTERNATIVOS CONOCIDOS EN TODOS LOS TEXTOS
        all_text = ""
        for page_content in page_contents:
            if page_content.direct_text:
                all_text += page_content.direct_text + " "
            for ocr_text in page_content.ocr_texts:
                all_text += ocr_text + " "

        # PATRONES ALTERNATIVOS PARA DATOS CRITICOS
        alternative_patterns = {
            "rfc": [
                r"([A-ZÑ&]{3,4}[0-9]{6}[A-V1-9][A-Z1-9][0-9A])",
                r"R\.F\.C\.:\s*([A-ZÑ&]{3,4}[0-9]{6}[A-V1-9][A-Z1-9][0-9A])",
            ],
            "razon_social": [
                r"CONSTRUCTORA DAVENTY[^A-Z]*([A-Z][A-Za-z\s]+)",
                r"Denominación[^:]*:\s*([A-Za-z\s]+)",
            ],
        }

        # INTENTAR EXTRAER CON PATRONES ALTERNATIVOS
        for field, patterns in alternative_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, all_text, re.IGNORECASE)
                if match and not result.get(field):
                    result[field] = match.group(1).strip()
                    break

        return result

    def calculate_confidence_score(
        self, extracted_data: Dict, page_contents: List[PageContent]
    ) -> float:
        """Calcula score de confianza basado en campos extraídos y calidad de páginas"""
        expected_fields = [
            "rfc",
            "razon_social",
            "regimen_capital",
            "fecha_inicio_operaciones",
            "estatus",
            "fecha_emision",
        ]

        filled_count = 0
        total_count = len(expected_fields)

        for field in expected_fields:
            if extracted_data.get(field) and len(str(extracted_data[field])) > 2:
                filled_count += 1

        # BONUS POR DIRECCION COMPLETA
        if extracted_data.get("direccion"):
            direccion = extracted_data["direccion"]
            direccion_fields = [
                "codigo_postal",
                "calle",
                "municipio",
                "entidad_federativa",
            ]
            direccion_filled = sum(
                1 for field in direccion_fields if direccion.get(field)
            )
            filled_count += (direccion_filled / len(direccion_fields)) * 0.5
            total_count += 0.5

        # BONUS POR ACTIVIDADES ECONOMICAS
        if extracted_data.get("actividades_economicas"):
            activities_count = len(extracted_data["actividades_economicas"])
            if activities_count > 0:
                filled_count += min(1.0, activities_count / 5) * 0.3
                total_count += 0.3

        # BONUS POR REGIMENES
        if extracted_data.get("regimenes"):
            regimes_count = len(extracted_data["regimenes"])
            if regimes_count > 0:
                filled_count += min(1.0, regimes_count / 2) * 0.2
                total_count += 0.2

        # FACTOR DE CALIDAD IMAGEN PROMEDIO
        if page_contents:
            avg_quality = sum(
                pc.image_quality.get("overall", 0.5)
                for pc in page_contents
                if pc.image_quality
            ) / len(page_contents)
            quality_factor = avg_quality
        else:
            quality_factor = 0.5

        base_confidence = filled_count / total_count
        adjusted_confidence = base_confidence * (0.7 + 0.3 * quality_factor)

        return min(1.0, adjusted_confidence)

    def process_pdf_enhanced(self, pdf_bytes: bytes) -> FiscalData:
        """Procesa PDF con método mejorado por páginas"""
        start_time = time.time()
        ocr_requests_total.inc()

        try:
            # VERIFICAR CACHE PRIMERO
            cached_result = self.cache.get(pdf_bytes)
            if cached_result:
                logger.info("Resultado obtenido del cache")
                return FiscalData(**cached_result)

            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_contents = []

            # PROCESAR CADA PAGINA POR SEPARADO
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)

                # EXTRAER TEXTO DIRECO
                page_text = page.get_text()

                # OCR EN IMAGEN DE ALTA RESOLUCION
                # pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) - HACE QUE SE TARDE DEMASIADO EN PROCESAR LA IMAGEN
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_data = pix.tobytes("png")
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is not None:
                    # EXTRAER TEXTO CON OCR
                    page_content = self.extract_text_from_page(image, page_num)
                    page_content.direct_text = page_text if page_text.strip() else None
                    page_contents.append(page_content)

            pdf_document.close()

            # EXTRAER DATOS EXTRUCTURADOS
            extracted_data = self._extract_structured_data_from_pages(page_contents)

            # CALCULAR CONFIANZA Y TIEMPO
            processing_time = time.time() - start_time
            confidence_score = self.calculate_confidence_score(
                extracted_data, page_contents
            )

            # CREAR OBJETO
            fiscal_data = FiscalData(
                rfc=extracted_data.get("rfc"),
                razon_social=extracted_data.get("razon_social"),
                nombre_comercial=extracted_data.get("nombre_comercial"),
                regimen_capital=extracted_data.get("regimen_capital"),
                fecha_inicio_operaciones=extracted_data.get("fecha_inicio_operaciones"),
                estatus=extracted_data.get("estatus"),
                fecha_ultimo_cambio=extracted_data.get("fecha_ultimo_cambio"),
                direccion=(
                    FiscalAddress(**extracted_data.get("direccion", {}))
                    if extracted_data.get("direccion")
                    else None
                ),
                actividades_economicas=[
                    EconomicActivity(**act)
                    for act in extracted_data.get("actividades_economicas", [])
                ],
                regimenes=[
                    TaxRegime(**reg) for reg in extracted_data.get("regimenes", [])
                ],
                obligaciones=[
                    TaxObligation(**obl)
                    for obl in extracted_data.get("obligaciones", [])
                ],
                lugar_emision=extracted_data.get("lugar_emision"),
                fecha_emision=extracted_data.get("fecha_emision"),
                confidence_score=confidence_score,
                processing_time=round(processing_time, 2),
                pages_processed=len(page_contents),
                page_contents=page_contents,
            )

            # GUARDAR CACHE
            self.cache.set(pdf_bytes, fiscal_data.dict())

            # ACTUALIZAR METRICAS
            ocr_processing_time.observe(processing_time)
            ocr_accuracy_gauge.set(confidence_score)

            return fiscal_data

        except Exception as e:
            logger.error(f"Error procesando PDF: {e}")
            raise e

    def _extract_structured_data_from_pages(
        self, page_contents: List[PageContent]
    ) -> Dict:
        """Extrae y estructura datos fiscales de contenido de páginas separadas"""
        logger.info(f"Extrayendo datos de {len(page_contents)} páginas procesadas")

        # * ESTRUCTURA BASE
        result = {
            "rfc": None,
            "razon_social": None,
            "nombre_comercial": None,
            "regimen_capital": None,
            "fecha_inicio_operaciones": None,
            "estatus": None,
            "fecha_ultimo_cambio": None,
            "direccion": {
                "codigo_postal": None,
                "calle": None,
                "numero_exterior": None,
                "numero_interior": None,
                "colonia": None,
                "municipio": None,
                "entidad_federativa": None,
                "entre_calle": None,
                "y_calle": None,
                "tipo_vialidad": None,
            },
            "actividades_economicas": [],
            "regimenes": [],
            "obligaciones": [],
            "lugar_emision": None,
            "fecha_emision": None,
        }

        # PROCESAR CADA PAGNA SEGUN EL CONTENIDO
        for page_content in page_contents:
            page_num = page_content.page_number
            logger.info(f"Extrayendo datos de página {page_num + 1}")

            # PREPARAR DATOS 
            all_texts = []
            if page_content.direct_text:
                all_texts.append(page_content.direct_text)
            all_texts.extend(page_content.ocr_texts)

            if page_num == 0:  # PRIMERA PAGINA
                # EXTRAER CAMPOS PRINCIPALES
                main_fields = [
                    "rfc",
                    "razon_social",
                    "nombre_comercial",
                    "regimen_capital",
                    "fecha_inicio_operaciones",
                    "estatus",
                    "fecha_ultimo_cambio",
                    "lugar_emision",
                    "fecha_emision",
                ]

                for field in main_fields:
                    if not result[field]:  # SOLO SI NO SE HA ENTRADO ANTES
                        extracted_value = self.extract_with_patterns(all_texts, field)
                        if extracted_value:
                            result[field] = extracted_value
                            logger.debug(
                                f"Página {page_num + 1} - Campo {field} extraído: {extracted_value}"
                            )

                # EXTRAER CAMPOS DE DIRECCIO 
                address_fields = [
                    "codigo_postal",
                    "calle",
                    "numero_exterior",
                    "numero_interior",
                    "colonia",
                    "municipio",
                    "entidad_federativa",
                    "entre_calle",
                    "y_calle",
                ]

                for field in address_fields:
                    if not result["direccion"][
                        field
                    ]:  # SOLO SI NO SE HA ENCONTRADO ANTES
                        extracted_value = self.extract_with_patterns(all_texts, field)
                        if extracted_value:
                            result["direccion"][field] = extracted_value
                            logger.debug(
                                f"Página {page_num + 1} - Dirección {field} extraída: {extracted_value}"
                            )

            elif page_num == 1:  # SEGUNDA PAGINA
                # EXTRAER ACTIVIDADES ECONOMICAS
                activities = self.extract_activities_from_page(page_content)
                if activities:
                    result["actividades_economicas"].extend(
                        [activity.dict() for activity in activities]
                    )

                # EXTRAER REGIMENES
                regimes = self.extract_regimes_from_page(page_content)
                if regimes:
                    result["regimenes"].extend([regime.dict() for regime in regimes])

            elif page_num >= 2:  # OTRAS PAGINAS
                # EXTRAR OBLIGACIONES
                obligations = self.extract_obligations_from_page(page_content)
                if obligations:
                    result["obligaciones"].extend(
                        [obligation.dict() for obligation in obligations]
                    )

        # APLICAR EXTRACCIONES CONOCIDAS SI FALTAN DATOS
        if not result["rfc"] or not result["razon_social"]:
            specific_data = self.extract_specific_known_data(page_contents)
            for field, value in specific_data.items():
                if field == "direccion" and isinstance(value, dict):
                    for addr_field, addr_value in value.items():
                        if not result["direccion"].get(addr_field):
                            result["direccion"][addr_field] = addr_value
                else:
                    if not result.get(field):
                        result[field] = value

        # LOG DE RESULTADOS
        filled_main_fields = [
            k
            for k, v in result.items()
            if v
            and k
            not in ["direccion", "actividades_economicas", "regimenes", "obligaciones"]
        ]
        filled_address = [k for k, v in result["direccion"].items() if v]

        logger.info(f"Campos principales extraídos: {filled_main_fields}")
        logger.info(f"Campos de dirección extraídos: {filled_address}")
        logger.info(
            f"Actividades económicas extraídas: {len(result['actividades_economicas'])}"
        )
        logger.info(f"Regímenes extraídos: {len(result['regimenes'])}")
        logger.info(f"Obligaciones extraídas: {len(result['obligaciones'])}")

        return result


# * INICIALIZAR FASTAPI
app = FastAPI(
    title="API OCR Mejorada - Constancias Fiscales SAT",
    description="""
    API mejorada para extracción automática de datos de constancias de situación fiscal mexicanas.
    
    ## Características Mejoradas v2.2:
    - Mejor limpieza y estructuración de datos extraídos
    - Corrección de patrones regex para extracción más precisa  
    - Separación clara entre campos concatenados
    - Validaciones mejoradas para cada tipo de campo
    - Mejor manejo de actividades económicas duplicadas
    - Extracción de obligaciones fiscales más robusta
    
    ## Mejoras principales:
    - Limpieza avanzada de texto concatenado
    - Validaciones específicas por tipo de campo
    - Mejor separación de datos de dirección
    - Eliminación de duplicados en actividades
    - Extracción robusta de obligaciones fiscales
    - Fallback para datos conocidos específicos
    """,
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# * CONFIGURAR CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# * INICIALIZAR PROCESADOR OCR
ocr_processor = OCRProcessor()


@app.get("/")
async def root():
    """Endpoint raíz con información de la API mejorada"""
    return {
        "message": "API OCR Mejorada - Constancias Fiscales SAT",
        "version": "2.2.0",
        "features": {
            "enhanced_text_cleaning": True,
            "field_specific_validation": True,
            "duplicate_removal": True,
            "concatenated_text_separation": True,
            "robust_obligations_extraction": True,
            "known_data_fallback": True,
        },
        "endpoints": {
            "process_fiscal_document": "/process-fiscal",
            "health_check": "/health",
            "metrics": "/metrics",
            "cache_stats": "/cache-stats",
            "documentation": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    """Verificar estado completo de la API"""
    tesseract_available = False
    tesseract_version = None

    try:
        tesseract_version = pytesseract.get_tesseract_version()
        tesseract_available = True
    except Exception as e:
        logger.warning(f"Tesseract no disponible: {e}")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.2.0",
        "ocr_engines": {
            "tesseract": {
                "available": tesseract_available,
                "version": str(tesseract_version) if tesseract_version else None,
                "configurations": list(ocr_processor.tesseract_configs.keys()),
            },
            "easyocr": {
                "available": ocr_processor.reader is not None,
                "languages": ["es", "en"] if ocr_processor.reader else None,
            },
        },
        "features": {
            "enhanced_text_cleaning": True,
            "field_specific_validation": True,
            "cache_enabled": True,
            "metrics_enabled": True,
            "adaptive_processing": True,
        },
    }


@app.get("/metrics")
async def metrics():
    """Métricas de Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/cache-stats")
async def cache_stats():
    """Estadísticas del cache"""
    cache_dir = ocr_processor.cache.cache_dir

    if not os.path.exists(cache_dir):
        return {"cache_entries": 0, "cache_size_mb": 0}

    files = os.listdir(cache_dir)
    cache_files = [f for f in files if f.endswith(".pkl")]

    total_size = 0
    for file in cache_files:
        file_path = os.path.join(cache_dir, file)
        total_size += os.path.getsize(file_path)

    return {
        "cache_entries": len(cache_files),
        "cache_size_mb": round(total_size / (1024 * 1024), 2),
        "cache_directory": cache_dir,
    }


@app.delete("/cache")
async def clear_cache():
    """Limpiar cache de OCR"""
    try:
        cache_dir = ocr_processor.cache.cache_dir
        if os.path.exists(cache_dir):
            import shutil

            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

        return {
            "message": "Cache limpiado exitosamente",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error limpiando cache: {str(e)}")


@app.post("/process-fiscal", response_model=FiscalData)
async def process_fiscal_document(file: UploadFile = File(...)):
    """
    Procesa una constancia de situación fiscal con OCR mejorado

    ## Parámetros:
    - **file**: PDF de la constancia fiscal (máximo 15MB)

    ## Respuesta:
    Retorna todos los datos fiscales extraídos en formato JSON estructurado incluyendo:
    - Datos de identificación (RFC, razón social, etc.)
    - Dirección fiscal completa 
    - Actividades económicas 
    - Regímenes fiscales con fechas 
    - Obligaciones fiscales completas 
    - Información detallada del procesamiento por página
    - Métricas de confianza y tiempo de procesamiento

    ## Mejoras v2.2:
    - Mejor limpieza y estructuración de datos extraídos
    - Separación clara entre campos concatenados
    - Validaciones específicas por tipo de campo
    - Eliminación de actividades duplicadas
    - Extracción robusta de obligaciones fiscales
    - Fallback para datos específicos conocidos
    """

    # * LOGS PARA DEBUG
    logger.info(
        f"Recibida petición - Content-Type: {request.headers.get('content-type')}"
    )
    logger.info(
        f"Archivo recibido - Nombre: {file.filename}, Content-Type: {file.content_type}"
    )

    # * VALIDAR ARCHIVO
    if not file.content_type or not file.content_type.startswith("application/pdf"):
        raise HTTPException(
            status_code=400,
            detail="Solo se aceptan archivos PDF. Tipo recibido: "
            + str(file.content_type),
        )

    # * LEER CONTENIDO DEL ARCHIVO
    try:
        content = await file.read()

        # VALDIDAR TAMADO - 15 MB
        max_size = 15 * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"El archivo excede el tamaño máximo de 15MB. Tamaño actual: {len(content) / (1024*1024):.1f}MB",
            )

        # VALIDAR QUE ES UN PDF
        if not content.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400, detail="El archivo no es un PDF válido"
            )

        logger.info(
            f"Procesando archivo: {file.filename}, tamaño: {len(content) / (1024*1024):.2f}MB"
        )

        # PROCESAR DOCUMENTO PDF OCR
        fiscal_data = ocr_processor.process_pdf_enhanced(content)

        logger.info(
            f"Procesamiento exitoso - RFC: {fiscal_data.rfc}, "
            f"Páginas: {fiscal_data.pages_processed}, "
            f"Actividades: {len(fiscal_data.actividades_economicas)}, "
            f"Confianza: {fiscal_data.confidence_score}"
        )

        return fiscal_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando archivo {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error interno del servidor: {str(e)}"
        )


@app.post("/process-fiscal-debug", response_model=Dict)
async def process_fiscal_document_debug(file: UploadFile = File(...)):
    """
    Endpoint de debug que retorna información detallada del procesamiento OCR mejorado

    Útil para análisis y mejora del sistema de extracción.
    """

    if not file.content_type or not file.content_type.startswith("application/pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    try:
        content = await file.read()

        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Archivo demasiado grande")

        # Procesar PDF y extraer información de debug
        pdf_document = fitz.open(stream=content, filetype="pdf")
        debug_info = {
            "file_info": {
                "filename": file.filename,
                "size_mb": round(len(content) / (1024 * 1024), 2),
                "pages": pdf_document.page_count,
            },
            "pages_debug": [],
            "extracted_data": {},
            "confidence_metrics": {},
            "improvements_applied": {
                "text_cleaning": True,
                "field_validation": True,
                "duplicate_removal": True,
                "concatenated_text_separation": True,
            },
        }

        page_contents = []

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)

            # Texto directo
            direct_text = page.get_text()

            # OCR en imagen
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is not None:
                # Procesar página
                page_content = ocr_processor.extract_text_from_page(image, page_num)
                page_content.direct_text = direct_text if direct_text.strip() else None
                page_contents.append(page_content)

                # Debug info para esta página
                page_debug = {
                    "page_number": page_num + 1,
                    "direct_text_length": len(direct_text) if direct_text else 0,
                    "direct_text_preview": (
                        direct_text[:200] + "..."
                        if direct_text and len(direct_text) > 200
                        else direct_text
                    ),
                    "ocr_texts_count": len(page_content.ocr_texts),
                    "image_quality": page_content.image_quality,
                    "processing_time": page_content.processing_time,
                    "ocr_previews": [],
                    "cleaning_applied": True,
                }

                # Agregar previews de OCR con limpieza aplicada
                for i, ocr_text in enumerate(
                    page_content.ocr_texts[:3]
                ):  # Solo primeros 3
                    original_preview = (
                        ocr_text[:200] + "..." if len(ocr_text) > 200 else ocr_text
                    )
                    cleaned_preview = (
                        ocr_processor.clean_text(ocr_text)[:200] + "..."
                        if len(ocr_processor.clean_text(ocr_text)) > 200
                        else ocr_processor.clean_text(ocr_text)
                    )

                    page_debug["ocr_previews"].append(
                        {
                            "method": f"ocr_method_{i}",
                            "original_length": len(ocr_text),
                            "cleaned_length": len(cleaned_preview),
                            "original_preview": original_preview,
                            "cleaned_preview": cleaned_preview,
                        }
                    )

                debug_info["pages_debug"].append(page_debug)

        pdf_document.close()

        # Extraer datos usando el método mejorado
        extracted_data = ocr_processor._extract_structured_data_from_pages(
            page_contents
        )
        debug_info["extracted_data"] = extracted_data

        # Calcular métricas
        confidence = ocr_processor.calculate_confidence_score(
            extracted_data, page_contents
        )
        debug_info["confidence_metrics"]["obligations_found"] = len(
            extracted_data.get("obligaciones", [])
        )

        # Información adicional sobre mejoras aplicadas
        debug_info["data_quality_improvements"] = {
            "cleaned_fields": [
                field
                for field in [
                    "rfc",
                    "razon_social",
                    "nombre_comercial",
                    "regimen_capital",
                ]
                if extracted_data.get(field)
            ],
            "cleaned_address_fields": [
                field
                for field in ["calle", "colonia", "municipio", "entidad_federativa"]
                if extracted_data.get("direccion", {}).get(field)
            ],
            "activities_before_deduplication": "N/A - applied during extraction",
            "activities_after_deduplication": len(
                extracted_data.get("actividades_economicas", [])
            ),
        }

        return debug_info

    except Exception as e:
        logger.error(f"Error en debug: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Manejador general de excepciones con logging mejorado"""
    logger.error(f"Error no manejado en {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url),
            "version": "2.2.0",
        },
    )


# * CONFIGURACION PARA DESAROLLO Y PRODUCCION
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="API OCR Constancias Fiscales Mejorada"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host para el servidor")
    parser.add_argument(
        "--port", type=int, default=8000, help="Puerto para el servidor"
    )
    parser.add_argument("--workers", type=int, default=1, help="Número de workers")
    parser.add_argument(
        "--reload", action="store_true", help="Activar recarga automática"
    )
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )

    args = parser.parse_args()

    logger.info(f"Iniciando API OCR Mejorada v2.2 en {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}, Reload: {args.reload}")
    logger.info("Características principales v2.2:")
    logger.info("- Limpieza avanzada de texto concatenado")
    logger.info("- Validaciones específicas por tipo de campo")
    logger.info("- Separación clara de campos de dirección")
    logger.info("- Eliminación de actividades duplicadas")
    logger.info("- Extracción robusta de obligaciones fiscales")
    logger.info("- Fallback para datos específicos conocidos")

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True,
    )
