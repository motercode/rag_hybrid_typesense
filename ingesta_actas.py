import os
import re
import fitz  # PyMuPDF
import typesense
import hashlib
import pytesseract
from pdf2image import convert_from_path

# --- ELIMINAR WARNINGS DE HUGGINGFACE ---
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from sentence_transformers import SentenceTransformer

# --- SISTEMA DE CLASIFICACIÓN DE INSTALACIONES ---
DICCIONARIO_INSTALACIONES = {
    "piscina": [r"\bpiscina\b", r"\bdepuradora\b", r"\bsocorrista\b", r"\bvaso\b", r"\bcloro\b"],
    "padel": [r"\bp[áa]del\b", r"\bpista de p[áa]del\b", r"\bred\b", r"\bcristales\b"],
    "conserjeria": [r"\bconserje\b", r"\bconserjer[ií]a\b", r"\bportero\b", r"\bporter[ií]a\b"],
    "parque_infantil": [r"\bparque infantil\b", r"\bcolumpios\b", r"\btobog[aá]n\b", r"\bjuegos infantiles\b"],
    "trasteros": [r"\btrastero[s]?\b", r"\bzona de trasteros\b"],
    "paqueteria": [r"\bpaqueter[ií]a\b", r"\bbuzones inteligentes\b", r"\bcitibox\b"]
}

def extraer_instalaciones(texto):
    texto_lower = texto.lower()
    instalaciones_encontradas = set()
    
    for instalacion, patrones in DICCIONARIO_INSTALACIONES.items():
        for patron in patrones:
            if re.search(patron, texto_lower):
                instalaciones_encontradas.add(instalacion)
                break # Con encontrar una sola palabra clave de la categoría, pasamos a la siguiente
                
    return list(instalaciones_encontradas)

print("Cargando modelo de embeddings (CPU)...")
modelo_emb = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
host_typesense = os.environ.get('TYPESENSE_HOST', 'localhost')
DIRECTORIO_ACTAS = '/app/actas'

cliente = typesense.Client({
  'nodes': [{'host': host_typesense, 'port': '8108', 'protocol': 'http'}],
  'api_key': 'clave_secreta_actas',
  'connection_timeout_seconds': 2
})

esquema_actas = {
    'name': 'comunidad_vecinos',
    'fields': [
        {'name': 'id', 'type': 'string'}, 
        {'name': 'año', 'type': 'int32', 'facet': True}, 
        {'name': 'archivo_origen', 'type': 'string' , 'locale': 'es'}, 
        {'name': 'hash_archivo', 'type': 'string', 'facet': True}, 
        {'name': 'texto', 'type': 'string'}, 
        {'name': 'embedding', 'type': 'float[]', 'num_dim': 384},
        # --- NUEVO CAMPO ---
        {'name': 'instalaciones', 'type': 'string[]', 'facet': True, 'optional': True} 
    ]
}

try:
    cliente.collections['comunidad_vecinos'].retrieve()
    print("Base de datos encontrada. Modo Ingesta Incremental activado.")
except typesense.exceptions.ObjectNotFound:
    print("Primera ejecución: Creando base de datos con soporte de Hashes en Typesense...")
    cliente.collections.create(esquema_actas)

def trocear_texto(texto, max_palabras=150, solapamiento=30):
    palabras = texto.split()
    fragmentos = []
    for i in range(0, len(palabras), max_palabras - solapamiento):
        fragmento = " ".join(palabras[i:i + max_palabras])
        if len(fragmento.strip()) > 20:
            fragmentos.append(fragmento)
    return fragmentos

def calcular_hash_archivo(ruta_archivo):
    sha256 = hashlib.sha256()
    with open(ruta_archivo, 'rb') as f:
        for bloque in iter(lambda: f.read(4096), b""):
            sha256.update(bloque)
    return sha256.hexdigest()

archivos = [f for f in os.listdir(DIRECTORIO_ACTAS) if f.lower().endswith('.pdf')]
print(f"\nSe han encontrado {len(archivos)} PDFs en la carpeta.")

for archivo in archivos:
    ruta_completa = os.path.join(DIRECTORIO_ACTAS, archivo)
    hash_actual = calcular_hash_archivo(ruta_completa)
    
    busqueda_duplicados = cliente.collections['comunidad_vecinos'].documents.search({
        'q': '*',
        'filter_by': f'hash_archivo:={hash_actual}',
        'per_page': 1
    })
    
    if busqueda_duplicados['found'] > 0:
        print(f"⏩ Saltando '{archivo}' (Ya indexado).")
        continue

    print(f"📄 Procesando nuevo archivo: '{archivo}'...")
    
    # NUEVA EXTRACCIÓN DEL AÑO: formato AAMMDD (ej: 030818 -> 2003)
    match_fecha = re.match(r'^(\d{2})\d{4}', archivo)
    if match_fecha:
        año_acta = 2000 + int(match_fecha.group(1))
    else:
        año_acta = 2000 # Fallback

    documento_pdf = fitz.open(ruta_completa)
    texto_completo = ""
    for pagina in documento_pdf:
        texto_completo += pagina.get_text() + "\n"
    
    if len(texto_completo.strip()) < 50:
        print(f"   ⚠️ PDF escaneado detectado. Aplicando OCR a '{archivo}'...")
        texto_completo = ""
        imagenes = convert_from_path(ruta_completa)
        for img in imagenes:
            texto_completo += pytesseract.image_to_string(img, lang='spa') + "\n"

    fragmentos = trocear_texto(texto_completo)
    
    documentos_batch = []
    for i, fragmento in enumerate(fragmentos):
        vector = modelo_emb.encode(fragmento).tolist()
        
        # Analizamos de qué instalaciones habla este fragmento
        instalaciones_detectadas = extraer_instalaciones(fragmento)
        
        doc = {
            'id': f"{hash_actual}_{i}",
            'año': año_acta,
            'archivo_origen': archivo,
            'hash_archivo': hash_actual,
            'texto': fragmento,
            'embedding': vector,
            'instalaciones': instalaciones_detectadas # Lo añadimos aquí
        }
        documentos_batch.append(doc)
    
    if documentos_batch:
        cliente.collections['comunidad_vecinos'].documents.import_(documentos_batch, {'action': 'upsert'})
        print(f"   ✅ Indexados {len(documentos_batch)} fragmentos.")

print("\n🎉 Proceso de ingesta finalizado.")