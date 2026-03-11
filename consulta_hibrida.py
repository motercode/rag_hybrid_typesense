import os
import re
import typesense
import requests
import json
import logging
import warnings

# --- 1. SILENCIAR WARNINGS MOLESTOS DE HUGGINGFACE ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

# 2. Configuración de Conexiones
cliente_db = typesense.Client({
  'nodes': [{'host': 'localhost', 'port': '8108', 'protocol': 'http'}],
  'api_key': 'clave_secreta_actas',
  'connection_timeout_seconds': 2
})

print("Cargando motor de embeddings local...")
modelo_emb = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

URL_OLLAMA = "http://localhost:11434/api/generate"
MODELO_LLM = "ActasRAG"

def analizar_pregunta_con_ia(pregunta_original):
    print(f"\n[1] Pidiendo a Ollama que traduzca la pregunta a lenguaje de Typesense...")
    
    prompt = f"""Eres un experto en bases de datos. Analiza la siguiente consulta de un usuario y extrae cuatro cosas:
    1. La instalación mencionada ('filtro_instalacion') seleccionada de una lista estricta.
    2. Un filtro de año ('filtro_ano') usando la sintaxis de Typesense.
    3. La pregunta original limpia ('question') sin menciones a años.
    4. 4 variantes cortas sobre el texto inicial (máximo 4 palabras cada una).

    REGLAS PARA EL FILTRO DE AÑO:
    - "en 2012" -> "año:=2012"
    - "antes de 2015" -> "año:<2015"
    - "entre 2000 y 2008" -> "año:>=2000 && año:<=2008"
    - Si no hay fechas -> ""
    
    REGLAS PARA LA INSTALACIÓN:
    - Las únicas categorías válidas son: piscina, padel, conserjeria, parque_infantil, trasteros, paqueteria.
    - Si el usuario menciona "socorrista" o "depuradora", clasifícalo como "piscina".
    - Si NO menciona ninguna instalación clara -> ""

    Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta:
    {{
    "question": "Pregunta original limpia de años",
    "variantes": ["variante 1", "variante 2", "variante 3", "variante 4"],
    "filtro_ano": "filtro de años o cadena vacía",
    "filtro_instalacion": "nombre de la instalacion o cadena vacía"
    }}
    
    Consulta: "{pregunta_original}"
    """
    
    payload = {
        "model": MODELO_LLM, 
        "prompt": prompt, 
        "stream": False, 
        "format": "json",
        "options": {"temperature": 0.1}
    }
    
    try:
        respuesta = requests.post(URL_OLLAMA, json=payload).json()
        datos_json = json.loads(respuesta['response'])
        
        variantes = datos_json.get("variantes", [])
        filtro_ano = datos_json.get("filtro_ano", "").strip()
        filtro_inst = datos_json.get("filtro_instalacion", "").strip().lower()
        pregunta_limpia = datos_json.get("question", pregunta_original)
        
        filtros_combinados = []
        if filtro_ano:
            filtros_combinados.append(filtro_ano)
        if filtro_inst in ["piscina", "padel", "conserjeria", "parque_infantil", "trasteros", "paqueteria"]:
            filtros_combinados.append(f"instalaciones:={filtro_inst}")
            
        filtro_extraido = " && ".join(filtros_combinados)
        variantes_limpias = [str(v).strip(' -*"\'') for v in variantes if len(str(v).strip()) > 3]
        
        lista_final = [pregunta_limpia]
        if not filtro_inst:
            for v in variantes_limpias:
                if v not in lista_final:
                    lista_final.append(v)
                
        return lista_final, filtro_extraido
        
    except Exception as e:
        print(f"Error al procesar JSON de Llama 3: {e}")
        return [pregunta_original], ""

# 5. Búsqueda Híbrida 
def buscar_en_typesense(variantes, filtro_opcional=None):
    print(f"[2] Realizando Búsqueda Híbrida simultánea de {len(variantes)} variantes...")
    resultados_unicos = {}
    peticiones_busqueda = []
    
    for variante in variantes:
        vector = modelo_emb.encode(variante).tolist()
        vector_str = ",".join(map(str, vector))
        
        peticion = {
            'collection': 'comunidad_vecinos',
            'q': variante,
            'query_by': 'texto',
            'vector_query': f'embedding:([{vector_str}], k: 10, alpha: 0.6)', 
            'per_page': 5
        }
        
        if filtro_opcional:
            peticion['filter_by'] = filtro_opcional
            
        peticiones_busqueda.append(peticion)
        
    resultados_brutos = cliente_db.multi_search.perform({'searches': peticiones_busqueda}, {})
    
    for res in resultados_brutos['results']:
        if 'hits' in res:
            for hit in res['hits']:
                doc = hit['document']
                id_doc = doc.get('id', doc.get('texto', '')[:20])
                
                if id_doc not in resultados_unicos:
                    origen = doc.get('archivo_origen', 'Acta desconocida')
                    año = doc.get('año', 'Sin año')
                    # MAGIA: Guardamos un diccionario en lugar de un string para mantener el ID
                    resultados_unicos[id_doc] = {
                        'id': id_doc,
                        'origen': origen,
                        'año': año,
                        'texto': doc['texto']
                    }
                    
    # Devolvemos la lista de diccionarios
    return list(resultados_unicos.values())

# 6. Generación de Respuesta Final (Sin bucles infinitos)
def generar_respuesta_final(pregunta, contexto_lista):
    print(f"[3] Enviando contexto unificado ({len(contexto_lista)} fragmentos) a Llama 3 para el resumen final...\n")
    
    if len(contexto_lista) == 0:
        print("❌ ADVERTENCIA: La base de datos no ha devuelto ningún fragmento de texto.")
        return

    # Formatear los diccionarios con presentación mejorada del ID
    bloques_formateados = []
    fuentes_unicas = set()
    
    for ctx_dict in contexto_lista:
        id_doc = ctx_dict.get('id', 'Sin ID')
        origen = ctx_dict.get('origen', 'Acta desconocida')
        año = ctx_dict.get('año', 'Sin año')
        instalaciones = ctx_dict.get('instalaciones', [])
        texto = ctx_dict.get('texto', '')
        
        bloque_formateado = f"[ID: {id_doc} | Fuente: {origen} - Año {año}]\n{texto}"
        bloques_formateados.append(bloque_formateado)
        fuentes_unicas.add(f"{origen} - Año {año}")

    bloque_contexto = "\n\n".join(bloques_formateados)

    prompt_final = f"""Eres un administrador de fincas experto. Responde a la pregunta basándote ÚNICAMENTE en el contexto proporcionado.

REGLAS ESTRICTAS:

Responde de forma clara y directa utilizando solo los datos de los textos.

IMPORTANTE: Los textos proporcionados tienen fecha de 2003 y 2004. Comprende que ambos años están dentro de la franja solicitada (2000 a 2008), por lo que DEBES extraer la información. No digas que no hay datos.

Presenta la información en un formato de lista con viñetas (*), resumiendo cada norma o mención de forma clara.

NO incluyas ninguna lista de "Fuentes consultadas" o "Archivos" al final de tu respuesta.


    Contexto extraído de las actas:
    {bloque_contexto}
    
    Pregunta: {pregunta}
    """
    
    payload = {
        "model": MODELO_LLM, 
        "prompt": prompt_final, 
        "stream": True,
        "options": {"temperature": 0.1}
    }
    print (f"\n[Prompt enviado a Llama 3 para generación de respuesta final:]\n{prompt_final}\n")
    print("--- RESPUESTA OFICIAL ---")
    try:
        respuesta_stream = requests.post(URL_OLLAMA, json=payload, stream=True)
        for linea in respuesta_stream.iter_lines():
            if linea:
                datos = json.loads(linea)
                print(datos.get('response', ''), end='', flush=True)
    except Exception as e:
        print(f"\n[Error de conexión con Ollama: {e}]")
        
    print("\n\n[📖 Fragmentos consultados por el motor de búsqueda:]")
    for ctx_dict in contexto_lista:
        id_doc = ctx_dict.get('id', 'Sin ID')
        origen = ctx_dict.get('origen', 'Acta desconocida')
        año = ctx_dict.get('año', 'Sin año')
        print(f" - ID: {id_doc} | {origen} - Año {año}")
    print("-------------------------")

if __name__ == "__main__":
    #pregunta_usuario = "¿A qué empresa se contrató para sustituir el servicio de conserjería entre 2000 y 2008  y  por qué?"
    pregunta_usuario = "¿Qué se ha mencionado sobre la piscina entre 2000 y 2008?"
    lista, filtro = analizar_pregunta_con_ia(pregunta_usuario)
    for i, v in enumerate(lista):
        print(f"  - Variante ia {i}: '{v}'")
    print(f"  - Filtro extraído por IA: '{filtro}'")
      
    # 3. Pasamos el filtro opcional a la función de búsqueda
    print(f"\n🔍 Realizando búsqueda híbrida con el filtro: '{filtro}'...")
    fragmentos = buscar_en_typesense(lista, filtro_opcional=filtro)
    print(f"  - fragmentos crudos recuperados: {len(fragmentos)}")
    pregunta_ampliada_con_variantes = pregunta_usuario + " o " + " ".join(lista)  # string con la pregunta original + variantes (para que Llama 3 tenga más contexto)
    generar_respuesta_final(pregunta_ampliada_con_variantes, fragmentos)