import sys
import typesense

# Configuración de Conexión a Typesense
cliente_db = typesense.Client({
  'nodes': [{'host': 'localhost', 'port': '8108', 'protocol': 'http'}],
  'api_key': 'clave_secreta_actas',
  'connection_timeout_seconds': 2
})

def obtener_contexto_ampliado(id_doc_encontrado):
    """
    Toma un ID (ej: 'hash_5') y recupera de Typesense los fragmentos 'hash_4', 'hash_5' y 'hash_6'.
    """
    print(f"\n[+] Conectando a Typesense para extraer el contexto del fragmento: {id_doc_encontrado}...\n")
    
    try:
        hash_archivo, str_indice = id_doc_encontrado.rsplit('_', 1)
        indice_actual = int(str_indice)
    except ValueError:
        return "❌ Error: El ID del documento no tiene el formato esperado (hash_indice)."

    id_anterior = f"{hash_archivo}_{indice_actual - 1}"
    id_posterior = f"{hash_archivo}_{indice_actual + 1}"
    
    ids_a_buscar = [id_anterior, id_doc_encontrado, id_posterior]
    
    try:
        respuesta = cliente_db.collections['comunidad_vecinos'].documents.search({
            'q': '*',
            'filter_by': f"id:=[{','.join(ids_a_buscar)}]",
            'per_page': 3
        })
        
        documentos_recuperados = respuesta.get('hits', [])
        
        if not documentos_recuperados:
            return "❌ No se encontró ningún documento con ese ID en la base de datos."
            
        # Ordenamos basándonos en el índice final del ID (4, 5, 6)
        documentos_recuperados.sort(key=lambda hit: int(hit['document']['id'].rsplit('_', 1)[1]))
        
        texto_ampliado = ""
        for hit in documentos_recuperados:
            doc = hit['document']
            if doc['id'] == id_anterior:
                texto_ampliado += f"--- [FRAGMENTO ANTERIOR (ID: {doc['id']})] ---\n{doc['texto']}\n\n"
            elif doc['id'] == id_doc_encontrado:
                texto_ampliado += f"--- [FRAGMENTO ENCONTRADO (ID: {doc['id']})] ---\n{doc['texto']}\n\n"
            elif doc['id'] == id_posterior:
                texto_ampliado += f"--- [FRAGMENTO POSTERIOR (ID: {doc['id']})] ---\n{doc['texto']}\n\n"
                
        return texto_ampliado

    except Exception as e:
        return f"❌ Error de conexión con la base de datos: {e}"

if __name__ == "__main__":
    # Verificamos que el usuario ha pasado el ID por consola
    if len(sys.argv) < 2:
        print("Uso correcto: python extractor_de_contexto.py <ID_DEL_DOCUMENTO>")
        print("Ejemplo: python extractor_de_contexto.py d490adf4da4bb09151_24")
        sys.exit(1)

    id_buscar = sys.argv[1]
    resultado = obtener_contexto_ampliado(id_buscar)
    print(resultado)