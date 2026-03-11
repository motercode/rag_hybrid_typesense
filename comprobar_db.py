import typesense
cliente_db = typesense.Client({
  'nodes': [{'host': 'localhost', 'port': '8108', 'protocol': 'http'}],
  'api_key': 'clave_secreta_actas',
  'connection_timeout_seconds': 2
})
info = cliente_db.collections['comunidad_vecinos'].retrieve()
print(f"Total de fragmentos guardados: {info['num_documents']}")