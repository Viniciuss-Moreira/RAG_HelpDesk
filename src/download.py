import nltk

print("Baixando o recurso 'punkt' do NLTK...")
# Nota: O erro menciona 'punkt_tab', mas o pacote correto a ser baixado é 'punkt'.
nltk.download('punkt_tab') 
print("Download concluído com sucesso!")