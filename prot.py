import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Criação de Perfis de Usuários
users = {
    1: {'extroversao': 4, 'curiosidade': 5, 'visual': 1, 'verbal': 0, 'pratico': 0, 'analitico': 1},
    2: {'extroversao': 2, 'curiosidade': 3, 'visual': 0, 'verbal': 1, 'pratico': 1, 'analitico': 0},
    3: {'extroversao': 5, 'curiosidade': 4, 'visual': 1, 'verbal': 1, 'pratico': 0, 'analitico': 1},
    4: {'extroversao': 1, 'curiosidade': 2, 'visual': 0, 'verbal': 1, 'pratico': 1, 'analitico': 0},
    5: {'extroversao': 3, 'curiosidade': 5, 'visual': 1, 'verbal': 0, 'pratico': 1, 'analitico': 1},
    6: {'extroversao': 2, 'curiosidade': 1, 'visual': 0, 'verbal': 1, 'pratico': 0, 'analitico': 1},
    7: {'extroversao': 4, 'curiosidade': 3, 'visual': 1, 'verbal': 1, 'pratico': 1, 'analitico': 0},
    8: {'extroversao': 1, 'curiosidade': 4, 'visual': 0, 'verbal': 1, 'pratico': 0, 'analitico': 1},
    9: {'extroversao': 5, 'curiosidade': 5, 'visual': 1, 'verbal': 0, 'pratico': 1, 'analitico': 0},
    10: {'extroversao': 3, 'curiosidade': 2, 'visual': 0, 'verbal': 1, 'pratico': 1, 'analitico': 1}
}

# Criação de Perfis de Vídeos
videos = {
    1: {'highlight': 1, 'analise': 0, 'entrevista': 0, 'duracao': 5, 'visual': 1, 'verbal': 0},
    2: {'highlight': 0, 'analise': 1, 'entrevista': 0, 'duracao': 15, 'visual': 0, 'verbal': 1},
    3: {'highlight': 0, 'analise': 0, 'entrevista': 1, 'duracao': 9, 'visual': 1, 'verbal': 1},
    4: {'highlight': 1, 'analise': 0, 'entrevista': 0, 'duracao': 10, 'visual': 1, 'verbal': 1},
    5: {'highlight': 0, 'analise': 1, 'entrevista': 0, 'duracao': 20, 'visual': 1, 'verbal': 1},
    6: {'highlight': 0, 'analise': 0, 'entrevista': 1, 'duracao': 12, 'visual': 0, 'verbal': 1},
    7: {'highlight': 1, 'analise': 0, 'entrevista': 0, 'duracao': 7, 'visual': 1, 'verbal': 1},
    8: {'highlight': 0, 'analise': 1, 'entrevista': 0, 'duracao': 8, 'visual': 0, 'verbal': 1},
    9: {'highlight': 0, 'analise': 0, 'entrevista': 1, 'duracao': 16, 'visual': 1, 'verbal': 1},
    10: {'highlight': 0, 'analise': 1, 'entrevista': 0, 'duracao': 18, 'visual': 1, 'verbal': 0}
}

# Histórico de Interações dos Usuários 
matriz_interecao = np.array([
#              Vídeos
#    1  2  3  4  5  6  7  8  9  10    
    [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],  # Arthur (Usuário 1)
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],  # Agatha (Usuário 2)
    [1, 0, 1, 1, 1, 1, 1, 0, 0, 0],  # José (Usuário 3)
    [0, 1, 0, 0, 0, 1, 0, 1, 1, 1],  # Vitoria (Usuário 4)
    [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],  # Henrique (Usuário 5)
    [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],  # Maria (Usuário 6)
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],  # Lucas (Usuário 7)
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # Laura (Usuário 8)
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 1],  # Matheus (Usuário 9)
    [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],  # Sofia (Usuário 10)
])

# Função para recomendação por Filtragem Colaborativa
def recomendacao_fc(user_id, matriz_interecao, top_n=2):
    user_similares = cosine_similarity(matriz_interecao)

    user_pontuacao_similaridade = user_similares[user_id - 1]

    pontuacao_ponderada = np.dot(user_pontuacao_similaridade, matriz_interecao)

    user_interacao = matriz_interecao[user_id - 1]

    pontuacao_ponderada = pontuacao_ponderada * (user_interacao == 0)

    videos_recomendados = np.argsort(pontuacao_ponderada)[::-1][:top_n] + 1
   
    return videos_recomendados

# Função para recomendação por Filtragem Baseada em Conteúdo
def recomendacao_fbc(user_id, users, videos, top_n=2):
    user_atributos = np.array(list(users[user_id].values())).reshape(1, -1)
   
    video_atributos = np.array([list(videos[vid].values()) for vid in videos])
   
    video_similarities = cosine_similarity(user_atributos, video_atributos).flatten()
   
    videos_recomendados = np.argsort(video_similarities)[::-1][:top_n] + 1
  
    return videos_recomendados

# Função para recomendação híbrida, que combina Filtragem Colaborativa e Filtragem Baseada em Conteúdo
def recomendacao_hibrida(user_id, matriz_interecao, users, videos, alpha=0.5, top_n=2):
    fc_recomendacao = recomendacao_fc(user_id, matriz_interecao, top_n=top_n)
  
    fbc_recomendacao = recomendacao_fbc(user_id, users, videos, top_n=top_n)
 
    pontuacoes_finais = {}
    
    for vid in fc_recomendacao:
        pontuacoes_finais[vid] = alpha
    
    for vid in fbc_recomendacao:
        if vid in pontuacoes_finais:
            pontuacoes_finais[vid] += (1 - alpha)
        else:
            pontuacoes_finais[vid] = (1 - alpha)
    
    recomendacao_final = sorted(pontuacoes_finais, key=pontuacoes_finais.get, reverse=True)[:top_n]
    
    return recomendacao_final

# Gera as recomendações para cada usuário
for user_id in users.keys():
    recommendations = recomendacao_hibrida(user_id, matriz_interecao, users, videos, top_n=3)
    
    recommendations = list(map(int, recommendations))
    
    print(f"Recomendações para o usuário {user_id}: {recommendations}")
