# -*- coding: utf-8 -*-
"""analise-deputados.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C7x3XgB_C8mZx2s56_R8fYzvBolASxOD
"""

# Artigo base: https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
# Objetivo: fazer download automatico da tabela dos gastos dos deputados
# disponivel em http://meucongressonacional.com/deputado e elencar os 10
# deputados com maior media de gasto diario.

import requests
import lxml.html as lh
import pandas as pd

url = "http://meucongressonacional.com/deputado"

# download da pagina
page = requests.get(url)
doc = lh.fromstring(page.content)

# pegar elementos entre tags HTML tr
tr_elements = doc.xpath("//tr")

# Os primeiros 650 itens (cabecalho = tr_elements[0] + 649 deputados) sao a tabela principal.
# Os outros sao da lista do top 5 de gastos, que vamos ignorar por agora.
# Ultimo elemento: Zonta
print(tr_elements[649].text_content())

dados_deputados = tr_elements[:650].copy()

# inspecionando o padrao do texto
dados_deputados[0].text_content().split("\t")

# filtrando os casos de strings vazias apos o split acima
[s.replace('\n', '') for s in dados_deputados[0].text_content().split('\t') if s not in ('', '\n')]

# testando para a primeira linha apos o cabecalho: funcionando!!
[s.replace('\n', '') for s in dados_deputados[1].text_content().split('\t') if s not in ('', '\n')]

# script para coletar os dados de cada linha criando uma tabela com isso
colunas = [s.replace('\n', '') for s in tabela_deputados[0].text_content().split('\t') if s not in ('', '\n')]
# corrigindo o nome da ultima coluna de interesse
colunas[3] = colunas[3].replace('Gastos', '')

# aplicando o padrao que descobrimos em cada linha
tabela = []
for linha in tabela_deputados[1:]:
  valores = [s.replace('\n', '') for s in linha.text_content().split('\t') if s not in ('', '\n')]
  linha_dict = dict()
  for coluna, valor in zip(colunas, valores):
    if coluna == 'R$/dia':
      valor = float(valor.replace(',', ''))
    linha_dict[coluna] = valor
  tabela.append(linha_dict)

df = pd.DataFrame(tabela)

df.head()

# obter o top 10 com respeito a gastos por dia:
df.sort_values('R$/dia', ascending=False).head(10)