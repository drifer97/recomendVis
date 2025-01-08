Sistema de Recomendação Visual com MobileNetV2

Este projeto implementa um sistema de recomendação visual que encontra imagens similares a uma imagem de consulta em um conjunto de imagens. Utiliza o modelo pré-treinado MobileNetV2 para extrair embeddings de características e compara as imagens usando similaridade do cosseno.

Funcionalidades

Extração de Embeddings: Extrai vetores de características das imagens usando MobileNetV2.

Persistência de Dados: Salva os embeddings calculados em um arquivo para evitar recálculos.

Busca de Similaridade: Encontra as imagens mais similares a uma imagem de consulta.

Visualização dos Resultados: Mostra a imagem de consulta e as imagens mais similares com seus níveis de similaridade.



Requisitos

Python 3.7 ou superior

Bibliotecas:

tensorflow

numpy

matplotlib

scikit-learn

pickle (nativo do Python)



Como Usar

Coloque as imagens do conjunto de dados na pasta images/.

Execute o script principal:

python main.py

O script processará as imagens, calculará os embeddings (se não existentes) e exibirá as imagens mais similares à imagem de consulta.

Consulta com Imagem Externa

Para consultar uma imagem que não está no conjunto de dados, edite a variável external_image_path no script principal:

external_image_path = "caminho/para/sua/imagem.jpg"

Dataset Utilizado

Exemplo de Saída

Após executar o script, uma janela gráfica será exibida mostrando:

A imagem de consulta

As imagens mais similares do conjunto, acompanhadas de seus níveis de similaridade.
