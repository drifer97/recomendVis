import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle

# Função para carregar imagens e processá-las
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Pré-processamento para MobileNetV2
    return np.expand_dims(img_array, axis=0)

# Carregar MobileNetV2 pré-treinado
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Pasta de imagens e arquivo de embeddings
image_folder = "images"
embeddings_file = "embeddings.pkl"

# Função para calcular ou carregar embeddings
def get_embeddings(image_folder, embeddings_file):
    if os.path.exists(embeddings_file):
        # Carregar embeddings salvos
        with open(embeddings_file, 'rb') as f:
            embeddings, image_paths = pickle.load(f)
    else:
        # Extrair embeddings e salvar
        image_paths = [
            os.path.join(image_folder, img)
            for img in os.listdir(image_folder)
            if img.endswith(('.jpg', '.png'))
        ]

        embeddings = []
        for path in image_paths:
            img = load_and_preprocess_image(path)
            embedding = model.predict(img)  # Vetores de características
            embeddings.append(embedding[0])

        embeddings = np.array(embeddings)

        # Salvar embeddings
        with open(embeddings_file, 'wb') as f:
            pickle.dump((embeddings, image_paths), f)

    return embeddings, image_paths

# Carregar ou calcular embeddings
embeddings, image_paths = get_embeddings(image_folder, embeddings_file)

# Função para encontrar imagens mais similares
def find_similar_images(query_image_path, image_paths, embeddings, top_n=5):
    # Extrair embedding da imagem de consulta
    query_img = load_and_preprocess_image(query_image_path)
    query_embedding = model.predict(query_img)

    # Calcular similaridade
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1][:top_n]  # Top N similares

    return sorted_indices, similarities

# Consulta com imagem externa ou do dataset
def query_image(query_image_path=None):
    if query_image_path is None:
        query_image_path = image_paths[0]  # Usar a primeira imagem do dataset como padrão

    similar_indices, similarities = find_similar_images(query_image_path, image_paths, embeddings)

    # Visualizar resultados
    plt.figure(figsize=(15, 5))

    # Mostrar imagem de consulta
    plt.subplot(1, len(similar_indices) + 1, 1)
    plt.imshow(load_img(query_image_path))
    plt.title("Consulta")
    plt.axis("off")

    # Mostrar imagens recomendadas
    for i, idx in enumerate(similar_indices):
        plt.subplot(1, len(similar_indices) + 1, i + 2)
        plt.imshow(load_img(image_paths[idx]))
        plt.title(f"Similaridade: {similarities[idx]:.2f}")
        plt.axis("off")

    plt.show()

# Exemplo de uso
# Use uma imagem do dataset ou forneça o caminho para uma imagem externa
external_image_path = "E:\\recomendação\\fashion-dataset\corinthians.jpg"  # Substitua pelo caminho da sua imagem
if os.path.exists(external_image_path):
    query_image(external_image_path)
else:
    query_image()
