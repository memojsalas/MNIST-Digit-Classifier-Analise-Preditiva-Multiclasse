### Analise Preditiva com MNIST / Classificação de Dígitos Manuscritos

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-green.svg)
![Keras](https://img.shields.io/badge/Keras-2.13+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Problema Resolvido](#-problema-resolvido)
- [Tecnologias e Técnicas](#-tecnologias-e-técnicas-utilizadas)
- [Arquitetura do Pipeline](#-arquitetura-do-pipeline)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação e Execução](#-instalação-e-execução)
- [Resultados e Insights](#-resultados-e-insights)
- [Análise de Robustez](#-análise-de-robustez)
- [Melhorias Futuras](#-melhorias-futuras)
- [Como Contribuir](#-como-contribuir)
- [Licença](#-licença)
- [Autor](#-autor)
- [Referências](#-referências)
- [Agradecimentos](#-agradecimentos)

---

## 📖 Sobre o Projeto

Este projeto implementa um pipeline completo de **Ciência de Dados** para classificação de dígitos manuscritos utilizando o dataset **MNIST** (Modified National Institute of Standards and Technology). O objetivo principal é demonstrar e comparar diferentes abordagens de Machine Learning, desde algoritmos clássicos até redes neurais profundas, além de testar a robustez dos modelos em cenários adversos e com dados do mundo real.

> **🎯 Nota**: Este projeto foi desenvolvido como parte de um Mini-Projeto Avaliativo, simulando o papel de um **Cientista de Dados** e **Engenheiro de Machine Learning** em um ambiente profissional.

### ✨ Principais Características

- ✅ **Pipeline End-to-End**: Desde o carregamento dos dados até a inferência final
- ✅ **3 Modelos Contrastantes**: Random Forest, KNN e MLP (Rede Neural)
- ✅ **Análise Exploratória Completa**: Visualizações e interpretações detalhadas
- ✅ **Testes de Robustez**: Class Masking e Inferência OOD
- ✅ **Integração com Imagens Reais**: Processamento de dígitos manuscritos próprios
- ✅ **Avaliação Multiclasse Completa**: Matrizes de confusão, métricas detalhadas
- ✅ **Documentação Profissional**: README completo, código comentado, requirements.txt

---

## 🎯 Problema Resolvido

A classificação de dígitos manuscritos é um problema clássico da Visão Computacional que serve como **benchmark** universal para avaliar algoritmos de aprendizado de máquina. Este projeto aborda:

### Problemas Técnicos

1. **Classificação Multiclasse**: Identificar corretamente 10 classes (0-9) em imagens de 28x28 pixels
2. **Comparação de Abordagens**: Avaliar trade-offs entre modelos clássicos e deep learning
3. **Robustez**: Testar modelos sob condições adversas (dados fora da distribuição)
4. **Generalização**: Validar com imagens reais digitalizadas pelos próprios alunos

### Aplicações Práticas

- 🔍 **Sistemas de Reconhecimento**: Leitura de cheques, códigos postais
- 📱 **OCR (Optical Character Recognition)**: Digitalização de documentos
- 🏦 **Automação Bancária**: Processamento de formulários
- 📊 **Benchmark Acadêmico**: Referência para novos algoritmos

---

## 🚀 Tecnologias e Técnicas Utilizadas

### Modelos Implementados

| Modelo | Tipo | Características |
|--------|------|-----------------|
| **Random Forest** | Ensemble | 100 árvores, max_depth=20, min_samples_split=10 |
| **KNN** | Instance-based | k=5, weights='distance', distância Euclidiana |
| **MLP** | Rede Neural | 3 camadas ocultas (256, 128, 64), Dropout, Adam |

### Técnicas de Machine Learning

- **Pré-processamento**: Normalização [0,1], StandardScaler
- **Divisão Estratificada**: 70% Treino, 10% Validação, 20% Teste
- **Otimização de Hiperparâmetros**: Ajuste manual justificado
- **Regularização**: Dropout, Early Stopping
- **Avaliação**: Matriz de Confusão, Classification Report

### Ferramentas e Bibliotecas

```python
# Principais Bibliotecas
import numpy as np              # Operações matemáticas
import pandas as pd             # Manipulação de dados
import matplotlib.pyplot as plt # Visualizações
import seaborn as sns           # Estatísticas visuais

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Processamento de Imagens
from PIL import Image, ImageOps
