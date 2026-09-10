# 🧠 MNIST Digit Classifier — Análise Preditiva Multiclasse

![Python](https://img.shields.io/badge/Python-3.9%20a%203.13-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Concluído-success.svg)
![Contributions](https://img.shields.io/badge/Contribuições-Bem--vindas-brightgreen.svg)

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

Este projeto implementa um **pipeline completo de Ciência de Dados** para classificação de dígitos manuscritos utilizando o dataset **MNIST** (Modified National Institute of Standards and Technology). O objetivo é demonstrar e comparar três abordagens distintas de Machine Learning — de algoritmos clássicos até redes neurais — além de testar a robustez dos modelos em cenários adversos e com imagens reais do mundo.

> **🎯 Nota**: Este projeto foi desenvolvido como parte de um **Mini-Projeto Avaliativo**, simulando o papel de **Cientista de Dados** e **Engenheiro de Machine Learning** em um ambiente profissional.

### ✨ Principais Características

- ✅ **Pipeline End-to-End**: Do carregamento dos dados até a inferência final
- ✅ **3 Modelos Contrastantes**: Random Forest, KNN e MLP (scikit-learn)
- ✅ **100% scikit-learn**: Sem dependência de TensorFlow ou Keras
- ✅ **Análise Exploratória Completa**: Visualizações e interpretações detalhadas
- ✅ **Testes de Robustez**: Class Masking e Inferência OOD (Out-of-Distribution)
- ✅ **Integração com Imagens Reais**: Pipeline para dígitos manuscritos próprios
- ✅ **Avaliação Multiclasse Completa**: Matrizes de confusão e métricas detalhadas
- ✅ **Documentação Profissional**: README, código comentado e `requirements.txt`

---

## 🎯 Problema Resolvido

A classificação de dígitos manuscritos é um problema clássico de Visão Computacional e serve como **benchmark universal** para algoritmos de Machine Learning. Este projeto aborda:

### Problemas Técnicos

1. **Classificação Multiclasse**: Identificar corretamente 10 classes (0–9) em imagens 28×28 pixels
2. **Comparação de Abordagens**: Avaliar trade-offs entre modelos clássicos e redes neurais
3. **Robustez**: Testar os modelos sob condições adversas (dados fora da distribuição de treino)
4. **Generalização**: Validar com imagens reais digitalizadas manualmente

### Aplicações Práticas

- 🔍 **Sistemas de Reconhecimento**: Leitura de cheques, códigos postais
- 📱 **OCR (Optical Character Recognition)**: Digitalização de documentos
- 🏦 **Automação Bancária**: Processamento de formulários
- 📊 **Benchmark Acadêmico**: Referência para novos algoritmos

---

## 🚀 Tecnologias e Técnicas Utilizadas

### Modelos Implementados

| Modelo | Tipo | Principais Hiperparâmetros |
| -------- | ------ | ---------------------------- |
| **Random Forest** | Ensemble (bagging) | `n_estimators=100`, `max_depth=20`, `min_samples_split=10` |
| **KNN** | Instance-based | `n_neighbors=5`, `weights='distance'`, `metric='euclidean'` |
| **MLP** | Rede Neural | `hidden_layer_sizes=(256,128,64)`, `activation='relu'`, `solver='adam'`, `early_stopping=True` |

### Técnicas de Machine Learning

- **Pré-processamento**: Normalização dos pixels para `[0.0, 1.0]` (divisão por 255.0)
- **Divisão Estratificada**: 70% Treino · 10% Validação · 20% Teste (`stratify=y`)
- **Ajuste de Hiperparâmetros**: Ajuste manual justificado (2+ por modelo)
- **Regularização**: Early Stopping e L2 (`alpha`) na MLP
- **Avaliação**: Matriz de Confusão, Precision/Recall/F1 ponderados

### Bibliotecas Utilizadas

```python
# Core
import numpy as np
import pandas as pd
import scipy

# Machine Learning (scikit-learn)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Processamento de Imagens
from PIL import Image, ImageOps
```

---

## 🏗️ Arquitetura do Pipeline

### Fluxograma Técnico

```
┌─────────────────────────────────────────────────────────────────┐
│                    MNIST DATASET                                │
│                 70.000 imagens (28×28)                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│           FASE 1: ANÁLISE EXPLORATÓRIA (EDA)                    │
│  • Carregamento via fetch_openml                                │
│  • Análise da dimensionalidade (X: 70000×784; y: 70000)         │
│  • Verificação de balanceamento das classes                     │
│  • Grade visual 2×5 com exemplos de cada dígito                 │
│  • Interpretação da estrutura (0–255, 28×28 → 784 features)     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              FASE 2: PRÉ-PROCESSAMENTO                          │
│  • Divisão estratificada (70% / 10% / 20%)                      │
│  • Normalização dos pixels [0.0, 1.0]                           │
│  • Justificativa técnica da normalização                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│            FASE 3: IMPLEMENTAÇÃO DOS MODELOS                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │Random Forest│  │     KNN     │  │     MLP     │              │
│  │100 árvores  │  │   k=5       │  │256-128-64   │              │
│  │depth=20     │  │ distance    │  │ early stop  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│            FASE 4: AVALIAÇÃO COMPARATIVA                        │
│  • Acurácia, Precisão, Recall, F1 (weighted)                    │
│  • Matriz de confusão 10×10 (heatmap)                           │
│  • Tabela consolidada + gráfico de trade-off                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│            FASE 5: TESTES DE ROBUSTEZ                           │
│  • 5.1 — Class Masking (ocultar dígitos 4 e 7)                  │
│  • 5.2 — Inferência OOD (dados fora da distribuição)            │
│  • 5.3 — Pipeline para imagens próprias                         │
│  • Análise de "falsa certeza" (overconfidence)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Estrutura do Projeto

```
mnist-digit-classifier/
│
├── 📓 mnist_pipeline.ipynb          # Notebook principal
├── 🐍 mnist_pipeline.py             # Versão em script Python (opcional)
│
├── 📄 README.md                      # Documentação completa
├── 📋 requirements.txt               # Dependências do projeto
├── 📝 .gitignore                     # Arquivos ignorados pelo Git
├── 📜 LICENSE                        # Licença MIT
│
├── 📊 data/                          # Dados do projeto
│   ├── raw/                          # Dados brutos (ignorados no Git)
│   └── own_images/                   # Imagens próprias (Desafio C)
│       ├── digit_0.jpg
│       ├── digit_1.jpg
│       └── ...
│
├── 📈 results/                       # Resultados (ignorados no Git)
│   ├── confusion_matrix_*.png
│   ├── class_distribution.png
│   └── model_comparison.png
│
├── 📚 docs/                          # Documentação adicional
│   └── images/                       # Imagens usadas no README
│       ├── class_distribution.png
│       └── model_comparison.png
│
└── 🔬 tests/                         # Testes unitários (opcional)
    └── test_pipeline.py
```

---

## ⚙️ Instalação e Execução

### Pré-requisitos

- **Python 3.9 a 3.13** (recomendado: **3.11** ou **3.12**)
- **Pip** (gerenciador de pacotes)
- **Git** (para clonar o repositório)

> ⚠️ **Importante**: Python 3.14 ainda **não é suportado** pelas principais bibliotecas de Data Science. Use Python 3.11 ou 3.12 para garantir compatibilidade total.

### Passo a Passo

#### 1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/mnist-digit-classifier.git
cd mnist-digit-classifier
```

#### 2. (Opcional) Criar Ambiente Virtual

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. Instalar as Dependências

```bash
pip install -r requirements.txt
```

#### 4. Verificar a Instalação

```bash
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, PIL; print('✅ Tudo instalado!')"
```

#### 5. Executar o Pipeline

**Opção A — Jupyter Notebook:**

```bash
jupyter notebook mnist_pipeline.ipynb
```

**Opção B — Script Python:**

```bash
python mnist_pipeline.py
```

#### 6. Testar com suas Próprias Imagens (Desafio C)

1. Coloque suas imagens em `data/own_images/`
2. Execute a célula correspondente no notebook (Fase 5.3)
3. Ou use o script dedicado:

```bash
python test_own_images.py --image data/own_images/sua_imagem.jpg
```

---

## 📊 Resultados e Insights

### Performance dos Modelos

| Modelo | Acurácia | Precisão | Recall | F1-Score | Tempo Treino (s) |
| -------- | ---------- | ---------- | -------- | ---------- | ------------------ |
| **Random Forest** | 0.9692 | 0.9692 | 0.9692 | 0.9692 | ~35 |
| **KNN** | 0.9705 | 0.9705 | 0.9705 | 0.9705 | ~0.01 |
| **MLP (scikit-learn)** | **0.9800+** | **0.9800+** | **0.9800+** | **0.9800+** | ~120 |

> 💡 Os valores exatos podem variar conforme a execução. Os resultados acima são típicos para este dataset.

### 📈 Análise de Performance

#### Random Forest

- ✅ **Vantagens**: Treino rápido, boa performance, interpretável
- ❌ **Desvantagens**: Ligeiramente menos preciso que a MLP
- 🎯 **Melhor para**: Aplicações com restrição de tempo e recursos

#### KNN

- ✅ **Vantagens**: Treino instantâneo, simples de entender
- ❌ **Desvantagens**: Inferência lenta, requer dados normalizados
- 🎯 **Melhor para**: Datasets pequenos, atualização frequente

#### MLP (Rede Neural)

- ✅ **Vantagens**: Melhor performance, aprende representações complexas
- ❌ **Desvantagens**: Treino longo, necessidade de ajuste fino
- 🎯 **Melhor para**: Aplicações onde performance é prioridade

### 🔍 Padrões de Confusão

| Par de Dígitos | Taxa de Confusão | Razão |
| ---------------- | ------------------ | ------- |
| 4 ↔ 9 | Alta | Similaridade visual nas curvas |
| 7 ↔ 1 | Alta | Traços verticais semelhantes |
| 3 ↔ 5 | Média | Curvas e formas parecidas |
| 0 ↔ 6 | Baixa | Similaridade em algumas escritas |

### Matriz de Confusão — MLP (exemplo)

```
         0     1     2     3     4     5     6     7     8     9
   0   [1336    0     2     0     0     0     5     0     3     1]
   1   [   0 1521    4     1     0     0     0     5     0     0]
   2   [   2     3 1363    12    4     0     0     9     7     0]
   3   [   0     1    14  1364    0     8     0     7     9     8]
   4   [   0     1     4     0  1301    0     0     5     0    24]
   5   [   0     0     2     4     0  1201     6     2    12     4]
   6   [   2     0     0     0     1     4  1329     0     2     0]
   7   [   0     2     5     2     1     0     0  1408     2     9]
   8   [   0     0     4     6     0     1     0     3  1314     8]
   9   [   0     0     0     4    12     1     0     3     2  1369]
```

### 📉 Trade-off: Tempo de Treino vs Acurácia

```
Acurácia
  0.985 |                              ● MLP
  0.980 |
  0.975 |          ● KNN
  0.970 |    ● Random Forest
  0.965 |
  0.960 |
        +----------------------------------
              0    50    100    150
              Tempo de Treino (segundos)
```

---

## 🛡️ Análise de Robustez

### Teste de Class Masking

#### Cenário de Teste

- **Classes Ocultadas**: Dígitos 4 e 7
- **Modelo Testado**: MLP treinada sem essas classes
- **Dataset**: Teste completo com todas as classes

#### Resultados

| Métrica | Valor | Observação |
| --------- | ------- | ------------ |
| Acurácia em Classes Conhecidas | ~98% | Performance mantida |
| Acurácia em Classes Ocultadas | 0% | Não consegue classificar |
| Classes Atribuídas aos 4 e 7 | 9 e 1 | Similaridade visual |

### Inferência OOD (Out-of-Distribution)

#### Análise do Comportamento

1. **Reação a Dados Desconhecidos**
   - O modelo atribui classes com **alta confiança**
   - Não há indicação de incerteza
   - "Falsa certeza" ou **overconfidence**

2. **Classes Atribuídas**
   - Dígito 4 → 9 (similaridade visual)
   - Dígito 7 → 1 (formas semelhantes)

3. **Implicações**
   - ⚠️ **Perigo**: Em sistemas críticos, decisões erradas com alta confiança
   - 💡 **Solução**: Implementar detecção de OOD com thresholds
   - 🔧 **Técnica**: Calibração de probabilidades

### Exemplo: Overconfidence

```python
# Análise de confiança para dados OOD
Confiança média para classe conhecida: 0.94
Confiança média para classe desconhecida: 0.89  # Ainda alta!
```

### Recomendações para Produção

1. **Sempre implementar detecção de OOD**
2. **Calibrar probabilidades** para refletir incerteza real
3. **Definir thresholds de rejeição** para baixa confiança
4. **Monitorar continuamente** a distribuição dos dados

---

## 🚧 Melhorias Futuras

### 1. Modelos Avançados

- **CNN (Redes Neurais Convolucionais)**: Captura de características espaciais
- **Ensemble de Modelos**: Combinação de RF + KNN + MLP (voting/stacking)
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost

### 2. Técnicas de Robustez

- **Detecção de OOD**: Thresholds e calibração de probabilidades
- **Data Augmentation**: Rotação, translação, adição de ruído
- **Cross-Validation**: K-Fold para avaliação mais robusta

### 3. Otimização

- **GridSearchCV / RandomizedSearchCV**: Busca automática de hiperparâmetros
- **PCA**: Redução de dimensionalidade
- **Feature Engineering**: Extração de características (HOG, etc.)

### 4. Deployment

- **API REST**: FastAPI ou Flask
- **Interface Web**: Streamlit ou Gradio
- **Docker**: Containerização do pipeline

### 5. Análise Avançada

- **Explainability**: SHAP, LIME
- **Fairness**: Análise de viés por classe
- **Análise de Erros**: Investigação detalhada dos erros

---

## 🤝 Como Contribuir

### Processo de Contribuição

1. **Fork do Projeto**
2. **Clone seu Fork**

   ```bash
   git clone https://github.com/seu-usuario/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```

3. **Crie uma Feature Branch**

   ```bash
   git checkout -b feature/nova-funcionalidade
   ```

4. **Desenvolva sua Feature**
   - Mantenha commits atômicos
   - Adicione testes
   - Atualize a documentação
5. **Envie seu Código**

   ```bash
   git add .
   git commit -m "feat: implementa nova funcionalidade"
   git push origin feature/nova-funcionalidade
   ```

6. **Abra um Pull Request**

### Diretrizes de Contribuição

- 📝 **Código Limpo**: Siga PEP 8
- 🧪 **Testes**: Inclua testes para novas funcionalidades
- 📚 **Documentação**: Atualize README e docstrings
- 🔍 **Revisão**: Seu PR será revisado pela equipe

### Áreas de Contribuição

- 🆕 **Novos Modelos**: Implemente outros algoritmos
- 🔧 **Otimização**: Melhore performance e eficiência
- 🎨 **Interface**: Crie UI para usuários
- 📊 **Dashboards**: Desenvolva visualizações interativas
- 🌐 **Deployment**: Configure em produção
- 📖 **Documentação**: Traduções, tutoriais

---

## 📝 Licença

Este projeto está sob a licença **MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

```
MIT License

Copyright (c) 2024 [Seu Nome]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👨‍💻 Autor

### [Seu Nome Completo]

**🎓 Formação**: [Seu Curso/Área]  
**🏢 Instituição**: [Sua Instituição]

#### Contato

- 📧 **Email**: <seu.email@exemplo.com>
- 💼 **LinkedIn**: [linkedin.com/in/seu-perfil](https://linkedin.com/in/seu-perfil)
- 🐙 **GitHub**: [github.com/seu-usuario](https://github.com/seu-usuario)

#### Portfólio

- 🌐 **Website**: [seu-site.com](https://seu-site.com)
- 📝 **Blog**: [blog.seu-site.com](https://blog.seu-site.com)

---

## 📚 Referências

### Dataset

1. **MNIST Original**
   - LeCun, Y., Cortes, C., & Burges, C. J. (1998). The MNIST database of handwritten digits.
   - [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

### Livros e Cursos

2. **Hands-On Machine Learning**
   - Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.

2. **Pattern Recognition and Machine Learning**
   - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

### Artigos Científicos

4. **Random Forests**
   - Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

2. **K-Nearest Neighbors**
   - Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*.

3. **MLP e Backpropagation**
   - Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*.

### Documentação

7. **Scikit-learn**: [https://scikit-learn.org/](https://scikit-learn.org/)
2. **NumPy**: [https://numpy.org/](https://numpy.org/)
3. **Pandas**: [https://pandas.pydata.org/](https://pandas.pydata.org/)
4. **Matplotlib**: [https://matplotlib.org/](https://matplotlib.org/)
5. **Seaborn**: [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

---

## 🙏 Agradecimentos

- **Instituto Nacional de Padrões e Tecnologia (NIST)** — Pela disponibilização do dataset MNIST
- **Comunidade Open-Source** — Por todas as ferramentas e bibliotecas
- **Professores e Mentores** — Pela orientação e conhecimento compartilhado
- **Colegas de Curso** — Pelas discussões e colaborações

---

## 📋 Checklist do Projeto

### Requisitos Atendidos ✅

- [x] Carregamento e EDA do dataset MNIST
- [x] Pipeline de pré-processamento completo
- [x] 3 modelos contrastantes (Random Forest, KNN, MLP)
- [x] Avaliação comparativa com métricas multiclasse
- [x] Matrizes de confusão com heatmaps
- [x] Class Masking (dígitos 4 e 7)
- [x] Teste de generalização OOD
- [x] Pipeline para imagens próprias
- [x] Código no GitHub público
- [x] README completo
- [x] `requirements.txt`
- [x] Branch `develop` configurada
- [x] Commits atômicos e descritivos

### Bônus Implementados 🎁

- [x] Visualizações avançadas com Seaborn
- [x] Análise de trade-off (tempo vs performance)
- [x] Conceitos de XAI e overconfidence
- [x] Guia completo de instalação
- [x] Estrutura de projeto profissional
- [x] Badges e formatação do README
- [x] 100% scikit-learn (sem TensorFlow/Keras)

---

## ⭐ Finalização

**Se este projeto te ajudou de alguma forma, considere dar uma estrela ⭐ no GitHub!**

```bash
# Clonar e executar o projeto
git clone https://github.com/seu-usuario/mnist-digit-classifier.git
cd mnist-digit-classifier
pip install -r requirements.txt
jupyter notebook mnist_pipeline.ipynb
```

**Happy Coding! 🚀**

---

*Última atualização: [Data Atual]*  
*Criado com ❤️ por [Seu Nome]*
