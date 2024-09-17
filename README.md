# Implementação NumPy do Modelo de Linguagem Probabilístico Neural

Este repositório apresenta uma implementação do Modelo de Linguagem Probabilístico Neural proposto por Yoshua Bengio e colaboradores no artigo "A Neural Probabilistic Language Model" (2003). O código é desenvolvido exclusivamente em NumPy, com o objetivo de proporcionar uma compreensão mais profunda da arquitetura do modelo. Embora o uso de frameworks como PyTorch ou TensorFlow tornasse a implementação mais fácil e otimizada, eles introduzem um nível de abstração que pode dificultar o entendimento dos conceitos fundamentais.

Este artigo é reconhecido por sua contribuição significativa à introdução da camada de embeddings. A implementação neste repositório foi treinada utilizando o texto "Dom Casmurro" de Machado de Assis.

## Uso

Primeiramente deve ser feito o treinamento do modelo rodando o arquivo `train.py`.Depois, o modelo pode ser utilizado rodando o arquivo `generate.py` e o contexto pode ser alterado na linha tokenizer.encode("José Dias amava ").

## Resultados

Os resultados podem ser encontrados no arquivo `output.txt` na raiz do projeto.

## Referências

- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model. Journal of Machine Learning Research, 3(Feb), 1137-1155.
- Assis, Machado de. Dom Casmurro. Rio de Janeiro: Garnier, 1899.