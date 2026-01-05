<h1 align="center" style="font-weight: bold;">Mixture of Experts (MoE) PKLot 🚗</h1>

<p align="center">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
    <img src="https://img.shields.io/badge/PIL-%23013243.svg?style=for-the-badge&logo=python&logoColor=white" alt="PIL"/>
    <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
</p>

<p align="center">
  <a href="#sobre">Sobre</a> •
  <a href="#arquitetura">Arquitetura</a> • 
  <a href="#pkLot">Base de Dados PKLot</a> •
  <a href="#experimentos">Experimentos</a> •
  <a href="#uso">Como Usar</a> •
  <a href="#resultados">Resultados</a>
</p>

<p align="center">
  <i>Implementação de Mixture of Experts com roteador sparse para classificação de vagas de estacionamento, testando diferentes arquiteturas de experts.</i>
</p>

---

<h2 id="sobre">📋 Sobre o Projeto</h2>

Este projeto implementa uma arquitetura **Mixture of Experts (MoE)** para detecção de vagas de estacionamento utilizando a base de dados **PKLot** e **CNR-Park**. O objetivo é investigar como diferentes arquiteturas de experts impactam o desempenho do modelo, permitindo análise comparativa de diversos designs de redes neurais.

**Características principais:**
- ✅ Arquitetura MoE com roteador sparse (top-k selection)
- ✅ Suporte a múltiplos experts com arquiteturas customizáveis
- ✅ Treino, validação e teste automáticos
- ✅ Cálculo de métricas (Loss, Acurácia)
- ✅ Compatível com GPU e CPU

---

<h2 id="arquitetura">🧠 Arquitetura MoE</h2>

### Componentes Principais

```mermaid
graph LR
  A["Input (Imagem 124x124x3)"]
  B["Router (Gating Network)\ntop-k=2"]
  A --> B

  C["Expert 1 (CNN)"]
  D["Expert 2 (CNN)"]
  E["Expert 3 (CNN)"]

  B -->|selecionado| C
  B -->|selecionado| D
  B -->|não selecionado| E

  F["Weighted Merge"]

  C --> F
  D --> F
  E -.->|ignorado| F

  G["Output (2 classes)"]
  F --> G
```

### Componentes

1. **Router (Gating Network)**
   - Seleciona os top-k melhores experts para cada entrada
   - Economiza computação (sparse routing)
   - Saída: weights para cada expert

2. **Experts (CNNs independentes)**
   - Múltiplas CNNs especializadas
   - Cada uma processa a entrada independentemente
   - Saída: logits de classificação

3. **Merge (Weighted Sum)**
   - Combina saídas dos experts usando pesos do router
   - Produz classificação final

---



<h2>🤝 Autor</h2>
<table align="left">
  <tr>
    <td align="left">
      <a href="https://www.linkedin.com/in/lucasdoc/">
        <img src="https://avatars.githubusercontent.com/u/89359426?v=4" width="100px;" alt="Foto de Lucas Cunha"/>        <sub>
        <br>
          <b>Lucas Cunha</b>
        </sub>
      </a>
    </td>
  </tr>
</table>
