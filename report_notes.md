# Notes de rapport — LLM + Fuzzy TOPSIS pour la sélection de personnel

*Cours : IA générative & aide multicritère à la décision (MCDM)*

## 1. Contexte

Le recrutement combine deux difficultés :

1. **lire et comprendre** des CV hétérogènes (formats, vocabulaire, langues) ;
2. **arbitrer** entre plusieurs critères en conflit (compétences, expérience, diplôme…),
   avec des jugements par nature **imprécis** (« bonne maîtrise de Python »).

Les LLM savent faire (1) mais leurs jugements sont non déterministes, peu explicables et
potentiellement biaisés. Les méthodes MCDM floues savent faire (2) de manière transparente,
mais supposent qu'un expert a déjà noté chaque candidat. Le projet **couple les deux** : le LLM
joue le rôle de l'évaluateur (notes linguistiques + preuves), la logique floue modélise
l'imprécision, et Fuzzy TOPSIS produit un classement auditable.

## 2. Problème

Soit m candidats A₁..Aₘ, n critères C₁..Cₙ et une offre d'emploi. On cherche un classement des
candidats qui :

- utilise l'information des CV extraite automatiquement ;
- tient compte de l'incertitude des jugements (linguistiques, et instables d'un appel LLM à l'autre) ;
- est explicable (chaque note est justifiée par une citation du CV) ;
- n'utilise pas d'attributs protégés (nom, genre, âge, nationalité).

Critères (tous de type bénéfice, configurables dans `config/criteria.yaml`) :
C1 adéquation des compétences techniques, C2 expérience pertinente, C3 formation,
C4 certifications, C5 soft skills, C6 langues.

## 3. Théorie

### 3.1 Nombres flous triangulaires (TFN)

Un TFN ã = (l, m, u), l ≤ m ≤ u, a pour fonction d'appartenance

    μ(x) = (x − l)/(m − l)   si l ≤ x ≤ m
         = (u − x)/(u − m)   si m ≤ x ≤ u
         = 0                 sinon.

Opérations utilisées (TFN positifs) :

- ã ⊕ b̃ = (l₁+l₂, m₁+m₂, u₁+u₂)
- ã ⊗ b̃ ≈ (l₁l₂, m₁m₂, u₁u₂)
- k·ã = (kl, km, ku)
- distance de sommet (vertex) : d(ã, b̃) = √( ⅓ [(l₁−l₂)² + (m₁−m₂)² + (u₁−u₂)²] )
- défuzzification par centroïde : (l + m + u)/3.

### 3.2 Échelles linguistiques

Notes des candidats et poids des critères (même échelle sur [0,1]) :

| Terme | TFN |
|---|---|
| Very Low | (0, 0, 0.25) |
| Low | (0, 0.25, 0.5) |
| Medium | (0.25, 0.5, 0.75) |
| High | (0.5, 0.75, 1) |
| Very High | (0.75, 1, 1) |

Échelle à 5 termes : un LLM distingue de façon fiable 5 niveaux, au-delà la granularité est
illusoire. Les TFN se chevauchent de 0,25, ce qui traduit le fait que deux termes voisins
sont partiellement compatibles.

### 3.3 Agrégation de K évaluateurs

Chen (2000), éq. (13) : x̃ᵢⱼ = (1/K) (x̃ᵢⱼ¹ ⊕ … ⊕ x̃ᵢⱼᴷ).
Variante utilisée pour les K exécutions du LLM (« stabilité du prompt ») :

    lᵢⱼ = minₖ lᵢⱼᵏ,   mᵢⱼ = (1/K) Σₖ mᵢⱼᵏ,   uᵢⱼ = maxₖ uᵢⱼᵏ.

Ici, chaque exécution du LLM (température 0 puis 0,7 avec d'autres graines) est traitée comme
un **décideur**. Le désaccord entre exécutions élargit le support [l, u] du TFN : l'instabilité
du LLM devient une incertitude explicite au lieu d'être cachée.

### 3.4 Fuzzy TOPSIS (Chen, 2000)

1. Matrice de décision floue D̃ = [x̃ᵢⱼ], poids W̃ = [w̃ⱼ].
2. Normalisation linéaire :
   - bénéfice : r̃ᵢⱼ = (aᵢⱼ/cⱼ*, bᵢⱼ/cⱼ*, cᵢⱼ/cⱼ*), avec cⱼ* = maxᵢ cᵢⱼ ;
   - coût : r̃ᵢⱼ = (aⱼ⁻/cᵢⱼ, aⱼ⁻/bᵢⱼ, aⱼ⁻/aᵢⱼ), avec aⱼ⁻ = minᵢ aᵢⱼ.
3. Matrice pondérée : ṽᵢⱼ = r̃ᵢⱼ ⊗ w̃ⱼ.
4. Solutions idéales : FPIS A* avec ṽⱼ* = (1,1,1), FNIS A⁻ avec ṽⱼ⁻ = (0,0,0).
   Variante « minmax » : ṽⱼ* = maxᵢ vᵢⱼ₃ et ṽⱼ⁻ = minᵢ vᵢⱼ₁.
5. Distances : dᵢ* = Σⱼ d(ṽᵢⱼ, ṽⱼ*) et dᵢ⁻ = Σⱼ d(ṽᵢⱼ, ṽⱼ⁻).
6. Coefficient de proximité : CCᵢ = dᵢ⁻ / (dᵢ* + dᵢ⁻) ∈ [0, 1].
7. Classement par CC décroissant.

### 3.5 Méthodes de comparaison

- **TOPSIS net** (Hwang & Yoon, 1981) sur les notes défuzzifiées :
  - normalisation vectorielle rᵢⱼ = xᵢⱼ / √Σᵢxᵢⱼ² ;
  - distances euclidiennes à A⁺ et A⁻ ;
  - score Cᵢ = Sᵢ⁻/(Sᵢ⁺+Sᵢ⁻).
- **Somme pondérée (SAW)** : Sᵢ = Σⱼ wⱼ rᵢⱼ, avec normalisation min-max.
- **TF-IDF cosinus**, sans LLM : similarité entre le vecteur TF-IDF du CV et celui de l'offre.

### 3.6 Métriques

- **Spearman ρ** : corrélation des rangs.
- **Kendall τ-b** : proportion de paires concordantes moins discordantes ; gère les ex æquo.
- **Top-k overlap** : |top-k(prédit) ∩ top-k(vrai)| / k.
- **top-1** : le meilleur candidat est-il retrouvé ?
- **Précision@k** sur données réelles (labels faibles = catégorie du CV).

## 4. Méthodologie

### 4.1 Pipeline

1. **Parsing** : pdfplumber pour les PDF, décodage UTF-8 ou Latin-1 pour les TXT, puis nettoyage des espaces.
2. **Anonymisation avant tout appel LLM**. On retire :
   - email, URL et téléphone, sans toucher aux plages de dates « 2019 - 2023 » ;
   - les champs étiquetés (Name, Gender, Age, Nationality, Date of birth…) ;
   - les mentions « 34 years old » ;
   - les pronoms genrés, remplacés par they/their.

   L'équité est ainsi **structurelle** : le modèle ne peut pas utiliser ce qu'il ne voit pas.
3. **Extraction structurée** : le LLM produit un `CandidateProfile` (pydantic) avec skills,
   years_experience, education[level], certifications, languages, projects, soft_skills.
   - Sortie JSON uniquement, schéma imposé via le décodage contraint d'Ollama.
   - Validation pydantic ; en cas d'échec, l'erreur est renvoyée au modèle pour qu'il corrige (3 essais).
4. **Notation linguistique** : un appel par candidat qui note les 6 critères. Pour chaque
   critère, le modèle renvoie {citation verbatim du CV, justification, terme}.
   - L'ordre des champs force le modèle à chercher la preuve *avant* de choisir le terme.
   - **Contrôle d'ancrage** : la citation est recherchée dans le CV (rapidfuzz
     `partial_ratio` ≥ 85 sur un texte normalisé).
   - Si elle est introuvable, trois politiques sont possibles : `flag` (signaler seulement),
     `downgrade` (−1 niveau, par défaut) ou `very_low`.
5. **Stabilité** : K = 3 exécutions (T = 0 puis T = 0,7 avec des graines différentes),
   agrégées par min / moyenne / max (§3.3).
6. **Fuzzy TOPSIS** de Chen (§3.4), implémenté en numpy et vectorisé. Les poids flous viennent :
   - d'un préréglage « expert » : C1 VH, C2 H, C3 M, C4 L, C5 M, C6 M ;
   - d'une suggestion du LLM ;
   - ou d'une édition dans l'app.
7. **Explications** : le LLM ne reçoit **que** le rang, le CC, les poids, les termes et les
   citations (jamais le CV). Un filtre regex rejette les mentions d'attributs protégés
   (genre, âge, nom, nationalité…) et redemande ; si l'échec persiste, un gabarit
   déterministe prend le relais.

### 4.2 Données

- **Benchmark synthétique** (`src/synthetic.py`) :
  - 20 candidats, échantillonnés par archétype avec la graine 7 : data scientist ×4,
    data analyst ×3, ML engineer ×2, backend ×4, frontend ×2, devops ×2, comptable ×2, marketing ×1 ;
  - 2 offres : **J1 Data Scientist** et **J2 Backend Java Engineer**.
- **Vérité terrain définie AVANT la rédaction des CV**, par des formules déterministes sur
  les attributs latents :
  - C1 = (|requis ∩ S| + 0,5·|préférés ∩ S|) / (|requis| + 0,5·|préférés|) ;
  - C2 = min(années pertinentes / (2 × années requises), 1) ;
  - C3 = score du niveau (bac 0,2 ; licence 0,6 ; master 0,85 ; doctorat 1) × (1 si la filière
    est pertinente, sinon 0,5) ;
  - C4 = min(#certifications pertinentes / 2, 1) ;
  - C5 = part des soft skills demandées ;
  - C6 = part des langues requises.

  Le score de vérité terrain est Σ wⱼ sⱼ, avec les poids experts défuzzifiés. Le LLM ne fait
  que **rédiger** le CV à partir de ces faits. On vérifie ensuite que chaque fait figure dans
  le texte (sinon nouvel essai, puis gabarit). Il n'y a donc pas de circularité : le modèle
  évalué ne définit pas la cible.
- **Test contrefactuel** : chaque CV existe en deux variantes qui ne diffèrent que par l'en-tête :
  - A : homme ~30 ans, nom et nationalité français ;
  - B : femme ~55 ans, nom et nationalité marocaine, nigériane, sénégalaise ou chinoise.
- **Données réelles** : Kaggle *Resume Dataset* (snehaanbhawal, CC0, ~2 400 CV Livecareer,
  24 catégories). Il n'existe pas de classement de référence, donc on utilise un label faible :
  pour une offre « IT » ou « Accountant », un CV de la catégorie correspondante est
  « pertinent ». Les autres sources ont été vérifiées : HF *resume-atlas* (licence MIT) et
  Kaggle *LinkedIn Job Postings 2023-24* (CC BY-SA 4.0, 556 Mo, optionnel).

### 4.3 Protocole d'évaluation

- Comparaison de 7 méthodes au classement vrai (Spearman, Kendall τ, top-1, recouvrement
  top-3 et top-5) :
  - Fuzzy TOPSIS (poids experts) ;
  - Fuzzy TOPSIS (poids LLM) ;
  - Fuzzy TOPSIS avec idéaux minmax ;
  - TOPSIS net ;
  - SAW ;
  - TF-IDF cosinus ;
  - « oracle », c'est-à-dire Fuzzy TOPSIS sur les vraies sous-notes arrondies au terme
    linguistique le plus proche. C'est une borne haute : ce que donnerait un LLM parfait
    avec une échelle à 5 termes.
- Ablations recalculées depuis les réponses stockées, sans nouvel appel LLM : politique
  d'ancrage, 1 run ou 3 runs, opérateur d'agrégation.
- Sensibilité des poids :
  - un critère à la fois (×0,5 … ×1,5) ;
  - scénarios « critère dominant » ;
  - Monte-Carlo (2 000 tirages, perturbation U(0,7 ; 1,3)).
- Équité : ΔCC et Δrang entre les variantes A et B, avec et sans anonymisation.
- Validation de l'implémentation sur l'exemple numérique publié de Chen (2000),
  contre pyDecision et contre scikit-fuzzy.

## 5. Implémentation

| Module | Rôle | Choix défendable |
|---|---|---|
| `llm.py` | client unique Ollama / OpenAI / Gemini | indépendance vis-à-vis du fournisseur, JSON contraint, auto-réparation |
| `cache.py` | cache SQLite des appels | reproductibilité : mêmes entrées → mêmes sorties, relance gratuite |
| `fuzzy.py` | TFN, échelles, agrégation, distance | tout est vectorisé sur l'axe (…, 3) |
| `topsis.py` | Fuzzy TOPSIS from scratch + baselines nettes | 3 définitions d'idéaux (chen / minmax / componentwise) |
| `scoring.py` | notation avec preuves + ancrage | séparation appels LLM / calcul flou → ablations gratuites |
| `explain.py` | justification sans le CV + filtre | aucune hallucination possible hors des preuves |
| `evaluation.py` | métriques, TF-IDF, sensibilité, équité | |
| `app.py` | Streamlit | changer les poids recalcule TOPSIS instantanément (aucun appel LLM) |

Tests : 50 tests pytest, exécutés hors ligne grâce à un faux backend LLM. Ils couvrent :

- les propriétés des TFN et l'exemple de Chen ;
- la parité avec pyDecision ;
- l'anonymisation, les PDF, les politiques d'ancrage et les tentatives répétées ;
- le pipeline complet et l'exécution sans tête de l'app Streamlit.

Matériel : portable sans GPU dédié (Intel Iris Xe, 16 Go de RAM). L'inférence passe par Ollama
sur CPU :

- **qwen3.5:4b** (modèle principal) : ~45 s par appel de notation ;
- **qwen3.5:0.8b** : comparaison de taille.

L'iGPU via Vulkan a été mesuré **plus lent** que le CPU (122 contre 185 tokens/s en lecture du
prompt), il n'est donc pas utilisé.

Tous les chiffres ci-dessous proviennent de `results/` et sont regénérés par
`scripts/run_all.ps1` ; une relance est gratuite grâce au cache.

## 6. Résultats

### 6.1 Validation de l'implémentation (Chen, 2000)

| | A1 | A2 | A3 | Classement |
|---|---|---|---|---|
| CC publié (Table 8) | 0,62 | 0,77 | 0,71 | A2 > A3 > A1 |
| Notre CC, depuis la Table 5 imprimée | 0,640 | 0,766 | 0,707 | A2 > A3 > A1 |
| Notre CC, depuis les notes de la Table 4 | 0,638 | 0,765 | 0,702 | A2 > A3 > A1 |
| d* publié | 2,10 | 1,24 | 1,59 | |
| d* recalculé à la main depuis la Table 7 imprimée | 1,94 | 1,26 | 1,59 | |
| Notre d* | 1,942 | 1,262 | 1,591 | |

**Constat : le papier contient deux incohérences internes.**

- Cinq cellules de la Table 5 ne sont pas la moyenne des notes de la Table 4 (par exemple
  A1·C2 = moy(G, MG, F) = (5 ; 7 ; 8,67), imprimé (5 ; 7 ; 9)), et les poids sont arrondis.
- **d₁\* = 2,10 est une erreur de calcul** : la somme des distances de la Table 7 du papier vaut
  1,94, ce qui donne CC₁ = 0,64.

Notre implémentation reproduit donc tous les calculs du papier ; CC₂, CC₃, d⁻ et le classement
sont identiques à ±0,01.

**Comparaison avec pyDecision** (`fuzzy_topsis_method`) :

- La librairie définit FPIS et FNIS composante par composante, à partir des données, et non
  (1,1,1) et (0,0,0). Elle divise aussi la distance par le nombre de critères au lieu de 3 ;
  ce facteur commun ne change pas le CC.
- Avec la même définition des idéaux (`ideal="componentwise"`), l'écart maximal est de
  **1,1 × 10⁻¹⁶**.

**Comparaison avec scikit-fuzzy** (fonctions d'appartenance `trimf` et centroïde) : écart de
0 et 2 × 10⁻¹⁶. Cette librairie n'implémente pas TOPSIS.

### 6.2 Benchmark synthétique (qwen3.5:4b, 3 runs, ancrage « downgrade », poids experts)

**Qualité de l'extraction (20 CV) :**

| Mesure | Valeur |
|---|---|
| Rappel des compétences | 0,78 |
| Précision des compétences | 0,80 |
| Niveau de diplôme correct | 95 % |
| Rappel des langues | 0,95 |
| Rappel des certifications | 0,83 |
| Années d'expérience | 3,95 en moyenne contre 6,35 réelles (erreur absolue moyenne 2,4 ans) |

Le LLM additionne mal les durées des postes.

**Accord avec le classement vrai :**

| Méthode | J1 ρ | J1 τ | J1 top-1 | J2 ρ | J2 τ | J2 top-1 |
|---|---|---|---|---|---|---|
| **Fuzzy TOPSIS (poids experts)** | 0,751 | 0,575 | ✔ | 0,780 | 0,628 | ✔ |
| Fuzzy TOPSIS (poids LLM) | 0,685 | 0,511 | ✔ | 0,780 | 0,628 | ✔ |
| Fuzzy TOPSIS, idéaux minmax | 0,778 | 0,606 | ✔ | 0,784 | 0,639 | ✔ |
| TOPSIS net (mêmes notes défuzzifiées) | 0,772 | 0,596 | ✔ | 0,806 | 0,649 | ✔ |
| Somme pondérée (SAW) | 0,792 | 0,628 | ✔ | 0,808 | 0,649 | ✔ |
| TF-IDF cosinus (sans LLM) | 0,756 | 0,606 | ✔ | 0,841 | 0,649 | ✘ |
| *Oracle : Fuzzy TOPSIS sur les vraies sous-notes* | *0,938* | *0,818* | ✘ | *0,959* | *0,887* | ✔ |

Recouvrement du top-5 : 0,6 sur J1 et 1,0 sur J2 pour toutes les méthodes LLM.

**Qualité des notes par critère** (ρ de Spearman entre le centroïde LLM et la vraie sous-note) :

| | C1 compétences | C2 expérience | C3 formation | C4 certifications | C5 soft skills | C6 langues |
|---|---|---|---|---|---|---|
| J1 | 0,90 | **0,28** | 0,83 | 0,62 | 0,46 | 0,89 |
| J2 | 0,82 | 0,54 | 0,92 | 0,91 | **0,00** | 0,39 |

Taux de citations retrouvées dans le CV : 80–100 % selon le critère.

**Ablations** (ρ ; recalculées sur les mêmes réponses du LLM) :

| Ancrage | Runs | Agrégation | J1 | J2 |
|---|---|---|---|---|
| flag | 1 | — | **0,847** | **0,864** |
| flag | 3 | min/moy/max | 0,714 | 0,855 |
| downgrade | 1 | — | 0,788 | 0,840 |
| downgrade (défaut) | 3 | min/moy/max | 0,751 | 0,780 |
| downgrade | 3 | moyenne | 0,747 | 0,803 |
| very_low | 1 | — | 0,575 | 0,748 |
| very_low | 3 | min/moy/max | 0,555 | 0,762 |

**Stabilité du LLM entre runs :**

- Kendall τ entre les classements de deux runs : 0,56–0,69 sur J1 et 0,69–0,72 sur J2.
- Même terme sur les 3 runs : seulement 52 % (J1) et 37 % (J2) des couples
  (candidat, critère).
- Largeur moyenne des TFN agrégés (u − l) : 0,49–0,52.

**Poids suggérés par le LLM :** identiques aux poids experts sauf C6 langues (High au lieu de
Medium), sur les deux offres.

**Sensibilité des poids :**

- Un critère à la fois (×0,5 … ×1,5) : τ minimum de 0,83 sur J1 et 0,93 sur J2 ; **le premier ne
  change jamais** ; décalage maximal de 4 rangs.
- Monte-Carlo (2 000 tirages, ±30 % sur tous les poids) : le premier (S03 pour J1, S13 pour J2)
  reste 1er dans **100 %** des tirages.
- Scénarios « critère dominant » : seul « C3 éducation dominante » change le gagnant de J1
  (S13, un développeur backend titulaire d'un master).

**Explications :** 0 explication sur 40 mentionne un attribut protégé. Elles citent les critères
pondérés et les preuves, par exemple : *« Very high-rated senior data science experience spanning
2022 to 2026 »*.

### 6.3 Équité : test contrefactuel (J1, qwen3.5:4b, 1 run, T = 0)

Les 20 CV ont un corps identique ; seul l'en-tête change :

- variante A : « Thomas Martin, Male, 29, French »… ;
- variante B : « Fatima Zahra El Amrani, Female, 54, Moroccan »…

| | CC identiques | ΔCC moyen (B − A) | \|ΔCC\| moyen | \|ΔCC\| max | Kendall τ A/B | Décalage moyen de rang |
|---|---|---|---|---|---|---|
| **Sans anonymisation** | 50 % | −0,011 | 0,016 | 0,109 | 0,82 | 1,6 |
| **Avec anonymisation** | 100 % | 0 | 0 | 0 | 1,00 | 0 |

**Sans anonymisation :**

- 10 CV sur 20 changent de score : 7 baissent pour la variante B, 3 montent.
- Test de Wilcoxon (rangs signés) : p = 0,28. Le biais **directionnel** n'est donc pas
  significatif sur n = 20.
- Les écarts individuels sont pourtant importants : S01 passe de 0,365 à 0,256 et S04 de 0,383
  à 0,294 (deux data scientists).
- Les termes qui changent concernent surtout C2 expérience (4 cas) et C3 formation (3 cas).

**Interprétation :** le nom, le genre, l'âge et la nationalité modifient le jugement du LLM. Ces
attributs devraient être neutres, et le prompt ordonne explicitement de les ignorer. **Une
consigne dans le prompt ne suffit pas.**

L'anonymisation en amont rend la décision invariante **par construction** : après nettoyage, les
deux textes sont identiques. Cela ne protège pas contre les indices indirects (proxies) : nom
d'école, langue maternelle, trous dans le CV, etc.

### 6.4 Données réelles (Kaggle Resume Dataset, labels faibles)

Les offres d'emploi sont rédigées pour les catégories INFORMATION-TECHNOLOGY et ACCOUNTANT.
Pour chaque offre, on tire un échantillon aléatoire (graine 42) mêlant des CV de la catégorie
cible (pertinents) et des CV d'autres catégories. Les CV sont anonymisés et tronqués à 6 000
caractères ; la notation se fait en 1 run.

**qwen3.5:4b** (20 CV dont 6 pertinents ; un tirage au hasard donne P@6 = 0,30) :

| Offre | Méthode | P@5 | P@6 |
|---|---|---|---|
| IT | Fuzzy TOPSIS | 0,80 | 0,67 |
| IT | TOPSIS net / SAW | 0,80 | 0,67 |
| IT | TF-IDF | 1,00 | 0,83 |
| Comptable | Fuzzy TOPSIS | 0,80 | 0,83 |
| Comptable | TOPSIS net | 1,00 | 0,83 |
| Comptable | TF-IDF | 1,00 | 0,83 |

**qwen3.5:0.8b** (autre échantillon de 30 CV dont 10 pertinents ; un tirage au hasard donne
P@10 = 0,33) :

| Offre | Fuzzy TOPSIS P@5 / P@10 | TF-IDF P@5 / P@10 |
|---|---|---|
| IT | 0,60 / 0,50 | 1,00 / 0,80 |
| Comptable | 0,80 / 0,50 | 1,00 / 0,90 |

**Lecture :**

- Sur des CV réels, la méthode LLM fait nettement mieux que le hasard.
- Elle **ne bat pas** une similarité lexicale. Le label faible « catégorie du CV » est
  justement un critère lexical (le métier), ce qui favorise TF-IDF.
- La méthode LLM juge en plus le niveau (expérience, diplôme, certifications), ce que ce label
  ne mesure pas.
- La qualité dépend fortement de la taille du modèle (0.8B ≪ 4B).

Les deux échantillons étant différents, la comparaison entre modèles reste indicative.

### 6.5 Effet de la taille du LLM (benchmark synthétique, 1 run, ancrage downgrade)

| | qwen3.5:0.8b | qwen3.5:4b |
|---|---|---|
| Fuzzy TOPSIS ρ, J1 / J2 | 0,197 / 0,241 | 0,788 / 0,840 |
| Top-1 trouvé (J1, J2) | 0 / 2 | 2 / 2 (avec 3 runs) |
| ρ moyen par critère (12 couples critère × offre) | 0,15 | 0,63 |
| Citations retrouvées dans le CV | 20–75 % | 80–100 % |
| Extraction : rappel des compétences / langues / certifications | 0,98 / 1,00 / 1,00 | 0,78 / 0,95 / 0,83 |
| Extraction : erreur sur les années | 2,65 ans | 2,4 ans |
| Temps par appel de notation (CPU) | ~25 s | ~45 s |

Le 0.8B **extrait** correctement les listes : il les recopie presque mot pour mot. En revanche,
il **ne sait pas juger** : ρ ≈ 0,2, sous TF-IDF (0,76 / 0,84), et il recopie souvent l'offre
d'emploi à la place du CV. Ses citations, en grande partie introuvables, montrent que
**l'ancrage des preuves détecte bien les hallucinations**.

**La capacité de jugement est la ressource critique.** Elle demande au moins un modèle de
quelques milliards de paramètres.

## 7. Comparaison et discussion

1. **Le goulot d'étranglement est la note produite par le LLM, pas la méthode MCDM.**
   - Avec les vraies sous-notes (oracle), Fuzzy TOPSIS atteint ρ = 0,94–0,96.
   - Avec les notes de qwen3.5:4b, il tombe à 0,75–0,78.
   - La perte vient surtout de deux critères : C2 expérience (le LLM additionne mal les
     périodes) et C5 soft skills (critère vague, mal calibré).
2. **Fuzzy TOPSIS, TOPSIS net et SAW sont très proches** (écarts de ρ ≤ 0,04) sur les mêmes
   notes. SAW est légèrement devant, mais la vérité terrain est elle-même une somme pondérée :
   ce benchmark favorise SAW par construction. L'apport du flou n'est pas une meilleure
   corrélation. Il tient à la **représentation explicite de l'incertitude** (largeur des TFN,
   distances d* et d⁻) et à la robustesse du haut du classement (top-1 stable à 100 % en
   Monte-Carlo).
3. **TF-IDF est un concurrent sérieux** pour la corrélation globale. En effet, des CV synthétiques
   qui répètent le vocabulaire de l'offre lui sont favorables. Mais **TF-IDF rate le meilleur
   candidat de J2**, alors que Fuzzy TOPSIS le trouve sur les deux offres. Surtout, TF-IDF ne
   fournit ni critère, ni preuve, ni explication.
4. **Agréger 3 runs n'améliore pas la précision** ; ça la dégrade légèrement (ρ 0,75 contre 0,79
   sur J1). Les runs à température 0,7 sont plus bruités que le run déterministe. L'agrégation
   min/moy/max est utile pour *mesurer* l'incertitude, pas pour la réduire.
5. **L'ancrage strict des preuves est contre-productif.** Le prendre au sens strict
   (`very_low`) coûte jusqu'à 0,27 de ρ : le modèle reformule souvent la preuve légèrement, et
   une bonne note est alors annulée. Le simple signalement (`flag`) donne le meilleur ρ. Avec un
   modèle 4B, 80 à 100 % des citations sont pourtant authentiques. Le compromis retenu
   (`downgrade`) protège contre les hallucinations au prix d'une petite perte de précision.
6. **Les poids proposés par le LLM sont plausibles** : identiques à l'expert sauf pour les
   langues. Mais ils ne suffisent pas à justifier une délégation : ils changent ρ de −0,07 sur J1.

## 8. Analyse critique

### Forces

- **Chaîne auditable :** chaque note est liée à une citation vérifiée dans le CV, chaque rang à
  CC, d* et d⁻, et chaque explication aux seules preuves.
- **Robustesse du haut du classement :** aucun changement du premier pour des poids ×0,5 à ×1,5,
  et 100 % en Monte-Carlo.
- **Équité structurelle :** l'anonymisation neutralise un effet mesuré (décalage moyen de 1,6
  rang sans elle), alors que le prompt seul ne suffit pas.
- **Reproductibilité :** cache, graines fixées, 50 tests. L'implémentation est validée contre le
  papier (en révélant une erreur de calcul de celui-ci), contre pyDecision (1e-16) et contre
  scikit-fuzzy.
- **Indépendance vis-à-vis du fournisseur :** tout tourne en local et gratuitement ; un
  changement de variable d'environnement suffit pour passer sur OpenAI ou Gemini.

### Limites

- **Instabilité du LLM :** τ entre runs de 0,56 à 0,72 seulement, et le même terme dans 37 à 52 %
  des cas. Deux exécutions à température > 0 ne produisent pas le même classement.
- **Petits modèles locaux :** voir §6.5, la qualité chute fortement avec le 0.8B. Le 4B sur CPU
  coûte ~45 s par candidat et par run, ce qui limite la taille des expériences.
- **Benchmark synthétique :**
  - 20 CV seulement, d'où des intervalles de confiance larges (le test de Wilcoxon sur
    l'équité n'a pas assez de puissance) ;
  - CV rédigés par le même modèle que celui qui les évalue (familiarité de style) ;
  - le rédacteur ajoute parfois des faits non demandés (par exemple « TensorFlow » dans S03),
    alors que seuls les faits manquants sont contrôlés ;
  - l'âge des en-têtes n'est pas cohérent avec la carrière (27 ans pour 13 ans d'expérience).
- **La vérité terrain est une modélisation** (formules et poids choisis par l'auteur). Une
  meilleure corrélation avec elle ne prouve pas une meilleure décision réelle.
- **Données réelles sans classement de référence :** on n'y dispose que de labels faibles
  (catégorie du CV).
- **Biais résiduels :** l'anonymisation par regex ne retire pas les proxies (nom d'université,
  langue maternelle, associations, trous de carrière). Le remplacement « her → their » est
  approximatif.
- **Fuzzy TOPSIS lui-même :**
  - le résultat dépend de la définition des idéaux (Chen, minmax ou componentwise donnent des CC
    différents) ;
  - le produit flou ⊗ est approché ;
  - l'échelle linguistique (forme et recouvrement des TFN) est un choix arbitraire ;
  - comme pour tout TOPSIS, ajouter ou retirer un candidat peut inverser un classement
    (rank reversal).

### Éthique

Un système de présélection automatisée est un système à haut risque au sens de l'AI Act
européen (annexe III, emploi). Il doit rester **une aide à la décision** : un humain décide, avec
traçabilité, explications, audit de biais et possibilité de contestation pour le candidat.

## 9. Perspectives

- **Approche hybride :** calcul déterministe des années d'expérience à partir des dates (regex)
  au lieu du LLM, ce qui ciblerait directement le critère le plus faible (C2).
- Vérification de la correction des faits : chaque citation doit **impliquer** le terme choisi
  (modèle NLI), pas seulement exister dans le CV.
- Modèles plus grands ou API (GPT-4o, Gemini), avec un cache par fournisseur, et comparaison
  coût / qualité.
- Calibration des termes linguistiques sur des notes humaines ; échelles à 7 termes ; TFN
  personnalisés par critère.
- Pondération floue AHP / BWM à plusieurs experts et analyse de consensus ; extensions
  intuitionnistes ou de type 2 pour mieux représenter l'hésitation.
- Benchmark plus large : 100+ CV, plusieurs rédacteurs LLM, classement par des recruteurs.
- Audit d'équité sur des proxies (écoles, villes) et tests statistiques avec plus de puissance.

## 10. Conclusion

Le couplage LLM + Fuzzy TOPSIS fournit un classement **explicable, robuste aux poids et
neutre vis-à-vis des attributs protégés une fois anonymisé**. Sur le benchmark synthétique, avec
un modèle local de 4B paramètres, il obtient ρ ≈ 0,75–0,78 et retrouve le meilleur candidat sur
les deux offres.

Sa précision est limitée par la qualité des notes du LLM. La méthode MCDM n'en est pas la cause :
avec des notes parfaites, elle atteint 0,94–0,96. Le flou apporte surtout la transparence sur
l'incertitude, et non un gain de corrélation brut.

## Références

- Chen, C.-T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy
  environment. *Fuzzy Sets and Systems*, 114, 1–9.
- Hwang, C.-L., & Yoon, K. (1981). *Multiple Attribute Decision Making*. Springer.
- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8, 338–353.
- Zadeh, L. A. (1975). The concept of a linguistic variable… *Information Sciences*, 8.
- Pereira, V. — pyDecision (bibliothèque Python MCDM).
- Kaggle *Resume Dataset* (snehaanbhawal, CC0) ; Hugging Face *resume-atlas* (MIT) ;
  Kaggle *LinkedIn Job Postings 2023-24* (arshkon, CC BY-SA 4.0).
- Règlement (UE) 2024/1689 (AI Act), annexe III.
