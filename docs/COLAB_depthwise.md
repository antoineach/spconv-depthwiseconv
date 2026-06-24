# Tester la depthwise conv en ligne (Google Colab / Kaggle, GPU gratuit)

Sur Linux il n'y a **pas besoin de MSVC** : le kernel CUDA fusé est compilé en
JIT par `nvcc` + `gcc`, déjà présents sur Colab/Kaggle. C'est donc l'endroit
idéal pour valider le backend `fused CUDA kernel` et lancer le benchmark.

## Google Colab

1. Ouvre https://colab.research.google.com → *Nouveau notebook*.
2. Menu **Exécution → Modifier le type d'exécution → Accélérateur matériel : GPU**.
3. Colle et exécute la cellule suivante (installe spconv **précompilé** +
   applique le patch depthwise + lance la vérification) :

```python
# 0) infos environnement (Colab récent = Python 3.12)
!python -V && nvcc --version | tail -1

# 1) spconv PRÉCOMPILÉ. IMPORTANT: --only-binary empêche pip de retomber sur le
#    paquet `cumm` source (qui essaie de se compiler à l'import et échoue).
#    Choisis une variante CUDA qui publie des wheels pour ta version de Python:
#    cu126/cu124 ont des prebuilts cp312/cp313 en 2.3.8 (cu120 souvent non).
!pip -q install --only-binary=:all: "spconv-cu126==2.3.8" || \
 pip -q install --only-binary=:all: "spconv-cu124==2.3.8"

# sanity: cet import NE DOIT PAS déclencher de compilation cumm/ninja
!python -c "import cumm, spconv; print('cumm/spconv import OK')"

# 2) récupère la branche avec la depthwise
!rm -rf /content/spconv-depthwiseconv
!git clone -q --branch claude/magical-ramanujan-lb921d \
    https://github.com/antoineach/spconv-depthwiseconv.git /content/spconv-depthwiseconv

# 3) superpose les fichiers python par-dessus le spconv précompilé (pas de compilation ici)
%cd /content/spconv-depthwiseconv
!python tools/install_depthwise_over_prebuilt.py

# 4) lance la vérification depuis /content pour que `import spconv` prenne bien
#    le paquet installé (et patché), pas le checkout source.
%cd /content
!python /content/spconv-depthwiseconv/test/verify_depthwise.py
```

> **`import spconv` lance une compilation `ninja` qui échoue** (`tensorview/...:
> No such file`) ? Ça veut dire que le wheel installé n'est PAS un prebuilt pour
> ta version de Python (pip a pris le `cumm` source). Réinstalle avec
> `--only-binary=:all:` et une variante CUDA qui a des wheels cp3.x (cu126/cu124
> en 2.3.8). Ce problème vient de spconv/cumm lui-même, pas de la depthwise.

La **première exécution compile le kernel CUDA** (~30-60 s, c'est normal). Tu
dois voir :

```
depthwise backend = fused CUDA kernel
... PASS sur les 4 cas (max_abs ~1e-16) ...
gradcheck subm: PASS
=== benchmark ... ===   <- la colonne "vs full" devrait passer > 1x
ALL PASSED
```

> Si tu vois `pure-torch fallback` au lieu de `fused CUDA kernel`, c'est qu'il
> manque le toolkit CUDA (`nvcc`). Sur Colab GPU il est présent ; vérifie avec
> `!nvcc --version`. Au pire `!apt-get -q install -y cuda-toolkit` (ou choisis
> un runtime GPU standard).

### Forcer le repli torch (A/B)

Pour comparer kernel vs torch sur la même machine :

```python
import os
os.environ["SPCONV_DEPTHWISE_DISABLE_CUDA"] = "1"   # avant le 1er import de spconv
```

## Kaggle Notebooks

Même principe (Linux + GPU). Active *Settings → Accelerator → GPU*, puis les
mêmes 4 étapes. Kaggle bloque parfois le réseau : active *Internet → On* dans
les settings du notebook pour autoriser `pip install` et `git clone`.

## Note sur la version de spconv

Le script d'installation avertit si la version du wheel précompilé diffère de
celle du repo (`version.txt`). La depthwise n'ajoute que du Python par-dessus
des ops déjà compilées, donc une 2.3.x quelconque convient en général ; en cas
d'avertissement de mismatch, installe la même version mineure
(`pip install spconv-cu120==2.3.8`).
```
