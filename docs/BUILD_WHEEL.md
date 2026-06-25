# Distribuer un wheel précompilé avec la depthwise

La depthwise n'ajoute que du Python (+ un kernel CUDA JIT) par-dessus le
backend déjà compilé de spconv. Inutile donc de recompiler spconv : on
**repackage un wheel précompilé** en y injectant les fichiers patchés. Résultat
= un seul `.whl` drop-in, installable par n'importe qui.

## Construire le wheel

Sur n'importe quelle machine (aucun compilateur requis pour *construire* le
wheel, c'est juste de la copie de fichiers) :

```bash
python tools/build_patched_wheel.py --spec spconv-cu120==2.3.8
# -> dist/spconv_cu120-2.3.8-1-cp311-cp311-....whl
```

Choisis la variante CUDA + version qui a un prebuilt pour ton Python cible
(`spconv-cu120`, `-cu124`, `-cu126`… ; cp39–cp313 selon la release). Pour
couvrir plusieurs environnements, relance avec différents `--spec` (et
idéalement sur différentes versions de Python) : chaque wheel est spécifique à
un couple (CUDA, Python).

## Installer

```bash
pip install --force-reinstall --no-deps "dist/spconv_cu120-2.3.8-1-...whl"
```

`--no-deps` évite que pip retouche cumm/torch déjà en place.

## Publier sur GitHub (pour que d'autres l'installent)

1. Crée une *Release* sur ton dépôt et joins le ou les `.whl` produits.
2. Les autres installent directement par URL :

```bash
pip install https://github.com/<toi>/<repo>/releases/download/<tag>/<fichier>.whl
```

(ou via `pip install <repo>` si tu publies un index, mais une Release suffit.)

## Précompiler le kernel (aucune compilation chez l'utilisateur)

Par défaut le kernel CUDA se compile en JIT à la 1re utilisation. Pour livrer
un wheel **où l'utilisateur ne compile jamais**, ajoute `--precompile` : le
kernel est compilé maintenant en **fatbin multi-arch** et embarqué dans le
wheel.

```bash
python tools/build_patched_wheel.py --spec spconv-cu120==2.3.8 --precompile
# arch couvertes par défaut : 7.5 8.0 8.6 8.9 12.0+PTX
```

Cela nécessite **CUDA + un compilateur C++ sur la machine de build** (MSVC sous
Windows, gcc sous Linux). Les archs par défaut couvrent :

| GPU | compute capability |
|-----|--------------------|
| RTX 2000 (Turing) | 7.5 |
| RTX 3000 / A-series (Ampere) | 8.0, 8.6 |
| RTX 4000, RTX 5000 Ada, RTX 3500 Ada, RTX 6000 Ada (Lovelace) | 8.9 |
| RTX 5000-series, RTX PRO 6000 (Blackwell) | 12.0 (CUDA ≥ 12.8) |

`+PTX` ajoute la compat avant pour les archs futures. Pour personnaliser :
`--arch "7.5 8.6 8.9"`.

> ⚠️ Un wheel `--precompile` est **spécifique au couple (Python, torch, OS)** de
> la machine de build (c'est la nature d'une extension binaire). Construis-en un
> par environnement cible. Si l'extension précompilée ne matche pas chez un
> utilisateur, le code retombe automatiquement sur le JIT puis sur torch.

## Ce que contient le wheel

* spconv précompilé d'origine (backend `core_cc`, cumm, etc.) — inchangé.
* Les fichiers patchés : `pytorch/{ops,functional,conv,__init__}.py`,
  `pytorch/depthwise_kernel.py`, `pytorch/csrc/depthwise.cu`.
* Avec `--precompile` : le module binaire `spconv_depthwise_C` (chargé en
  priorité par `depthwise_kernel.py` → zéro compilation à l'usage).

Sans `--precompile`, le kernel se compile en JIT à la 1re utilisation (ou
retombe sur le chemin pur-torch). Pour figer l'archi du JIT :
`export TORCH_CUDA_ARCH_LIST=8.6` (selon ton GPU).

## Vérifier après installation

```bash
python test/verify_depthwise.py
```

Doit afficher `depthwise backend = fused CUDA kernel` puis `ALL PASSED`.
