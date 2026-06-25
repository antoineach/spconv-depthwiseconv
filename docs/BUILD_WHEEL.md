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

## Ce que contient le wheel

* spconv précompilé d'origine (backend `core_cc`, cumm, etc.) — inchangé.
* Les fichiers patchés : `pytorch/{ops,functional,conv,__init__}.py` +
  `pytorch/depthwise_kernel.py`.

Le kernel CUDA fusé **se compile en JIT à la première utilisation** sur la
machine de l'utilisateur (ou retombe sur le chemin pur-torch s'il n'y a pas de
toolchain). Il n'est volontairement pas précompilé dans le wheel car cela le
lierait à un ABI python/torch/CUDA exact. Pour figer l'archi GPU et accélérer
la compilation JIT : `export TORCH_CUDA_ARCH_LIST=8.6` (selon ton GPU).

## Vérifier après installation

```bash
python test/verify_depthwise.py
```

Doit afficher `depthwise backend = fused CUDA kernel` puis `ALL PASSED`.
