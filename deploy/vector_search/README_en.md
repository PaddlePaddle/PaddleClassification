# Vector search

**Attention**: Due to the system adaptability problem, this retrieval algorithm will be abandoned in the new version. [faiss](https://github.com/facebookresearch/faiss) will be used in the new version. The use process of the overall retrieval system base will remain unchanged, but the yaml files for build indexes and retrieval will be modified.

## 1. Introduction

Some vertical domain recognition tasks (e.g., vehicles, commodities, etc.) require a large number of recognized categories, and often use a retrieval-based approach to obtain matching predicted categories by performing a fast nearest neighbor search with query vectors and underlying library vectors. The vector search module provides the basic approximate nearest neighbor search algorithm based on Baidu's self-developed Möbius algorithm, a graph-based approximate nearest neighbor search algorithm for maximum inner product search (MIPS). This module provides python interface, supports numpy and tensor type vectors, and supports L2 and Inner Product distance calculation.

Details of the Mobius algorithm can be found in the paper.（[Möbius Transformation for Fast Inner Product Search on Graph](http://research.baidu.com/Public/uploads/5e189d36b5cf6.PDF), [Code](https://github.com/sunbelbd/mobius)）

## 2. Installation

### 2.1 Use the provided library files directly

This folder contains the compiled `index.so` (compiled under gcc8.2.0 for Linux) and `index.dll` (compiled under gcc10.3.0 for Windows), which can be used directly, skipping sections 2.2 and 2.3.

If the library files are not available due to a low gcc version or an incompatible environment, you need to manually compile the library files under a different platform.

**Note：** Make sure that C++ compiler supports the C++11 standard.

### 2.2 Compile and generate library files on Linux

Run the following command to install gcc and g++.

```
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install build-essential gcc g++
```

Check the gcc version by the command `gcc -v`.

`make` can be operated directly. If you wish to regenerate the `index.so`, you can first use `make clean` to clear the cache, and then use `make` to generate the updated library file.

### 2.3 Compile and generate library files on Windows

You need to install gcc compiler tool first, we recommend using [TDM-GCC](https://jmeubank.github.io/tdm-gcc/articles/2020-03/9.2.0-release), you can choose the right version on the official website. We recommend downloading [tdm64-gcc-10.3.0-2.exe](https://github.com/jmeubank/tdm-gcc/releases/download/v10.3.0-tdm64-2/tdm64-gcc-10.3.0-2.exe).

After the downloading, follow the default installation steps to install. There are 3 points to note here:

1.  The vector search module depends on openmp, so you need to check the `openmp` installation option when going on to `choose components` step, otherwise it will report an error `libgomp.spec: No such file or directory`, [reference link](https://github.com/dmlc/xgboost/issues/1027)
2.  When being asked whether to add to the system environment variables, it is recommended to check here, otherwise you need to add the system environment variables manually later.
3. The compile command is `make` on Linux and `mingw32-make` on Windows, so you need to distinguish here.

After installation, you can open a command line terminal and check the gcc version with the command `gcc -v`.

Run the command `mingw32-make` to generate the `index.dll` library file under the folder (deploy/vector_search). If you want to regenerate the `index.dll` file, you can first use `mingw32-make clean` to clear the cache, and then use `mingw32-make` to generate the updated library file.

### 2.4 Compile and generate library files on MacOS

Run the following command to install gcc and g++:

```
brew install gcc
```

#### Caution：

1. If prompted with `Error: Running Homebrew as root is extremely dangerous and no longer supported... `, refer to this [link](https://jingyan.baidu.com/article/e52e3615057a2840c60c519c.html)
2.  If prompted with `Error: Failure while executing; tar --extract --no-same-owner --file... `, refer to this [link](https://blog.csdn.net/Dawn510/article/details/117787358).

After installation the compiled executable is copied under /usr/local/bin, look at the gcc in this folder:

```
ls /usr/local/bin/gcc*
```

The local gcc version is gcc-11, and the compile command is as follows: (If the local gcc version is gcc-9, the corresponding command should be `CXX=g++-9 make`)

```
CXX=g++-11 make
```

## 3. Quick use

```
import numpy as np
from interface import Graph_Index

# Random sample generation
index_vectors = np.random.rand(100000,128).astype(np.float32)
query_vector = np.random.rand(128).astype(np.float32)
index_docs = ["ID_"+str(i) for i in range(100000)]

# Initialize index structure
indexer = Graph_Index(dist_type="IP") #support "IP" and "L2"
indexer.build(gallery_vectors=index_vectors, gallery_docs=index_docs, pq_size=100, index_path='test')

# Query
scores, docs = indexer.search(query=query_vector, return_k=10, search_budget=100)
print(scores)
print(docs)

# Save and load
indexer.dump(index_path="test")
indexer.load(index_path="test")
```
