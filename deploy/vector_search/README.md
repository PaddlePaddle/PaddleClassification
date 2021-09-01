# 向量检索

**注意**：由于系统适配性问题，在新版本中，此检索算法将被废弃。新版本中将使用[faiss](https://github.com/facebookresearch/faiss)，整体检索模型库的使用流程不变，但是简历索引及检索时的yaml文件有所修改。

## 1. 简介

一些垂域识别任务（如车辆、商品等）需要识别的类别数较大，往往采用基于检索的方式，通过查询向量与底库向量进行快速的最近邻搜索，获得匹配的预测类别。向量检索模块提供基础的近似最近邻搜索算法，基于百度自研的Möbius算法，一种基于图的近似最近邻搜索算法，用于最大内积搜索 (MIPS)。 该模块提供python接口，支持numpy和 tensor类型向量,支持L2和Inner Product距离计算。

Mobius 算法细节详见论文 （[Möbius Transformation for Fast Inner Product Search on Graph](http://research.baidu.com/Public/uploads/5e189d36b5cf6.PDF), [Code](https://github.com/sunbelbd/mobius)）



## 2. 安装

### 2.1 直接使用提供的库文件

该文件夹下有已经编译好的`index.so`(gcc8.2.0下编译，用于Linux)以及`index.dll`(gcc10.3.0下编译，用于Windows)，可以跳过2.2与2.3节，直接使用。

如果因为gcc版本过低或者环境不兼容的问题，导致库文件无法使用，则需要在不同的平台下手动编译库文件。

**注意：**
请确保您的 C++ 编译器支持 C++11 标准。


### 2.2 Linux上编译生成库文件

运行下面的命令，安装gcc与g++。

```shell
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install build-essential gcc g++
```

可以通过命令`gcc -v`查看gcc版本。

进入该文件夹，直接运行`make`即可，如果希望重新生成`index.so`文件，可以首先使用`make clean`清除已经生成的缓存，再使用`make`生成更新之后的库文件。

### 2.3 Windows上编译生成库文件

Windows上首先需要安装gcc编译工具，推荐使用[TDM-GCC](https://jmeubank.github.io/tdm-gcc/articles/2020-03/9.2.0-release)，进入官网之后，可以选择合适的版本进行下载。推荐下载[tdm64-gcc-10.3.0-2.exe](https://github.com/jmeubank/tdm-gcc/releases/download/v10.3.0-tdm64-2/tdm64-gcc-10.3.0-2.exe)。

下载完成之后，按照默认的安装步骤进行安装即可。这里有3点需要注意：
1. 向量检索模块依赖于openmp，因此在安装到`choose components`步骤的时候，需要勾选上`openmp`的安装选项，否则之后编译的时候会报错`libgomp.spec: No such file or directory`，[参考链接](https://github.com/dmlc/xgboost/issues/1027)
2. 安装过程中会提示是否需要添加到系统的环境变量中，这里建议勾选上，否则之后使用的时候还需要手动添加系统环境变量。
3. Linux上的编译命令为`make`，Windows上为`mingw32-make`，这里需要区分一下。


安装完成后，可以打开一个命令行终端，通过命令`gcc -v`查看gcc版本。

在该文件夹(deploy/vector_search)下，运行命令`mingw32-make`，即可生成`index.dll`库文件。如果希望重新生成`index.dll`文件，可以首先使用`mingw32-make clean`清除已经生成的缓存，再使用`mingw32-make`生成更新之后的库文件。

### 2.4 MacOS上编译生成库文件

运行下面的命令，安装gcc与g++:

```shell
brew install gcc
```
#### 注意：
1. 若提示 `Error: Running Homebrew as root is extremely dangerous and no longer supported...`,  参考该[链接](https://jingyan.baidu.com/article/e52e3615057a2840c60c519c.html)处理
2. 若提示 `Error: Failure while executing; `tar --extract --no-same-owner --file...`， 参考该[链接](https://blog.csdn.net/Dawn510/article/details/117787358)处理

在安装之后编译后的可执行程序会被复制到/usr/local/bin下面，查看这个文件夹下的gcc：
```
ls /usr/local/bin/gcc*
```
可以看到本地gcc对应的版本号为gcc-11，编译命令如下: (如果本地gcc版本为gcc-9, 则相应命令修改为`CXX=g++-9 make`)
```
CXX=g++-11 make
```

## 3. 快速使用

    import numpy as np
    from interface import Graph_Index

    # 随机产生样本
    index_vectors = np.random.rand(100000,128).astype(np.float32)
    query_vector = np.random.rand(128).astype(np.float32)
    index_docs = ["ID_"+str(i) for i in range(100000)]

    # 初始化索引结构
    indexer = Graph_Index(dist_type="IP") #支持"IP"和"L2"
    indexer.build(gallery_vectors=index_vectors, gallery_docs=index_docs, pq_size=100, index_path='test')

    # 查询
    scores, docs = indexer.search(query=query_vector, return_k=10, search_budget=100)
    print(scores)
    print(docs)

    # 保存与加载
    indexer.dump(index_path="test")
    indexer.load(index_path="test")
