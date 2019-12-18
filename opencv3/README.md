## SVM Classifier For OpenCV3

### 环境

在`Python3`环境下安装`OpenCV`，并确保装的是`OpenCV3`。

### 训练

1. 训练用的数据集放在`data/`目录下，每个类对应一个文件夹。
2. 在`train.py`文件中，

    设置图像需resize的大小：

    ```python
    WIDTH = 16
    HEIGHT = 16
    ```

    设置类别名和对应的标签：

    ```python
    name_and_label = {'circle': 0, 'forward': 1, 'left': 2, 'right': 3}
    ```

    (可选) 在`get_hog()`函数中设置HOG的参数：

    ```python
    winSize = IMAGE_SIZE
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (4, 4)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True
    ```

    (可选) 在`svmTrain`函数中建议使用`trainAuto`，这样可以得到较好的结果，不用手动设置SVM的参数，缺点是训练时间要稍微久一点：

    ```python
    def svmTrain(model, samples, responses):
    # model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    model.trainAuto(samples, cv2.ml.ROW_SAMPLE, responses)
    return model
    ```

    注释掉`test()`函数，使用`train()`函数进行训练。

    ```python
    if __name__ == '__main__':
        train()
        # test()
    ```

    然后运行`train.py` 即可进行训练，训练好的SVM模型将被保存到`svm.xml`文件中，HOG的参数会被保存到`hog_descriptor.xml`文件中。

### 测试

如果需要用其他的数据测试训练好的模型，可以把测试图片放入`test`目录下，文件命名方式为`类别名_*.jpg`，比如`left_1234.jpg`。然后注释掉`train()`函数使用`test()`函数进行测试。

```python
if __name__ == '__main__':
    # train()
    test()
```

### 用C++接口测试SVM模型

1. 编译C++程序`cpp_test.cpp`:

```bash
bash build.sh
```

2. 运行测试程序：

```bash
./cpp_test <image>
```

比如：

```bash
./cpp_test test/left_1234.jpg
```