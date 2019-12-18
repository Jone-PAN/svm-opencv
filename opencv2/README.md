## SVM Classifier For OpenCV2

### 环境
1. 首先安装`Python2`环境下的`Opencv`，并确保装的是`Opencv2`:

    ```bash
    sudo apt-get install python-opencv
    ```

    进入`Python`环境查看`Opencv`的版本：

    ```python
    python

    >>> import cv2
    >>> cv2.__version__
    '2.4.9.1'
    >>> 

    ```

    如果`import cv2` 报如下错误：

    ```python
    >>> import cv2
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    ImportError: libopencv_imgproc.so.2.4: cannot open shared object file: No such file or directory
    >>> 

    ```

    可执行脚本解决：

    ```bash
    source scripts/reinstall_opencv_package.sh 
    ```

### 训练

1. 训练用的数据集放在`data/`目录下，每个类对应一个文件夹。
2. HOG的参数保存在`hog_descriptor.xml`文件中，程序会直接加载该文件，请确保该文件存在。
3. 在`opencv2-train.py`文件中，

    设置图像需resize的大小：

    ```python
    WIDTH = 16
    HEIGHT = 16
    ```

    设置类别名和对应的标签：

    ```python
    name_and_label = {'circle': 0, 'forward': 1, 'left': 2, 'right': 3}
    ```

    (可选) 设置SVM的参数：

    ```python
    model = SVM(C=2.5, gamma=0.50625)
    ```

    注释掉`test()`函数，使用`train()`函数进行训练。

    ```python
    if __name__ == '__main__':
        train()
        # test()
    ```

    然后运行`opencv2-train.py` 即可进行训练，训练好的SVM模型将被保存到`svm-opencv2.xml`文件中。
4. 建议：先用`opencv3/`下的程序训练模型，得到`hog_descriptor.xml`文件，然后再拿到这里用。由于`opencv2`的SVM没有`trainAuto`这样的接口，需要自己设置参数，可以先用`opencv3`训练好SVM模型，再根据其参数设置`opencv2`下SVM的`C`和`gamma`，然后再训练。

### 测试

如果需要用其他的数据测试训练好的模型，可以把测试图片放入`test`目录下，文件命名方式为`类别名_*.jpg`，比如`left_1234.jpg`。然后注释掉`train()`函数使用`test()`函数进行测试。

```python
if __name__ == '__main__':
    # train()
    test()
```