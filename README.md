准备工作：
  图片什么尺寸需要在train.py文件的开头设置，如`WIDTH = 16    HEIGHT = 32`。         
  然后根据自己的数据集情况，在get_hog()函数中设置HOG的参数，尤其注意get_hog()函数中winSize尺寸需要自己指定为具体数字尺寸，不要用变量或者什么加减乘除的。（关于hog参数的设置解释可以参考 https://blog.csdn.net/yangpan011/article/details/78353873），之前识别箭头把winsize设置为图像相同大小是因为箭头占满了整张图片，对于关键特征并没有占满整张图片的时候，需要把winsize设置小一点。            
  然后再把train.py文件中的类别标签`name_and_label = {'lightOn': 0, 'lightOff': 1}`修改为自己的类别即可。      
1.如果环境需要使用opencv3的svm，先在`train.py`中设置label，然后运行opencv3下的`python3 train.py`,之后会得到`hog_descriptor.xml`和`svm.xml`两个文件。哪里需要用到svm模型直接用`hog_descriptor.xml`和`svm.xml`再新建一个`label.txt`文件共三个文件即可，这个新建的`label.txt`文件中的各个类别名称按svm的分类顺序填写。
2.如果环境需要使用opencv2的svm,可以先使用`opencv3/`下的程序训练模型，然后会得到一个`svm.xml`的模型文件，该文件中里面有`C`和`gamma`，然后把`opencv2-train.py`中`model = SVM(C=1.25, gamma=0.03375)`这句话的`C`和`gamma`值修改为opencv3训练完成的`svm.xml`中的`C`和`gamma`值，然后再进行使用opencv2进行训练，训练完成会得到`hog_descriptor.xml`和`svm-opencv2.xml`两个文件。
3.数据集放在`opencv2/`或`opencv3/`下的data文件夹下。
       把每个类别文件夹下的图片前缀以类别文件夹名命名（`python3 rename.py`是用来重命名图片名的），训练时可以不用这么命名，测试时候测试图片也需要这么来命名。

4.测试
如果需要用其他的数据测试训练好的模型，可以把测试图片放入test目录下，文件命名方式为 类别名_*.jpg，比如left_1234.jpg。然后注释掉train()函数使用test()函数进行测试。
```c
if __name__ == '__main__':
    # train()
    test()
```
