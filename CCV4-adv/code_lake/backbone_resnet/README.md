resnet v0 代码参考
https://www.jianshu.com/p/12bbc8662f71 

权重交叉熵实现参考
https://blog.csdn.net/u011939633/article/details/103499750

restnet v1,v2
https://github.com/MachineLP/models/blob/master/research/slim/nets/resnet_v1.py
https://github.com/MachineLP/models/blob/master/research/slim/nets/resnet_v2.py
https://github.com/MachineLP/models/blob/master/research/slim/nets/resnet_v2_test.py
其中，
https://github.com/MachineLP/models/blob/master/research/slim/nets
里面包含了大量的常见的net的实现，这个github仓库是fork了tensorflow的models：
https://github.com/tensorflow/models
现在slim的这些实现不再包含在tensorflow/tensorflow的代码库中,而是新建了一个仓库，所以查看tensorflow的库已经找不到这些nets的实现了；
