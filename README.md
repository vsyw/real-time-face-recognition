# real-time-face-recognition

![pipeline](./pipeline.jpg)
### Running FaceRecognitionPipeline.ipynb
```
$ jupyter notebook
```
### Pre-trained Tensroflow Facenet Model
Github: [TensorFlow implementation of the face recognizer described in the paper "FaceNet: A Unified Embedding for Face Recognition and Clustering".](https://github.com/davidsandberg/facenet)


### Facenet Model Visualization
```
$ tensorboard --logdir='logs/'
```

### Porting to iOS
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/ios
大家无法直接将该计算图加载至iOS应用当中。完整的计算图中包含的操作目前还不受TensorFlow C++ API的支持。正因为如此，我们才需要使用刚刚构建完成的其它两款工具。其中freeze_graph负责获取graph.pb以及包含有W与b训练结果值的checkpoint文件。其还会移除一切在iOS之上不受支持的操作。

在终端当中立足tensorflow目录运行该工具：
```
$ bazel-bin/tensorflow/python/tools/freeze_graph \ 
--input_graph=/tmp/voice/graph.pb --input_checkpoint=/tmp/voice/model \
--output_node_names=model/y_pred,inference/inference --input_binary \
--output_graph=/tmp/voice/frozen.pb 
```









