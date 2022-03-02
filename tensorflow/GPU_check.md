# DeepLearning

# GPU로 딥러닝 하는 방법

### CUDA TOOLKIT과 cudnn을 버전에 맞게 설치한다.

##### 1. cudnn은 회원가입이 필요함
##### 2. 다음 명령어를 사용해서 GPU가 잘 작동하는 것을 확인할 수 있다.

```
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
##### 여러 명이 서버를 사용할 때 탄력적으로 GPU 
```
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
```
