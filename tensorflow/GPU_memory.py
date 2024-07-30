import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.client import device_lib

### Version 1.0

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# tensorflow package에 한해 GPU를 사용 증가량 만큼 사용할 수 있음

### Version 2.0

# GPU 디바이스 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 증가 방식으로 설정
        for gpu in gpus:
            tf.config.set_memory_growth(gpu, True)

        # GPU 메모리 제한 설정 (예: 2GB)
        for gpu in gpus:
            tf.config.set_virtual_device_configuration(
                gpu,
                [tf.config.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)
