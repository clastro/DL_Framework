from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.client import device_lib

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# tensorflow package에 한해 GPU를 사용 증가량 만큼 사용할 수 있음
