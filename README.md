# deeplearning

# GPU로 딥러닝 하는 방법

## 먼저 CUDA TOOLKIT과 cudnn을 버전에 맞게 설치한다.

### 1. cudnn은 회원가입이 필요함
### 2. 다음 명령어를 사용해서 GPU가 잘 작동하는 것을 확인할 수 있다.

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 12539132906350912286
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 6938994944
locality {
  bus_id: 1
  links {
  }
}
incarnation: 8764573227598342969
physical_device_desc: "device: 0, name: NVIDIA GeForce RTX 3060 Ti, pci bus id: 0000:01:00.0, compute capability: 8.6"
]
