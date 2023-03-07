def numerial_diff(f, x, eps=1e-4): # 중앙 차분 - 오차가 적다고 알려져 있음, 그러나 계산량 많고 자릿수 누락으로 인한 오차 포함
  x0 = Variable(x, data - eps)
  x1 = Variable(x, data + eps)
  y0 = f(x0)
  y1 = f(x1)
  return (y1.data - y0.data) / (2 * eps)
