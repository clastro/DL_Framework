# for analyzing imbalanced data in tensorflow 
import tensorflow_addons as tfa

def focal_loss_custom(alpha, gamma):
    def binary_focal_loss(y_true, y_pred):
        fl = tfa.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)
        y_true_K = K.ones_like(y_true)
        focal_loss = fl(y_true, y_pred)
        return focal_loss
    return binary_focal_loss
  
adam = optimizers.Adam(lr=1e-04)
model.compile(loss = focal_loss_custom(alpha=0.2, gamma=2.0),
              optimizer=adam, 
              metrics=['accuracy',metrics.AUC()])
