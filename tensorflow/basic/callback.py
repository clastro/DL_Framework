from tensorflow.keras.callbacks import CSVLogger

history_fcn8 = model_fcn8.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, csv_logger]
)

#CSV 파일로 중간 저장 가능
