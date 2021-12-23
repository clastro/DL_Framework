## torch.save에서 흔히 할 수 있는 실수

torch.save({
                    'epoch': epoch,  # 현재 학습 epoch
                    'model_state_dict': model.state_dict(),  # 모델 저장
                    'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
                    'loss': loss.item(),  # Loss 저장
                    'train_step': i,  # 현재 진행한 학습
                    'total_train_step': len(train_loader)  # 현재 epoch에 학습 할 총 train step
                }, save_ckpt_path)


data_path = "./data/filename.csv"
checkpoint_path ="./checkpoint"
save_ckpt_path = f"{checkpoint_path}/kobert_youtube_topic_classification.pth"

# .pth라는 확장명으로 파일을 저장하는 건데 이 파일 이름을 제대로 지정하지 않으면 PermissionError (실제로는 경로 오류) 오류가 발생


