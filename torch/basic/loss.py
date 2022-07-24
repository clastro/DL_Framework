# Initialize the scores and ground truth
logits = torch.tensor([[-1.2, 0.12, 4.8]]) #최종 스코어
ground_truth = torch.tensor([2]) #Target Value

# Instantiate cross entropy loss
criterion = nn.CrossEntropyLoss() 

# Compute and print the loss
loss = criterion(logits, ground_truth)
print(loss)
