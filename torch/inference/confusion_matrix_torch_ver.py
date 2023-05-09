y_pred = []
y_true = []

# iterate over test data
for inputs, labels in tqdm(test_loader):
    output = model(inputs.cuda()) # Feed Network
    output = (torch.max(torch.exp(output), 1)[1]).cpu().numpy()
    y_pred.extend(output) # Save Prediction

    labels = labels.data.cpu().numpy()
    y_true.extend(labels) # Save Truth

# constant for classes
classes = ('Normal', 'AFib')
# Build confusion matrix
cf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sns.heatmap(df_cm, annot=True)
