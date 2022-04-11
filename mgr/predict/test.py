import torch
import tqdm as tq
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from mgr.train.data import MGRFeatures
from mgr.configuration import load_configurations


def predict_test(
        model,
        device,
        criterion,
        labels_map=[
            'Electronic',
            'Experimental',
            'Folk',
            'Hip-Hop',
            'Instrumental',
            'International',
            'Pop',
            'Rock'],
        plot_cf=True):
    """ Predict labels of test set.

    Arguments
    ---------
    model: torch.nn.Module
        Model to be used.
    device: torch.device
        Device to run the model on.
    criterion: torch.nn.Module
        Loss function to be used.
    labels_map: list
        List of labels.
    plot_cf: bool
        Plot confusion matrix.

    Returns
    -------
    list
        List of predicted labels.
    """

    CFG = load_configurations()

    features = np.load(CFG['transformer']['test']['features_path'])
    labels = np.load(CFG['transformer']['test']['labels_path'])
    features = torch.FloatTensor(features)

    transform = transforms.Compose([
        transforms.Normalize(0.5, 0.5)
    ])

    test_dataset = MGRFeatures(features, labels, transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=True)

    losses, predicts, correct = [], [], []
    model.eval()
    with torch.no_grad():
        test_bar = tq.tqdm_notebook(
            test_loader,
            total=len(test_loader),
            desc="Testing",
            position=0,
            leave=True)
        for features, labels in test_bar:
            features, labels = features.to(device), labels.type(
                torch.LongTensor).to(device)
            output = model(features)
            loss = criterion(output, labels)

            correct.extend(labels.cpu().numpy())
            predicts.extend(torch.argmax(output, dim=1).cpu().numpy())

            losses.append(loss.item())
            test_bar.set_postfix(loss=np.mean(losses))

    if plot_cf:
        predict_labels = [labels_map[x.item()] for x in predicts]
        correct_labels = [labels_map[x.item()] for x in correct]
        print(classification_report(correct_labels, predict_labels))

        cm = confusion_matrix(correct_labels, predict_labels)
        plt.figure(figsize=(16, 16))
        fig, ax = plt.subplots(figsize=(12, 12))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels_map)
        disp.plot(ax=ax)

    return np.mean(losses), accuracy_score(predicts, correct)
