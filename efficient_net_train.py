# from google.colab import drive
# drive.mount("/content/gdrive")
# !pip install efficientnet_pytorch

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import os

image_dimension = 512
batch_size = 8
num_workers = 2
num_classes = 2
num_epochs = 20
folder_location = "/content/gdrive/MyDrive/599_final_project/"
model_type = "efficient_net_original"
per_epoch_lr_decay = 0.9
recovered = False
weight_decay = 0.0005


def get_mask_data(augmentation=0):
    # no pretrain
    # model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
    model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)

    model = model.to(device)
    transform_train = transforms.Compose([
        transforms.Resize(image_dimension),
        transforms.RandomCrop(image_dimension, padding=8, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),  # 50% of time flip image along y-axis
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_dimension),
        transforms.CenterCrop(image_dimension),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(root=folder_location + 'FaceMaskDataset/Train',
                                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valset = torchvision.datasets.ImageFolder(root=folder_location + 'FaceMaskDataset/Validation',
                                              transform=transform_train)
    valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # testset = torchvision.datasets.ImageFolder(root=folder_location+'FaceMaskDataset/Test', transform=transform_test)
    testset = torchvision.datasets.ImageFolder(root=folder_location + 'human_faces', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = open(folder_location + "FaceMaskDataset/names.txt").read().strip().split("\n")

    # Backward mapping to original class ids (from folder names) and species name (from names.txt)
    class_to_idx = trainset.class_to_idx
    idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
    idx_to_name = {k: classes[v] for k, v in idx_to_class.items()}
    return model, {'dataset': trainset, 'train': trainloader, 'val': valloader, 'test': testloader,
                   'to_class': idx_to_class, 'to_name': idx_to_name}


def train(net, dataloader, epochs, optimizer, effective_epoch):
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    acc = 0.0
    loss_sum = 0.0
    total_itr = 0
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(dataloader))
        for i, batch in progress_bar:
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            curr_loss = loss.item()
            loss_sum += curr_loss
            total_itr += 1
            optimizer.step()  # takes a step in gradient direction

            get_acc_limit = 100
            if i % get_acc_limit == get_acc_limit - 1:
                # see predicted result
                softmax = torch.exp(outputs).cpu()
                prob = list(softmax.detach().numpy())
                predictions = np.argmax(prob, axis=1)
                acc = accuracy(predictions, batch[1].numpy())
            progress_bar.set_description(str(("epoch", effective_epoch, "lr", optimizer.param_groups[0]['lr'], "acc", acc, "loss", curr_loss)))
    return str(acc), str(loss_sum / total_itr)


def predict(net, dataloader, ofname):
    #out = open(ofname, 'w')
    #out.write("path,class\n")
    net.to(device)
    net.eval()
    test_correct = 0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(dataloader)):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            #fname, _ = dataloader.dataset.samples[i]
            test_correct += (predicted == labels).sum().item()
            #out.write("test/{},{}\n".format(fname.split('/')[-1], data['to_class'][predicted.item()]))
    test_acc = test_correct / (len(dataloader) * batch_size)
    print("test_acc", test_acc)
    #out.close()


def accuracy(y_pred, y):
    return np.sum(y_pred == y).item() / y.shape[0]


def train_and_valid(model, train_loader, val_loader, epoch_i):
    schedule = {0: 0.09, 5: 0.01, 15: 0.001, 20: 0.0001, 30: 0.00001}
    # schedule = {0: 0.09}
    if epoch_i in schedule:
        print("found new schedule, overriding optimizer learning rate")
        new_lr = schedule[epoch_i]
    else:
        new_lr = optimizer.param_groups[0]['lr'] * per_epoch_lr_decay
    for g in optimizer.param_groups:
        g['lr'] = new_lr

    train_acc, train_loss = train(model, train_loader, 1, optimizer, epoch_i)
    val_acc, val_loss = train(model, val_loader, 1, optimizer, epoch_i)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch_i
    }
    # So we write the log before the checkpoint. If we get duplicate epochs, use the last line.
    with open(folder_location + model_type + "_log.txt", "a+") as log_file:
        log_file.write("epoch " + str(epoch_i) + " train acc " + train_acc + " loss " + train_loss + "\n")
        log_file.write("epoch " + str(epoch_i) + " val acc " + val_acc + " loss " + val_loss + "\n")
    torch.save(checkpoint, checkpoint_path)


# next steps
# 1. use a model to extract faces from a video frame
# 2. feed our model with the extracted face
# import cv2
# # https://www.youtube.com/watch?v=7Rpna0mNXHE
# video_path = folder_location + "test_video.mp4"
# out_path = folder_location + "out.avi"
# def video_detect(net, video_path = video_path, out_path = out_path):
#   cap = cv2.VideoCapture(video_path)
#   success, image = cap.read()
#   if not success:
#     print("fail to open")

#   frame_width = int(cap.get(3))
#   frame_height = int(cap.get(4))
#   out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))
#   frame = 0
#   result = 0
#   net.eval()
#   while success:
#     success, image = cap.read()
#     if success:
#       if frame % 2 == 0:
#         resized = cv2.resize(image, (image_dimension, image_dimension), cv2.INTER_LINEAR)
#         resized = resized.transpose((2,0,1))
#         image_tensor = torch.Tensor(resized)
#         image_tensor = image_tensor[None, :]
#         with torch.no_grad():
#           image_tensor = image_tensor.to(device)
#           outputs = net(image_tensor)
#           _, predicted = torch.max(outputs.data, 1)
#           result = predicted.item()
#       if frame % 30 == 0:
#         print(frame)
#       cv2.putText(image, "wearing: %d" % result, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
#       frame += 1
#       out.write(image)
#     else:
#       break

#   cap.release()
#   out.release()
#   cv2.destroyAllWindows()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    model, data = get_mask_data()
    checkpoint_path = folder_location + model_type + '.pth'
    if weight_decay > 0:
        print("using weight decay", weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay)
    epoch = 0
    if os.path.exists(checkpoint_path):
        print("found checkpoint, recovering")
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch'] + 1
    else:
        print("no checkpoint, using new optimizer")
    for idx in range(epoch, num_epochs):
        train_and_valid(model, data['train'], data['val'], idx)
    predict_file_path = folder_location + model_type + ".csv"
    predict(model, data['test'], predict_file_path)
