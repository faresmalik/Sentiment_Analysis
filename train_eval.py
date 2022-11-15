
import torch
from tweets_utils import tweet_to_tensor

def train_model (model, train_loader, optimizer ,criterion , epochs = 10, batch_size = 64,num_train_ex = 8000): 
    """
    Train the model

    """
    stop = True
    step = 0
    loss_train = 0.0
    train_acc = 0.0
    epoch = 1
    
    counter = 0
    while stop: 
        model.train()
        for data, labels, _ in train_loader:
            optimizer.zero_grad()
            output = model(data)
            predictions = torch.argmax(output, dim=-1)
            train_acc += ((predictions == labels).sum().item())/len(labels)
            loss = criterion(output, labels)
            loss_train += loss.item()
            loss.backward()
            optimizer.step()
            step +=1
            counter +=1 
            if step == (num_train_ex /batch_size): 
                print(f'epochs : {epoch}') 
                print(f'Steps  = {step} \t loss = {(loss_train/step):.4f} \t Acc = {train_acc/step:.4f}')
                print('==============================================')
                step = 0
                train_acc = 0.0
                loss_train = 0.0
                epoch +=1 
            if counter == (epochs * num_train_ex /batch_size): 
                stop = False
                break
    return model

def evaluate_model (model, test_loader, criterion):
    """
    Evaluate model 

    """
    loss_test = 0.0 
    test_acc = 0.0
    c = 0
    train = False
    if train == False:
            model.eval()
            for data, labels, _ in test_loader:
                c += 1 
                output = model(data)
                predictions = torch.argmax(output, dim=-1)
                test_acc += ((predictions == labels).sum().item())/len(labels)
                loss = criterion(output, labels)
                loss_test += loss.item()
            print(f'loss = {(loss_test/c):.4f} \t Acc = {test_acc/c:.4f}')

def inference(tweet, vocab, model):
    my_tensor = tweet_to_tensor(tweet, vocabulary=vocab)
    model.eval()
    with torch.no_grad():
        output = torch.argmax(model(torch.tensor(my_tensor).unsqueeze(dim=0)), dim=-1)
    if output.item() == 0: 
        print('Negative Sentiment')
    else: 
        print('Positive Sentiment')