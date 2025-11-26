# =====================================================================
#        DL LAB COMPLETE PY FILE (ALL 8 PROGRAMS IN ONE FILE)
# =====================================================================

# ---------------------------------------------------------------------
# PROGRAM 1 : SKIP-GRAM WORD EMBEDDING (DL1)
# ---------------------------------------------------------------------
def program_1_skipgram():
    import torch, torch.nn as nn, torch.optim as optim
    import random

    corpus = "I love deep learning and I love natural language processing".lower().split()
    vocab = list(set(corpus))

    word_to_ix = {w: i for i, w in enumerate(vocab)}
    ix_to_word = {i: w for w, i in word_to_ix.items()}

    # Create skip-gram pairs
    window = 2
    pairs = []
    for i, word in enumerate(corpus):
        for j in range(max(0, i - window), min(len(corpus), i + window + 1)):
            if i != j:
                pairs.append((word, corpus[j]))

    class SkipGram(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.in_embed = nn.Embedding(vocab_size, embed_dim)
            self.out_embed = nn.Embedding(vocab_size, embed_dim)

        def forward(self, c, o):
            v = self.in_embed(c)
            u = self.out_embed(o)
            return torch.sum(v * u, dim=1)

    model = SkipGram(len(vocab), 10)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        total_loss = 0
        for center, context in random.sample(pairs, len(pairs)):
            c = torch.tensor([word_to_ix[center]])
            o = torch.tensor([word_to_ix[context]])
            neg = torch.randint(0, len(vocab), (3,))

            pos_score = model(c, o)
            neg_score = model(c.repeat(3), neg)

            loss = -(torch.log(torch.sigmoid(pos_score)) +
                     torch.sum(torch.log(torch.sigmoid(-neg_score))))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss={total_loss:.4f}")

    print("\nWord embeddings:")
    for w in vocab:
        print(w, model.in_embed.weight[word_to_ix[w]].detach().numpy())


# ---------------------------------------------------------------------
# PROGRAM 2 : DNN FOR IRIS CLASSIFICATION
# ---------------------------------------------------------------------
def program_2_iris_dnn():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X = iris.data
    y = iris.target

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    class DNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 8), nn.ReLU(),
                nn.Linear(8, 6), nn.ReLU(),
                nn.Linear(6, 3)
            )

        def forward(self, x):
            return self.net(x)

    model = DNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(200):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss={loss.item():.4f}")

    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)
        acc = (preds == y_test).float().mean()

    print(f"Accuracy: {acc * 100:.2f}%")


# ---------------------------------------------------------------------
# PROGRAM 3 : CNN FOR MNIST CLASSIFICATION
# ---------------------------------------------------------------------
def program_3_cnn_classification():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32*7*7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32*7*7)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")

    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            _, predicted = torch.max(preds, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


# ---------------------------------------------------------------------
# PROGRAM 4 : AUTOENCODER ON MNIST
# ---------------------------------------------------------------------
def program_4_autoencoder():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(784, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 32)
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 784), nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        total_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.view(-1, 784).to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")


# ---------------------------------------------------------------------
# PROGRAM 5 : TEXT CLASSIFICATION WITH LSTM
# ---------------------------------------------------------------------
def program_5_text_classification():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    print("Loading data...")
    data = fetch_20newsgroups(subset='all')
    texts, labels = data.data, data.target
    num_classes = len(set(labels))

    MAX_WORDS = 10000
    MAX_LEN = 300

    vectorizer = CountVectorizer(max_features=MAX_WORDS, stop_words='english')
    X = vectorizer.fit_transform(texts).toarray()

    if X.shape[1] < MAX_LEN:
        X = np.pad(X, ((0, 0), (0, MAX_LEN - X.shape[1])), constant_values=0)
    else:
        X = X[:, :MAX_LEN]

    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    class TextClassifier(nn.Module):
        def __init__(self, vocab_size, emb, hid, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb)
            self.lstm = nn.LSTM(emb, hid, batch_first=True)
            self.fc = nn.Linear(hid, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = TextClassifier(MAX_WORDS, 128, 128, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("Training...")
    for epoch in range(3):
        total_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(Xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")

    print("Evaluating...")
    correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            out = model(Xb)
            _, pred = torch.max(out, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


# ---------------------------------------------------------------------
# PROGRAM 6 : TIME SERIES FORECASTING WITH LSTM
# ---------------------------------------------------------------------
def program_6_timeseries():
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import math

    n = 1000
    t = np.arange(n)
    data = np.sin(0.02*t) + 0.5*np.random.randn(n)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    def create_seq(data, window=20):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i:i+window])
            y.append(data[i+window])
        return np.array(X), np.array(y)

    X, y = create_seq(data_scaled)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = LSTMNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("Training LSTM...")
    for epoch in range(20):
        optimizer.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss={loss.item():.6f}")

    with torch.no_grad():
        preds = model(X_test).numpy()
        y_real = y_test.numpy()

    preds = scaler.inverse_transform(preds)
    y_real = scaler.inverse_transform(y_real)

    mae = np.mean(np.abs(preds - y_real))
    rmse = math.sqrt(np.mean((preds - y_real)**2))

    print(f"MAE={mae:.4f} RMSE={rmse:.4f}")


# ---------------------------------------------------------------------
# PROGRAM 7 : SIMPLE CNN (MNIST)
# ---------------------------------------------------------------------
def program_7_simple_cnn():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1000, shuffle=False)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3)
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(5 * 5 * 32, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 5 * 5 * 32)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("Training...")
    for epoch in range(1):
        for X, y in train_loader:
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

    print("Evaluating...")
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            out = model(X)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


# ---------------------------------------------------------------------
# PROGRAM 8 : GRID WORLD ENVIRONMENT
# ---------------------------------------------------------------------
def program_8_gridworld():
    import numpy as np
    import os

    grid = np.array([
        ['S', ' ', ' ', 'X', ' ', ' '],
        [' ', 'X', ' ', ' ', 'X', ' '],
        [' ', ' ', ' ', ' ', ' ', 'G']
    ])

    start = (0, 0)
    goal = (2, 5)

    moves = {
        'W': (-1, 0),
        'S': (1, 0),
        'A': (0, -1),
        'D': (0, 1)
    }

    def display(pos):
        os.system('cls' if os.name == 'nt' else 'clear')
        m = grid.copy()
        m[start] = 'S'
        m[goal] = 'G'
        if pos not in [start, goal]:
            m[pos] = 'A'
        for row in m:
            print(" ".join(row))

    pos = start
    display(pos)

    while True:
        move = input("Move (W/A/S/D) or Q: ").upper()
        if move == "Q":
            break

        if move in moves:
            new = (pos[0] + moves[move][0], pos[1] + moves[move][1])
            if 0 <= new[0] < 3 and 0 <= new[1] < 6 and grid[new] != 'X':
                pos = new

        display(pos)

        if pos == goal:
            print("Goal Reached!")
            break


# =====================================================================
# PROGRAM MENU
# =====================================================================
print("""
===============================
   DL LAB - 8 PROGRAMS MENU
===============================
1. Skip-Gram Word Embedding
2. DNN Classification (Iris)
3. CNN MNIST
4. Autoencoder MNIST
5. Text Classification (LSTM)
6. Time Series Forecasting (LSTM)
7. Simple CNN
8. Grid World
""")

option = input("Enter program number (1-8): ")

if option == "1": program_1_skipgram()
elif option == "2": program_2_iris_dnn()
elif option == "3": program_3_cnn_classification()
elif option == "4": program_4_autoencoder()
elif option == "5": program_5_text_classification()
elif option == "6": program_6_timeseries()
elif option == "7": program_7_simple_cnn()
elif option == "8": program_8_gridworld()
else:
    print("Invalid choice!")
