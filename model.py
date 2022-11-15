import torch


class classifier(torch.nn.Module): 
    def __init__(self, vocab_size = 9089, output_dim = 2, embedding_dim = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.Embedding_layer = torch.nn.Embedding(num_embeddings= vocab_size, embedding_dim=embedding_dim)
        self.FC1 = torch.nn.Linear(in_features= embedding_dim, out_features=embedding_dim//4)
        self.FC2 = torch.nn.Linear(in_features= embedding_dim//4, out_features= output_dim)

    def forward(self, x): 
        x = self.Embedding_layer(x)
        x = torch.mean(x, dim=1)
        x = torch.nn.functional.relu(self.FC1(x))
        x = self.FC2(x)
        return x