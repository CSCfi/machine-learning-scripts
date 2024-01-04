embedding_dims = 50
lstm_units = 32

class TwoLayeredRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(nb_words, embedding_dims)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(embedding_dims, lstm_units, num_layers=2,
                            batch_first=True)
        self.linear = nn.Linear(lstm_units, 1)

        # With bidirectional
        #self.lstm = nn.LSTM(embedding_dims, lstm_units, num_layers=2,
        #                    batch_first=True, bidirectional=True)
        #self.linear = nn.Linear(lstm_units*2, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.emb(x)
        x = self.dropout(x)
        x, (hn, cn) = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return self.sigmoid(x.view(-1))
