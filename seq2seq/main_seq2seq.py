import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import torch.optim as optim
import pandas as pd

# Load dataset from CSV
def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    The file must contain two columns: 'description' and 'command'.
    """
    data = pd.read_csv(file_path)
    if "description" not in data.columns or "command" not in data.columns:
        raise ValueError("The dataset file must contain 'description' and 'command' columns.")
    return data[["description", "command"]].values.tolist()


# Dataset preparation
def tokenize(sentence):
    """
    Tokenize a sentence into words and replace numeric tokens with <num>.
    """
    tokens = word_tokenize(sentence.lower())
    return ["<num>" if token.isdigit() else token for token in tokens]


def build_vocab(data, min_freq=1):
    """
    Build a vocabulary from the dataset.
    Includes <pad>, <sos>, <eos>, and <unk> tokens for proper handling.
    """
    tokens = [tokenize(pair[0]) for pair in data] + [tokenize(pair[1]) for pair in data]
    vocab = build_vocab_from_iterator(tokens, specials=["<pad>", "<sos>", "<eos>", "<unk>"], min_freq=min_freq)
    vocab.set_default_index(vocab["<unk>"])  # Set <unk> as the default index for unknown tokens
    return vocab


class IptablesDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data  # List of (description, command) pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nl, cmd = self.data[idx]
        nl_tokens = [self.vocab["<sos>"]] + [self.vocab[token] for token in tokenize(nl)] + [self.vocab["<eos>"]]
        cmd_tokens = [self.vocab["<sos>"]] + [self.vocab[token] for token in tokenize(cmd)] + [self.vocab["<eos>"]]
        return torch.tensor(nl_tokens), torch.tensor(cmd_tokens)


def pad_batch(batch, pad_idx):
    nl, cmd = zip(*batch)
    nl_padded = nn.utils.rnn.pad_sequence(nl, padding_value=pad_idx, batch_first=True)
    cmd_padded = nn.utils.rnn.pad_sequence(cmd, padding_value=pad_idx, batch_first=True)
    return nl_padded, cmd_padded


# Encoder-Decoder Model
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, cell):
        embedded = self.dropout(self.embedding(trg.unsqueeze(1)))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[1]
        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        trg_input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(trg_input, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            """ Either use the grouth truth trg[:,t] or the predicted token (top1) as the decoder 
            input at next time step"""
            trg_input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs


# Training loop
def train_model(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


# Inference
def translate_sentence(model, sentence, vocab, device, max_len=50):
    model.eval()
    tokens = [vocab["<sos>"]] + [vocab[token] for token in tokenize(sentence)] + [vocab["<eos>"]]
    src = torch.tensor(tokens).unsqueeze(0).to(device)
    hidden, cell = model.encoder(src)
    trg_indexes = [vocab["<sos>"]]
    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == vocab["<eos>"]:
            break
    trg_tokens = [vocab.lookup_token(idx) for idx in trg_indexes]
    return " ".join(trg_tokens[1:-1])


# Main script
if __name__ == "__main__":
    
    # Load the dataset from my CSV file
    dataset_file = "iptables_commands_extra.csv"  
    data = load_dataset(dataset_file)

    # Build vocabulary
    nltk.download('punkt')
    nltk.download('punkt_tab')
    vocab = build_vocab(data)

    # Prepare dataset and dataloader
    dataset = IptablesDataset(data, vocab)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=lambda x: pad_batch(x, vocab["<pad>"]))

    # Model parameters
    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(vocab)
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, and loss
    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    # Training
    N_EPOCHS = 10
    CLIP = 1

    for epoch in range(N_EPOCHS):
        train_loss = train_model(model, dataloader, optimizer, criterion, CLIP)
        print(f"Epoch {epoch + 1}/{N_EPOCHS}, Train Loss: {train_loss:.3f}")

    # Inference
    sentence = "allow incoming traffic on port 443"
    #sentence = "append a rule to the OUTPUT chain to allow all outgoing traffic"
    #sentence = "Allow outgoing connection on port 25 to network 192.160.5.0/24"
    print("Description:", sentence)
    print("Generated Command:", translate_sentence(model, sentence, vocab, device))
