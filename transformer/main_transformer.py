import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.optim.lr_scheduler import StepLR

import pandas as pd
import os



MAX_LENGTH = 50  # Set the maximum sequence length globally


# Step 1: Dataset Preparation
class IptablesDataset(Dataset):
    def __init__(self, input_texts, output_texts, input_vocab, output_vocab, input_tokenizer, output_tokenizer, max_length=50):
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        src = [self.input_vocab["<sos>"]] + [self.input_vocab[token] 
                for token in self.input_tokenizer(self.input_texts[idx])] + [self.input_vocab["<eos>"]]
        tgt = [self.output_vocab["<sos>"]] + [self.output_vocab[token] 
                for token in self.output_tokenizer(self.output_texts[idx])] + [self.output_vocab["<eos>"]]

        # Ensure consistent padding or truncation
        src = src[:self.max_length] + [self.input_vocab["<pad>"]] * max(0, self.max_length - len(src))
        tgt = tgt[:self.max_length] + [self.output_vocab["<pad>"]] * max(0, self.max_length - len(tgt))

        # Debug sequence lengths
        assert len(src) == self.max_length, f"Source sequence length mismatch: {len(src)}"
        assert len(tgt) == self.max_length, f"Target sequence length mismatch: {len(tgt)}"

        return torch.tensor(src), torch.tensor(tgt)


# Step 2: Build Vocabulary
def build_vocab(data, tokenizer):
    def yield_tokens(data):
        for text in data:
            yield tokenizer(text)
    return build_vocab_from_iterator(yield_tokens(data), specials=["<unk>", "<pad>", "<sos>", "<eos>"])


# Step 3: Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size, num_heads, num_layers, ff_dim, max_length=50, dropout=0.1):
        assert embed_size % num_heads == 0, "Embedding size must be divisible by the number of heads"
        # print(f"\nforward =====> embed size: {embed_size}, num_heads: {num_heads}")
        super(TransformerModel, self).__init__()
        self.embedding_src = nn.Embedding(input_vocab_size, embed_size)
        self.embedding_tgt = nn.Embedding(output_vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn((1, max_length, embed_size))) 
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, ff_dim, dropout, batch_first=True), # Set batch_first=True
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, num_heads, ff_dim, dropout, batch_first=True), # Set batch_first=True
            num_layers
        )
        self.fc_out = nn.Linear(embed_size, output_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        print(f"src shape (before embedding): {src.shape}, \
                tgt shape (before embedding): {tgt.shape}, \
                positional encoding shape (before embedding): {self.positional_encoding.shape}")
        """

        src = self.embedding_src(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding_tgt(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        # print(f"src after embedding shape: {src.shape}, tgt after embedding shape: {tgt.shape}")
        
        memory = self.encoder(src, src_key_padding_mask=src_mask)
        # print(f"memory shape after encoding: {memory.shape}")
        
        output = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_mask)
        # print(f"decoder output shape: {output.shape}")
        
        return self.fc_out(output)


# Step 4: Training
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for src, tgt in data_loader:
        src, tgt = src.to(device), tgt.to(device)

        # Debug tensor shapes
        # print(f"\nTraining batch=======> Source shape: {src.shape}, Target shape: {tgt.shape}") # Debug input shapes
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_input)

        # Debug model output shape
        # print(f"\nTraining batch=======> Logits shape: {logits.shape}, Target Output shape: {tgt_output.shape}") # Debug output shapes
        
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))         # Replace view with reshape

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


# Step 5: Inference
def generate_command(model, src, input_vocab, output_vocab, input_tokenizer, max_len=50):
    model.eval()
    device = next(model.parameters()).device
    src = [input_vocab["<sos>"]] + [input_vocab[token] for token in input_tokenizer(src)] + [input_vocab["<eos>"]]
    src = torch.tensor(src).unsqueeze(0).to(device)
    tgt = torch.tensor([output_vocab["<sos>"]]).unsqueeze(0).to(device)

    for _ in range(max_len):
        output = model(src, tgt)
        next_token = output.argmax(-1)[:, -1].item()
        tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)
        if next_token == output_vocab["<eos>"]:
            break

    return " ".join([output_vocab.lookup_token(idx) for idx in tgt[0].tolist() if idx not in {output_vocab["<sos>"], output_vocab["<eos>"], output_vocab["<pad>"]}])

# Step 6: Main Script
if __name__ == "__main__":
    # Load data from CSV file
    data_file = "iptables_commands_extra.csv"  # Replace with your file path
    df = pd.read_csv(data_file)
    if "description" not in df.columns or "command" not in df.columns:
        raise ValueError("CSV file must have 'description' and 'command' columns.")
    
    input_texts = df["description"].tolist()
    output_texts = df["command"].tolist()

    # Build vocabularies
    input_tokenizer = get_tokenizer("basic_english")
    output_tokenizer = get_tokenizer("basic_english")
    input_vocab = build_vocab(input_texts, input_tokenizer)
    output_vocab = build_vocab(output_texts, output_tokenizer)

    # Add default indices
    input_vocab.set_default_index(input_vocab["<unk>"])
    output_vocab.set_default_index(output_vocab["<unk>"])

    # Prepare dataset and dataloader

    # Pass MAX_LENGTH consistently
    dataset = IptablesDataset(input_texts, output_texts, input_vocab, output_vocab, input_tokenizer, output_tokenizer, max_length=MAX_LENGTH)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    # Model parameters
    embed_size = 256
    num_heads = 8
    num_layers = 3
    ff_dim = 512
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model if available
    if os.path.exists("transformer_model.pth"):
        model.load_state_dict(torch.load("transformer_model.pth"))
        print("Model loaded from checkpoint.")

    # Initialize model, optimizer, and loss function
    model = TransformerModel(
                len(input_vocab), 
                len(output_vocab), 
                embed_size, 
                num_heads, 
                num_layers, 
                ff_dim, 
                max_length=MAX_LENGTH, 
                dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=output_vocab["<pad>"])

    # Initialize scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Train model
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model(model, data_loader, optimizer, criterion, device)
        scheduler.step()  # Adjust the learning rate
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "transformer_model.pth")

    # Inference
    test_description = "allow incoming traffic on port 188"
    generated_command = generate_command(model, test_description, input_vocab, output_vocab, input_tokenizer)
    print("Generated Command:", generated_command)
