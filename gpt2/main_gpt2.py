import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


# 1: Load Dataset
class IptablesDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=256, device="cpu"):
        """
        Dataset for training GPT to generate iptables commands.
        :param data: List of (description, command) pairs.
        :param tokenizer: Pre-trained tokenizer.
        :param max_length: Maximum sequence length.
        :param device: Device to move data tensors to ('cpu', 'cuda', or 'mps').
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, response = self.data[idx]
        input_text = f"Prompt: {prompt}\nResponse: {response}"
        encodings = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        # Set input_ids as labels (shifted during training)
        encodings["labels"] = encodings["input_ids"].copy()
        
        # Move tensors to the correct device
        return {
            key: (torch.tensor(val).to(self.device) if not isinstance(val, torch.Tensor) else val.to(self.device))
            for key, val in encodings.items()
        }

def load_dataset(file_path):
    """
    Load dataset from a CSV file with 'description' and 'command' columns.
    """
    data = pd.read_csv(file_path)
    if "description" not in data.columns or "command" not in data.columns:
        raise ValueError("The dataset must contain 'description' and 'command' columns.")
    return list(zip(data["description"], data["command"]))

# 2: Fine-Tune GPT Model
def fine_tune_gpt(data, model_name="gpt2", output_dir="./fine_tuned_gpt", epochs=3, batch_size=4):
    """
    Fine-tune a GPT model on the iptables dataset.
    :param data: List of (description, command) pairs.
    :param model_name: Pre-trained GPT model name.
    :param output_dir: Directory to save the fine-tuned model.
    :param epochs: Number of training epochs.
    :param batch_size: Training batch size.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})     # Add padding token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Set device to 'mps' (Metal Performance Shaders) if available, otherwise 'cpu'
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Move model and data to the correct device
    model = model.to(device)

    # Prepare dataset
    dataset = IptablesDataset(data, tokenizer, device=device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
        ignore_data_skip=True,  # Ensures padding tokens do not affect the loss
        report_to="none",  # Avoid extra logging for simplicity
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Fine-tuned model saved to {output_dir}")
    return model, tokenizer

# 3: Inference
def ensure_tensor(data, device="cpu"):
    """
    Ensure the input data is a tensor. If it's already a tensor, clone and detach it.
    Otherwise, convert it to a tensor.
    """
    if isinstance(data, torch.Tensor):
        return data.clone().detach().to(device)
    return torch.tensor(data).to(device)

def generate_command(description, model, tokenizer, max_length=256):
    """
    Generate an iptables command for a given description using the fine-tuned GPT model.
    :param description: Natural language description of the desired command.
    :param model: Fine-tuned GPT model.
    :param tokenizer: Tokenizer for the model.
    :param max_length: Maximum output length.
    """
    # Set device to the same as the model
    device = next(model.parameters()).device

    # Prepare the input text
    prompt = f"Prompt: {description}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    # Move inputs to the model's device
    #input_ids = inputs["input_ids"].to(device)
    input_ids = ensure_tensor(inputs["input_ids"], device)
    #attention_mask = inputs["attention_mask"].to(device)
    attention_mask = ensure_tensor(inputs["attention_mask"], device)


    # Generate output
    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_length=max_length, 
        num_beams=5, 
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id  # Ensure the model stops at <eos>
    )

    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=5, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the command part
    return generated_text.split("Response:")[1].strip()


# 4: Main Script
if __name__ == "__main__":
    # Set the input dataset file
    dataset_file = "iptables_commands_extra.csv"
    model_name = "gpt2"
    output_dir = "./fine_tuned_gpt"

    # Load dataset
    print("Loading dataset...")
    data = load_dataset(dataset_file)

    # Fine-tune the model
    print("Fine-tuning the GPT model...")
    model, tokenizer = fine_tune_gpt(data, model_name=model_name, output_dir=output_dir)

    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Move the model to the appropriate device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)


    
    # Example inference
    test_description = "allow incoming traffic on port 188"
    print(f"Generating command for: {test_description}")
    command = generate_command(test_description, model, tokenizer)
    print("Generated Command:", command)
