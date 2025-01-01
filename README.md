# Iptables Command Generation

This is a proof-of-concept project that aimed at verifying an idea of generating a command line based on natural language. Iptables is a Linux tool to configure Linux kernel’s firewall setting, defining the network traffic filtering rules in Linux kernel. This POC project is designed to generate iptables command lines given NL descriptions. 

## Training data
Initially, 15 example iptables commands and their descriptions were pulled from various websites as training data. This was a good starting point, considering it was a POC project. During the experiments, it became clear that the number of the training data was way too less than what’s required. Several iptables commands were duplicated, with different port numbers, which helped increase the sample data to several hundreds.

## Model experiment
Three models were tested in this project:

**1. Seq2Seq:** The first attempt leveraged a sequence-to-sequence (Seq2Seq) model that includes an Encoder and a Decoder. The Encoder takes the tokenized NL input and processes it to generate a fixed-length context vector representing the input. Words are embedded into dense vectors, and these embeddings are passed through an LSTM to produce the final hidden and cell states.  The Decoder uses the context vector (hidden and cell states from the Encoder) to generate the output sequence (the iptables command) token by token. 

Teacher forcing was used in training to aid faster convergence, where the actual next token from the ground truth is provided as input to the decoder at each step with a certain probability. The testing description and the command the seq2seq model generated are listed below. The command used an incorrect CLI option, which should have been “accept” instead of “drop”, and the model was unable to apply the provided port number in the output.

* Test description: ```"allow incoming traffic on port 443"```
* Generated command: ```"iptables -a input -p tcp — dport <num> -j drop"```

**2. Transformer:** A small transformer model was tested next, in response to the issues from the first seq2seq model. This transformer model follows a structured encoder-decoder sequence-to-sequence approach. The encoder processes the input description by embedding each token into a dense vector representation and adding positional encodings to preserve the sequence order. This enriched sequence is passed through multiple layers of the encoder, which uses multi-head self-attention and feed-forward layers to encode the input into a latent representation. The decoder generates the corresponding iptables command token-by-token by utilizing the encoder's latent representation, going through a series of multi-head attention layers and a fully connected layer. 

Typical values were used for critical hyperparameters in this transformer model, such as the embedding size and the number of attention heads, given the small use case and small model in this project. With the same test description, the transformer model provided less satisfying result as the Seq2Seq model: 

* Test description: ```"allow incoming traffic on port 188"```
* Generated Command: ```“iptables iptables input -p -p —dport drop . drop”```

**3. GPT-2:** A pre-trained GPT-2 model is used in the third attempt, considering that the training data set was too small for the transformer model. The GPT-2 model was the default small model (124M parameters) downloaded from Hugging Face. Tokenized description/command pairs were used to fine-tune the model, to support the specific task of translating descriptions into iptables commands. The fine-tuned GPT-2 model was able to provide the closest result among the three models for the same test description, shown as below. Some garbage texts followed the desired command, which requires further tweak.

* Test description: ```"allow incoming traffic on port 188"```
* Generated Command: ```“iptables -t filter -A INPUT -p tcp —dport 188 -j ACCEPTPlant.''.AllahFigure”```

## Conclusion 
In summary, this POC project confirmed that the idea of using a LLM to translate natural language into a command line of a specific tool is feasible. Three models were tested in the experiment: Seq2Seq, transformer, and a pre-trained GPT-2 small model with fine-tuning. Unsurprisingly, the GTP-2 small model provided the best result. The small size of training data set is obviously a challenge to overcome in a real implementation of the Seatbelt CLI command generation. Some repetitive commands but with different region and availability zone names, and different parameters would help increase the size of the training data. 
