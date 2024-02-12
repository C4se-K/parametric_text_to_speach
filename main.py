import os
from preprocessor import preprocessor
import encoder
import torch
import torch.nn as nn
import torch.optim as optim


def create_mask(input_tensor):
    #masks all padding
    mask = (input_tensor != 0).to(torch.float32)
    return mask.unsqueeze(1).unsqueeze(2)

def prepare_data(tokens, mel_specs, device, max_len = 20):
    padded_tokens = [torch.nn.functional.pad(t.clone().detach(), (0, max(0, max_len - len(t))), "constant", 0) for t in tokens]
    padded_tokens = torch.stack(padded_tokens).to(device)
    padded_mel_specs = torch.nn.utils.rnn.pad_sequence(mel_specs, batch_first=True).to(device)
    return padded_tokens, padded_mel_specs

def run():
    preprocessor_instance = preprocessor()
    dataset = preprocessor_instance.run()  #returns a list of (mel, tokens)
    
    #create tensors for both the tokens and mels
    tokens = [torch.tensor(item[1]) for item in dataset]
    mel_specs = [torch.tensor(item[0], dtype=torch.float32) for item in dataset]

    print('preprocessor')
    print(len(tokens))
    print(len(mel_specs[0]), len(mel_specs[1]))

    src_vocab_size = 254 # = len(tokens)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = encoder.make_model(src_vocab=src_vocab_size).to(device)
    
    prepared_tokens, prepared_mel_specs = prepare_data(tokens, mel_specs, device)

    print('padding')
    print(len(prepared_tokens), prepared_tokens.shape)
    print(len(prepared_mel_specs[0]), len(prepared_mel_specs[1]), " ", prepared_mel_specs.shape)

    
    #masking
    mask = create_mask(prepared_tokens).to(device)
    
    #loss and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    print('masking')
    print(len(prepared_tokens), prepared_tokens.shape)
    print(len(prepared_mel_specs[0]), len(prepared_mel_specs[1]), " ", prepared_mel_specs.shape)
    
    #training
    epochs = 100
    count = 0
    for epoch in range(epochs):
        count +=1
        print('loop number: ', count)
        
        optimizer.zero_grad()
        
        #forward pass: prediction of mel through masking of tokens
        predicted_mel_specs = model(prepared_tokens, mask)

        print(predicted_mel_specs.shape)
        
        #loss
        loss = loss_function(predicted_mel_specs, prepared_mel_specs)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
        
        loss.backward()
        optimizer.step()

    #saving the model
    model_save_path = os.path.join(os.getcwd(), 'trained_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == '__main__':
    run()
