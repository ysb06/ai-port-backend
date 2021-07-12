import torch
from trans_model.torch_transformer import TransformerTransModel
from torchtext.data.metrics import bleu_score
from konlpy.tag import Kkma

def translate(model: TransformerTransModel, source_sentence: str, kor2idx: dict, eng2idx: dict, idx2eng: dict, device=torch.device('cpu'), max_length: int=67):
    kkma = Kkma()
    tokenized_sentence = ["<sos>"] + kkma.morphs(source_sentence) + ["<eos>"]
    encoded_sentence = [kor2idx[morph] if morph in kor2idx else kor2idx["<unk>"] for morph in tokenized_sentence]
    source = torch.LongTensor(encoded_sentence).unsqueeze(1).to(device)
    target = torch.LongTensor([eng2idx["<sos>"]]).unsqueeze(1).to(device)

    model.eval()
    for _ in range(max_length):
        with torch.no_grad():
            output = model(source, target)

        best_guess = output.argmax(2)[-1, :]
        last_word = best_guess.item()
        best_guess = best_guess.unsqueeze(1)
        target = torch.cat((target, best_guess), 0)

        if last_word == eng2idx["<eos>"]:
            break
    
    translated_sentence = [idx2eng[idx] for idx in target.squeeze(1).cpu().numpy()]

    if translated_sentence[-1] != "<eos>":
        return translated_sentence[1:]
    else:
        return translated_sentence[1:-1]

def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(checkpoint["kor2idx"])