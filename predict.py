import torch
from argparse import ArgumentParser
import pandas as pd
from prime.utils import read_seq, scan_max_mutant, score_mutant
from prime.model import Config, ForMaskedLM


@torch.no_grad()
def main():
    psr = ArgumentParser()
    psr.add_argument("--model_path", type=str, required=True)
    psr.add_argument("--fasta", type=str, required=True)
    psr.add_argument("--mutant", type=str, required=True)
    psr.add_argument("--save", type=str, required=True)
    args = psr.parse_args()
    
    
    sequence = read_seq(args.fasta)
    df = pd.read_csv(args.mutant)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForMaskedLM(Config())
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model = model.to(device)

    df, sequence, offset = scan_max_mutant(df=df, seq=sequence)

    sequence_ids = model.tokenize(sequence).to(device)
    attention_mask = torch.ones_like(sequence_ids).to(device)
    logits = model(input_ids=sequence_ids, attention_mask=attention_mask)[0]
    logits = torch.log_softmax(logits, dim=-1)
    mutants = df["mutant"]
    scores = []
    for mutant in mutants:
        score = score_mutant(
            mutant, sequence, logits=logits, vocab=model.VOCAB, offset=offset
        )
        scores.append(score)
    df["predict_score"] = scores
    df.to_csv(args.save, index=False)


if __name__ == "__main__":
    main()
