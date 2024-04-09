# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import warnings
warnings.filterwarnings("ignore")


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        nargs="+",
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        nargs="+",
    )

    parser.add_argument(
        "--fasta",
        type=str,
        help="fasta file",
    )

    parser.add_argument(
        "--mutant",
        type=str,
    )

    parser.add_argument(
        "--sep",
        type=str,
        default=",",
    )

    parser.add_argument(
        "--mutation-col",
        type=str,
        default="mutant",
    )

    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "pseudo-ppl", "masked-marginals"],
        help=""
    )
    args = parser.parse_args()

    if args.mutant.endswith(".csv"):
        args.sep = ","
        # print("Warning: the separator is set to ','.")
    elif args.mutant.endswith(".tsv"):
        args.sep = "\t"
        # print("Warning: the separator is set to '\\t'.")

    if len(args.model_path) != len(args.tokenizer_path) and len(args.tokenizer_path) == 1:
        args.tokenizer_path = args.tokenizer_path * len(args.model_path)

    return args


def mutant_filter(df, mutant_site=0):
    rows = ''.join(df["mutant"].tolist())
    sep = ":"
    if ";" in rows:
        sep = ";"
    if mutant_site:
        trg_df = df.loc[df.apply(lambda row: len(
            row["mutant"].split(sep)) == mutant_site, axis=1)]
    else:
        trg_df = df
    return trg_df


def scan_max_mutant(df, seq, max_len=100000):
    # df: dataframe of the mutant file
    df_single = mutant_filter(df, 1)
    mutants = list(df_single["mutant"])

    # WT in the first row
    has_wt = False
    if mutants[0].lower() == "wt":
        mutants = mutants[1:]
        df_no_wt = df_single[1:]
        has_wt = True

    mutant_pos = [int(m[1:-1]) - 1 for m in mutants]
    max_pos, min_pos = max(mutant_pos), min(mutant_pos)

    if max_pos < max_len:
        return df, seq[:max_len], 1

    if max_pos - min_pos + 1 <= max_len:
        seq_left_to_right = seq[min_pos: min_pos + max_len]
        # contain the last mutation residue
        seq_right_to_left = seq[max_pos - max_len + 1: max_pos + 1]

        # select the longer sequence
        if len(seq_left_to_right) > len(seq_right_to_left):
            seq = seq_left_to_right
            offset = min_pos + 1
            if has_wt:
                df_bool = df_no_wt.apply(lambda row: int(
                    row["mutant"][1:-1]) >= offset and int(row["mutant"][1:-1]) < (min_pos + max_len + 1), axis=1)
                df_bool.loc[0] = True
                df = df.loc[df_bool]
            else:
                df = df.loc[df.apply(lambda row: int(row["mutant"][1:-1]) >= offset and int(
                    row["mutant"][1:-1]) < (min_pos + max_len + 1), axis=1)]
        else:
            seq = seq_right_to_left
            offset = max_pos - max_len + 2
            if has_wt:
                df_bool = df_no_wt.apply(lambda row: int(
                    row["mutant"][1:-1]) >= offset and int(row["mutant"][1:-1]) < max_pos + 2, axis=1)
                df_bool.loc[0] = True
                df = df.loc[df_bool]
            else:
                df = df.loc[df.apply(lambda row: int(
                    row["mutant"][1:-1]) >= offset and int(row["mutant"][1:-1]) < max_pos + 2, axis=1)]
        return df, seq, offset

    global_count = 0
    global_left, global_right = min_pos, max_len
    window_left, window_right = min_pos, max_len
    # scan the whole sequence
    while (window_right < len(seq)):
        window_count = len([pos for pos in mutant_pos if pos >=
                           window_left and pos < window_right])
        if window_count > global_count:
            global_left, global_right = window_left, window_right
            global_count = window_count
        window_left += 1
        window_right += 1

    if has_wt:
        df_bool = df_no_wt.apply(lambda row: int(
            row["mutant"][1:-1]) >= global_left+1 and int(row["mutant"][1:-1]) <= global_right, axis=1)
        df_bool.loc[0] = True
        df = df.loc[df_bool]
    else:
        df = df.loc[df.apply(lambda row: int(row["mutant"][1:-1]) >=
                             global_left+1 and int(row["mutant"][1:-1]) <= global_right, axis=1)]

    # if the sequence is too short, extend it to the max_len
    if global_right - global_left + 1 < max_len:
        global_left = global_right - max_len

    return df, seq[global_left:global_right], global_left + 1


def label_row(rows, sequence, token_probs, alphabet, offset_idx):
    if rows.lower() == "wt":
        return 0.0
    s = []
    for row in rows.split(":"):
        wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
        if idx >= len(sequence):
            return 0.0
        if sequence[idx] != wt:
            print("The listed wildtype does not match the provided sequence", row)
        wt_encoded, mt_encoded = alphabet[wt], alphabet[mt]
        # add 1 for BOS
        score = token_probs[0, 1 + idx, mt_encoded] - \
            token_probs[0, 1 + idx, wt_encoded]
        score = score.item()
        s.append(score)
    return sum(s)


def read_seq_from_fasta(fasta_file):
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)


def full_sequence(seq, mutants):
    if mutants.lower() == 'wt':
        return seq
    for mutant in mutants.split(':'):
        wt, idx, mt = mutant[0], int(mutant[1:-1]) - 1, mutant[-1]
        if not (0 < idx < len(seq)):
            print(
                f"Warning: the index {idx} is out of range, seq lengths is {len(seq)}. {mutants}")
            raise ValueError
        if wt != seq[idx]:
            print(
                f"WT does not match the sequence. {mutants} -> seq[{idx}] = {seq[idx]} != {wt}]")
            raise ValueError
        seq = seq[:idx] + mt + seq[idx+1:]
    return seq


@torch.no_grad()
def main():
    args = create_parser()
    # Load the deep mutational scan
    df = pd.read_csv(args.mutant, sep=args.sep)
    sequence = read_seq_from_fasta(args.fasta)

    is_truncate = False
    offset = 1
    # truncate the sequence
    if len(sequence) >= 1022:
        # max lenth 1024
        df_truncate, sequence_truncate, offset = scan_max_mutant(df, sequence)
        is_truncate = True

    # inference for each model
    for model_path, tokenizer_path in zip(args.model_path, args.tokenizer_path):
        # if model path exists
        # if model_path in df.columns:
        #     if df[model_path].sum() != 0:
        #         print("Scored, skip!")
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # input_ids = tokenizer([sequence], return_tensors="pt")["input_ids"]
        if is_truncate:
            sequence = sequence_truncate
            origin_df = df
            df = df_truncate

        input_ids = torch.tensor([[tokenizer.cls_token_id, ] + [tokenizer.get_vocab()[each]
                                 for each in sequence] + [tokenizer.eos_token_id, ], ], dtype=torch.long)
        input_ids = input_ids.to(device)

        if args.scoring_strategy == "wt-marginals":
            logits = model(input_ids)["logits"]
            token_probs = torch.log_softmax(logits, dim=-1)
            # token_probs = torch.softmax(logits, dim=-1)
            df[model_path] = df.apply(
                lambda row: label_row(
                    row[args.mutation_col],
                    sequence,
                    token_probs,
                    tokenizer.get_vocab(),
                    offset,
                ),
                axis=1,
            )
        elif args.scoring_strategy == "masked-marginals":
            all_token_probs = []
            for i in tqdm(range(input_ids.size(1))):
                masked = input_ids.clone()
                masked[0, i] = tokenizer.mask_token_id
                logits = model(masked)["logits"]
                token_probs = torch.log_softmax(
                    model(masked.cuda())["logits"], dim=-1
                )
                all_token_probs.append(token_probs[:, i])  # vocab size
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
            df[model_path] = df.apply(
                lambda row: label_row(
                    row[args.mutation_col],
                    sequence,
                    token_probs,
                    tokenizer.get_vocab(),
                    offset,
                ),
                axis=1,
            )
    if is_truncate:
        for model_path in args.model_path:
            # merge, keep on origin not chanaged, fill df values to origin according to mutant
            origin_df = origin_df.set_index('mutant')
            origin_df[model_path] = 0.0
            origin_df.loc[df.mutant.values, model_path] = df[model_path].values
            origin_df = origin_df.reset_index()
            origin_df.to_csv(args.mutant, sep=args.sep, index=False)
    else:
        df.to_csv(args.mutant, sep=args.sep, index=False)


if __name__ == "__main__":
    main()
