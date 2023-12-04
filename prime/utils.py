from Bio import SeqIO


def read_seq(fasta_file):
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)


def mutant_filter(df, mutant_site=0):
    if mutant_site:
        trg_df = df.loc[
            df.apply(lambda row: len(row["mutant"].split(":")) == mutant_site, axis=1)
        ]
    else:
        trg_df = df
    return trg_df


def scan_max_mutant(df, seq, max_len=4096):
    if len(seq) <= max_len:
        return df, seq, 1

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
        seq_left_to_right = seq[min_pos : min_pos + max_len]
        # contain the last mutation residue
        seq_right_to_left = seq[max_pos - max_len + 1 : max_pos + 1]

        # select the longer sequence
        if len(seq_left_to_right) > len(seq_right_to_left):
            seq = seq_left_to_right
            offset = min_pos + 1
            if has_wt:
                df_bool = df_no_wt.apply(
                    lambda row: int(row["mutant"][1:-1]) >= offset
                    and int(row["mutant"][1:-1]) < (min_pos + max_len + 1),
                    axis=1,
                )
                df_bool.loc[0] = True
                df = df.loc[df_bool]
            else:
                df = df.loc[
                    df.apply(
                        lambda row: int(row["mutant"][1:-1]) >= offset
                        and int(row["mutant"][1:-1]) < (min_pos + max_len + 1),
                        axis=1,
                    )
                ]
        else:
            seq = seq_right_to_left
            offset = max_pos - max_len + 2
            if has_wt:
                df_bool = df_no_wt.apply(
                    lambda row: int(row["mutant"][1:-1]) >= offset
                    and int(row["mutant"][1:-1]) < max_pos + 2,
                    axis=1,
                )
                df_bool.loc[0] = True
                df = df.loc[df_bool]
            else:
                df = df.loc[
                    df.apply(
                        lambda row: int(row["mutant"][1:-1]) >= offset
                        and int(row["mutant"][1:-1]) < max_pos + 2,
                        axis=1,
                    )
                ]
        return df, seq, offset

    global_count = 0
    global_left, global_right = min_pos, max_len
    window_left, window_right = min_pos, max_len
    # scan the whole sequence
    while window_right < len(seq):
        window_count = len(
            [pos for pos in mutant_pos if pos >= window_left and pos < window_right]
        )
        if window_count > global_count:
            global_left, global_right = window_left, window_right
            global_count = window_count
        window_left += 1
        window_right += 1

    if has_wt:
        df_bool = df_no_wt.apply(
            lambda row: int(row["mutant"][1:-1]) >= global_left + 1
            and int(row["mutant"][1:-1]) <= global_right,
            axis=1,
        )
        df_bool.loc[0] = True
        df = df.loc[df_bool]
    else:
        df = df.loc[
            df.apply(
                lambda row: int(row["mutant"][1:-1]) >= global_left + 1
                and int(row["mutant"][1:-1]) <= global_right,
                axis=1,
            )
        ]

    # if the sequence is too short, extend it to the max_len
    if global_right - global_left + 1 < max_len:
        global_left = global_right - max_len

    return df, seq[global_left:global_right], global_left + 1


def score_mutant(mutant, sequence, logits, vocab, offset):
    if mutant.lower() == "wt":
        return 0.0
    wt, idx, mt = mutant[0], int(mutant[1:-1]) - offset, mutant[-1]
    if idx >= len(sequence):
        return 0.0
    if sequence[idx] != wt:
        print("The listed wildtype does not match the provided sequence", mutant)
    wt_encoded, mt_encoded = vocab[wt], vocab[mt]
    score = logits[1 + idx, mt_encoded] - logits[1 + idx, wt_encoded]
    score = score.item()
    return score
