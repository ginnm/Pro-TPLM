In this directory, we provide the code for the ESM-2 and ESM-2(Homo) baseline used in the paper. 

## Usage
```
python esm2.py --model_path facebook/esm2_t33_650M_UR50D \
--tokenizer_path facebook/esm2_t33_650M_UR50D  \
--fasta example/GFP_AEQVI_Sarkisyan_2016.fasta \
--mutant example/GFP_AEQVI_Sarkisyan_2016.csv
```

## Available Models (Updating)

| Model Name | Description | Link |
|------------|-------------|------|
| ESM-2-official | ESM-2-official | [Link](https://huggingface.co/facebook/esm2_t33_650M_UR50D) |
| AI4Protein/esm2_for_protein_gym   | ESM-2 tuned on homology sequences of ProteinGYM | [Link](https://huggingface.co/AI4Protein/esm2_for_protein_gym) |
| AI4Protein/esm2_for_thermal_db    | ESM-2 tuned on homology sequences of ThermalDB | [Link](https://huggingface.co/AI4Protein/esm2_for_thermal_db) |
| AI4Protein/esm2_ft_tgod4k_1    | ESM-2 tuned on homology sequences of DNA Polymerase | [Link](https://huggingface.co/AI4Protein/esm2_ft_tgod4k_1) |
| AI4Protein/esm2_ft_tgod4k_2    | ESM-2 tuned on homology sequences of DNA Polymerase  | [Link](https://huggingface.co/AI4Protein/esm2_ft_tgod4k_2) |
| AI4Protein/esm2_ft_tgod4k_3    | ESM-2 tuned on homology sequences of DNA Polymerase | [Link](https://huggingface.co/AI4Protein/esm2_ft_tgod4k_3) |
| AI4Protein/esm2_ft_tgod4k_4    | ESM-2 tuned on homology sequences of DNA Polymerase | [Link](https://huggingface.co/AI4Protein/esm2_ft_tgod4k_4) |
| AI4Protein/esm2_ft_tgod4k_5    | ESM-2 tuned on homology sequences of DNA Polymerase | [Link](https://huggingface.co/AI4Protein/esm2_ft_tgod4k_5) |