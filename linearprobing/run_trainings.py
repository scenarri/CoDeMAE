import subprocess


for seed in [0, 1, 2, 3, 4]:
    # Linear probing  SwinSSL
    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B13 --modality RGB --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50  --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B13 --modality RGB --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B13 --modality RGB --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B13 --modality RGB --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B13 --modality RGB --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B13 --modality RGB --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B13 --modality SAR --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B13 --modality SAR --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B13 --modality SAR --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B13 --modality SAR --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B13 --modality SAR --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B13 --modality SAR --backbone SwinSSL --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    # Linear probing  DINOMM
    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B12 --modality RGB --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50  --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B12 --modality RGB --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B12 --modality RGB --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B12 --modality RGB --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B12 --modality RGB --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B12 --modality RGB --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B12 --modality SAR --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B12 --modality SAR --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B12 --modality SAR --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B12 --modality SAR --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B12 --modality SAR --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B12 --modality SAR --backbone DINOMM --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    # Linear probing  DeCUR
    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B13 --modality RGB --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50  --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B13 --modality RGB --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B13 --modality RGB --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B13 --modality RGB --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B13 --modality RGB --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B13 --modality RGB --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B13 --modality SAR --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B13 --modality SAR --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B13 --modality SAR --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B13 --modality SAR --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B13 --modality SAR --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B13 --modality SAR --backbone DeCUR --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    # Linear probing  FGMAE
    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B13 --modality RGB --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50  --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B13 --modality RGB --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B13 --modality RGB --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B13 --modality RGB --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B13 --modality RGB --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B13 --modality RGB --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B13 --modality SAR --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B13 --modality SAR --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B13 --modality SAR --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B13 --modality SAR --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B13 --modality SAR --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B13 --modality SAR --backbone FGMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    # Linear probing  SatViT
    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B13 --modality RGB --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50  --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B13 --modality RGB --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B13 --modality RGB --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B13 --modality RGB --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B13 --modality RGB --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B13 --modality RGB --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B13 --modality SAR --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B13 --modality SAR --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B13 --modality SAR --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B13 --modality SAR --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B13 --modality SAR --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B13 --modality SAR --backbone SatViT --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    # Linear probing  CROMA
    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B12 --modality RGB --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50  --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B12 --modality RGB --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B12 --modality RGB --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B12 --modality RGB --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B12 --modality RGB --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B12 --modality RGB --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B12 --modality SAR --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B12 --modality SAR --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B12 --modality SAR --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B12 --modality SAR --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B12 --modality SAR --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B12 --modality SAR --backbone CROMA --input_size 120 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    # Linear DOFA
    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands RGB --modality RGB --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50  --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands RGB --modality RGB --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands RGB --modality RGB --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B13 --modality RGB --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B13 --modality RGB --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B13 --modality RGB --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    #
    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands B13 --modality SAR --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands B13 --modality SAR --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands B13 --modality SAR --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands B13 --modality SAR --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands B13 --modality SAR --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands B13 --modality SAR --backbone DOFA --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    # Linear probing  MARS
    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands RGB --modality RGB --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50  --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands RGB --modality RGB --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands RGB --modality RGB --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands RGB --modality RGB --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands RGB --modality RGB --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands RGB --modality RGB --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands RGB --modality SAR --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands RGB --modality SAR --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands RGB --modality SAR --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands RGB --modality SAR --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands RGB --modality SAR --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands RGB --modality SAR --backbone MARS --input_size 256 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    # Linear probing  itpn
    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands RGB --modality RGB --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50  --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands RGB --modality RGB --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands RGB --modality RGB --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands RGB --modality RGB --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands RGB --modality RGB --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands RGB --modality RGB --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)

    subprocess.run(f'python classification.py --seed {seed} --dataset PIE --bands RGB --modality SAR --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DDHR-SK --bands RGB --modality SAR --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset WHU --bands RGB --modality SAR --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset DFC20 --bands RGB --modality SAR --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset BEN --bands RGB --modality SAR --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
    subprocess.run(f'python classification.py --seed {seed} --dataset EuroSat --bands RGB --modality SAR --backbone CoDeMAE --input_size 224 --linprob --amp --lr 0.5 --wd 0. --warmup_epochs 0 --epochs 50   --subset 10', shell=True)
