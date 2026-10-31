#### Training step (peak memory · patches per second)

| Aggregator | N = 1,024 | N = 2,048 | N = 4,096 | N = 8,192 | N = 16,384 | N = 32,768 | N = 65,536 | N = 131,072 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mean pooling + MLP | 0.04 GB · 268k/s | 0.06 GB · 460k/s | 0.09 GB · 814k/s | 0.15 GB · 1,967k/s | 0.26 GB · 3,252k/s | 0.45 GB · 9,592k/s | 0.84 GB · 12,613k/s | 1.65 GB · 13,445k/s |
| ABMIL | 0.05 GB · 184k/s | 0.07 GB · 245k/s | 0.10 GB · 738k/s | 0.17 GB · 1,458k/s | 0.31 GB · 2,592k/s | 0.56 GB · 5,354k/s | 1.09 GB · 6,711k/s | 2.16 GB · 6,914k/s |
| Multi-head ABMIL | 0.10 GB · 58k/s | 0.13 GB · 192k/s | 0.19 GB · 373k/s | 0.31 GB · 502k/s | 0.55 GB · 739k/s | 1.02 GB · 1,872k/s | 1.97 GB · 2,535k/s | 3.88 GB · 2,636k/s |
| Perceiver | 0.48 GB · 29k/s | 0.49 GB · 62k/s | 0.51 GB · 123k/s | 0.57 GB · 221k/s | 0.75 GB · 447k/s | 1.13 GB · 923k/s | 1.97 GB · 1,444k/s | 3.64 GB · 2,218k/s |
| TransMIL | 0.18 GB · 27k/s | 0.26 GB · 59k/s | 0.39 GB · 120k/s | 0.66 GB · 212k/s | 1.22 GB · 297k/s | 2.35 GB · 317k/s | 4.57 GB · 298k/s | 9.08 GB · 280k/s |
| ViT, 2-D ALiBi | 0.58 GB · 29k/s | 0.76 GB · 56k/s | 1.56 GB · 45k/s | 4.71 GB · 27k/s | 17.57 GB · 15k/s | OOM | — | — |
| Mamba-2 MIL | 0.11 GB · 31k/s | 0.15 GB · 64k/s | 0.23 GB · 96k/s | 0.39 GB · 208k/s | 0.70 GB · 447k/s | 1.34 GB · 928k/s | 2.63 GB · 1,078k/s | 5.20 GB · 1,161k/s |
| Bidirectional Mamba | 0.19 GB · 18k/s | 0.26 GB · 40k/s | 0.39 GB · 82k/s | 0.65 GB · 154k/s | 1.16 GB · 307k/s | 2.22 GB · 581k/s | 4.32 GB · 606k/s | 8.52 GB · 613k/s |

#### Inference (peak memory · patches per second)

| Aggregator | N = 1,024 | N = 2,048 | N = 4,096 | N = 8,192 | N = 16,384 | N = 32,768 | N = 65,536 | N = 131,072 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mean pooling + MLP | 0.03 GB · 1,248k/s | 0.04 GB · 2,545k/s | 0.06 GB · 5,053k/s | 0.11 GB · 13,502k/s | 0.19 GB · 18,412k/s | 0.33 GB · 19,325k/s | 0.64 GB · 21,784k/s | 1.26 GB · 22,898k/s |
| ABMIL | 0.02 GB · 870k/s | 0.03 GB · 1,777k/s | 0.05 GB · 3,451k/s | 0.09 GB · 7,289k/s | 0.17 GB · 13,395k/s | 0.33 GB · 15,862k/s | 0.64 GB · 17,517k/s | 1.27 GB · 9,665k/s |
| Multi-head ABMIL | 0.04 GB · 249k/s | 0.04 GB · 512k/s | 0.06 GB · 1,024k/s | 0.10 GB · 3,014k/s | 0.18 GB · 6,011k/s | 0.34 GB · 8,075k/s | 0.65 GB · 8,355k/s | 1.27 GB · 8,762k/s |
| Perceiver | 0.16 GB · 128k/s | 0.17 GB · 255k/s | 0.18 GB · 500k/s | 0.24 GB · 1,092k/s | 0.37 GB · 2,158k/s | 0.64 GB · 4,366k/s | 1.17 GB · 5,688k/s | 2.23 GB · 6,151k/s |
| TransMIL | 0.07 GB · 74k/s | 0.10 GB · 207k/s | 0.16 GB · 409k/s | 0.29 GB · 769k/s | 0.54 GB · 1,102k/s | 1.05 GB · 1,188k/s | 2.04 GB · 1,105k/s | 4.05 GB · 1,008k/s |
| ViT, 2-D ALiBi | 0.25 GB · 197k/s | 0.45 GB · 215k/s | 1.24 GB · 144k/s | 4.44 GB · 84k/s | 17.26 GB · 45k/s | OOM | — | — |
| Mamba-2 MIL | 0.04 GB · 41k/s | 0.06 GB · 124k/s | 0.10 GB · 120k/s | 0.17 GB · 240k/s | 0.31 GB · 541k/s | 0.59 GB · 1,094k/s | 1.17 GB · 1,523k/s | 2.31 GB · 2,483k/s |
| Bidirectional Mamba | 0.07 GB · 138k/s | 0.10 GB · 269k/s | 0.15 GB · 532k/s | 0.24 GB · 905k/s | 0.44 GB · 1,343k/s | 0.84 GB · 1,923k/s | 1.62 GB · 1,934k/s | 3.19 GB · 1,947k/s |

