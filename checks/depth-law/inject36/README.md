# Noise injection in ARC's reference chain (write-up §2.6)

Repeats the read-out noise experiment of forum topic 18264 on ARC's reference implementation (`mlp_cumulant_propagation`, `factored_nonlin_kprop_k3`, SIMPLE, n = 256, 16 layers) and splits each layer's effect into the error made at that layer and the share of it that reaches layer 15.

```bash
python inject36_v1.py <path-to-mlp_cumulant_propagation/src> 256 1 inj36_net0.json 0   # one network; we ran 0–7
python an36_v1.py inj36_net[0-7].json > an36_v1.out.txt
```

Needs torch (CPU is enough; about 10 s per network with gradients off). `an36_v1.out.txt` is our output for networks 0–7.

## Channel split and width scan (`channels/`, write-up §2.6)

- `inject36_v2.py`: D3 noise only; splits each layer's local mean error into the incoming-mean direction and the rest and propagates both through the network's first-order mean map.
- `inject36_v3.py`: three injections per layer (into the mean, into the variance, and into the D3 read-out), recording local mean and variance errors and the final error. `an36_v3.py` prints the two-channel prediction (`an36_v3.out.txt`).
- `inject36_v4.py`: as v3 with a mode list; we ran `mean` at n = 128 and 512 for the width scan (`v4_n128_*`, `v4_n512_*`).
