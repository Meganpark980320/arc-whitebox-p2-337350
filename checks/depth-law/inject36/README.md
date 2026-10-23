# Noise injection in ARC's reference chain (write-up §2.6)

Repeats the read-out noise experiment of forum topic 18264 on ARC's reference implementation (`mlp_cumulant_propagation`, `factored_nonlin_kprop_k3`, SIMPLE, n = 256, 16 layers) and splits each layer's effect into the error made at that layer and the share of it that reaches layer 15.

```bash
python inject36_v1.py <path-to-mlp_cumulant_propagation/src> 256 1 inj36_net0.json 0   # one network; we ran 0–7
python an36_v1.py inj36_net[0-7].json > an36_v1.out.txt
```

Needs torch (CPU is enough; about 10 s per network with gradients off). `an36_v1.out.txt` is our output for networks 0–7.
