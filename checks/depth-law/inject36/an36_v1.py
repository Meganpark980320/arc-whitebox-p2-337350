import json, math, sys
f=lambda c:(math.sqrt(max(0,1-c*c))+(math.pi-math.acos(c))*c)/math.pi
chi=lambda c:1-math.acos(c)/math.pi
L=16; c=[0.0]
for _ in range(L-1): c.append(f(c[-1]))
S=[math.prod(chi(c[k]) for k in range(l+1,L)) for l in range(L)]
R=[]
for a in sys.argv[1:]: R+=json.load(open(a))
nets=sorted({r['net'] for r in R})
win=[(0,5),(6,9),(10,12),(13,14),(15,15)]
for kinds in ('D3','D21','D3+D21'):
    print(f"\n=== {kinds} (σ=1e-3, n=256, 망 {len(nets)}개 평균)")
    fin=[sum(r['dmse'][15] for r in R if r['kinds']==kinds and r['layer']==l)/len(nets) for l in range(L)]
    # 현지: D3는 넣은 층 평균, D21은 다음 층 평균에 처음 나타남
    loc=[sum(r['dmse'][l] for r in R if r['kinds']==kinds and r['layer']==l)/len(nets) for l in range(L)]
    loc1=[sum(r['dmse'][min(l+1,15)] for r in R if r['kinds']==kinds and r['layer']==l)/len(nets) for l in range(L)]
    print(" ℓ   최종ΔMSE   현지ΔMSE(ℓ)  최종/현지   S(ℓ)   | 현지(ℓ+1) 최종/현지(ℓ+1) S(ℓ+1)")
    for l in range(L):
        r0=fin[l]/loc[l] if loc[l]>0 else float('nan')
        r1=fin[l]/loc1[l] if loc1[l]>0 else float('nan')
        print(f"{l:2d}  {fin[l]:.3e}  {loc[l]:.3e}   {r0:7.3f}  {S[l]:.3f}  | {loc1[l]:.3e}  {r1:7.3f}  {S[min(l+1,15)]:.3f}")
    base=sum(fin[0:6])/6
    print(" 구간 민감도(최종, 0–5 기준):", [round(sum(fin[a:b+1])/(b-a+1)/base,1) for a,b in win], " 18264 측정: [1, 9, 24, 30, 36]")
