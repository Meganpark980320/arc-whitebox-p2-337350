import json,glob,math,statistics as st
f=lambda c:(math.sqrt(max(0,1-c*c))+(math.pi-math.acos(c))*c)/math.pi
chi=lambda c:1-math.acos(c)/math.pi
c=[0.0]
for _ in range(15): c.append(f(c[-1]))
S=[math.prod(chi(c[k]) for k in range(l+1,16)) for l in range(16)]
R=[r for g in sorted(glob.glob('v3_net*.json')) for r in json.load(open(g))]
print(" ℓ | 평균주입: 최종/평균  S(ℓ) | 분산주입: 최종/분산 | D3: 측정  평균만예측  두통로예측  측정/두통로 | D3 분산/평균 에너지비")
for l in range(1,15):
    g=lambda m: [r for r in R if r['layer']==l and r['mode']==m]
    gm=st.mean(r['final']/r['loc_mean'] for r in g('mean'))
    gv=st.mean(r['final']/r['loc_var'] for r in g('var'))
    d=g('d3')
    meas=st.mean(r['final'] for r in d)
    p1=st.mean(r['loc_mean']*gm for r in d)
    p2=st.mean(r['loc_mean']*gm+r['loc_var']*gv for r in d)
    vr=st.mean(r['loc_var']/r['loc_mean'] for r in d)
    print(f"{l:2d} |   {gm:.3f}   {S[l]:.3f}     |   {gv:9.3e}      | {meas:.2e} {p1:.2e}  {p2:.2e}   {meas/p2:.2f}     | {vr:.3f}")
