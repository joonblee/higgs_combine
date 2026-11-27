"""Plot expected coupling contours from a Combine limit JSON."""
import argparse
import json
from pathlib import Path

import fitz
import matplotlib.pyplot as plt, numpy as np
from matplotlib.collections import LineCollection
plt.rcParams.update({'font.family':'Liberation Sans','mathtext.fontset':'custom','mathtext.rm':'Liberation Sans','mathtext.it':'Liberation Sans:italic','mathtext.bf':'Liberation Sans:bold','pdf.fonttype':42})
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("limits", type=Path, help="Mass-keyed JSON with exp0 limits")
parser.add_argument("output", type=Path, help="Output PDF")
args = parser.parse_args()
limits = {float(m): row["exp0"] for m, row in json.loads(args.limits.read_text()).items()}

f=plt.figure(figsize=(567/72,544/72)); a=f.add_axes([.156,.15,.81,.774]);a.set(xscale='log',yscale='log',xlim=(.004,1),ylim=(.004,1))
a.set_xlabel(r'$g_{bb}$',fontsize=21.312,loc='right');a.set_ylabel(r'$g_{\mu\mu}$',fontsize=18.24,loc='top')
a.tick_params(which='both',direction='in',top=True,right=True,labelsize=18.24);a.tick_params(which='major',length=8);a.tick_params(which='minor',length=4)
f.canvas.draw()
for m in (12, 40, 70):
 L = limits[m]
 x=(4.7/m)**2; A=4*np.pi*L;c=3*(1-x)*np.sqrt(1-4*x)
 b=np.geomspace(np.sqrt(2*A/(1-c*A)),1,6000);u=np.sqrt(c*A/(1-2*A/b**2))
 line,=a.plot(b,u,label=rf"$m(Z') = {m}\ \mathrm{{GeV}}$",lw=2)
 pts=a.transData.transform(np.c_[b,u]);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
 ticks=np.arange(1,s[-1],6);start=np.stack([np.interp(ticks,s,v) for v in pts.T],axis=1)

 assert np.allclose(b**2 / (4*np.pi) / (2+c*(b/u)**2), L, rtol=1e-12, atol=0)
 for k in range(40):
  delta = f.dpi * 2 / 2.54 / np.sqrt(2) / 40
  endpoints = np.vstack([start+k*delta, start+(k+1)*delta])
  segments = a.transData.inverted().transform(endpoints).reshape(2,-1,2).transpose(1,0,2)
  a.add_collection(LineCollection(segments, colors=line.get_color(), alpha=.32*(1-k/39), linewidths=.62))

a.set_xticks([.01,.1,1],labels=['0.01','0.1','1'],fontsize=17.508)
a.set_yticks([.01,.1,1],labels=['0.01','0.1','1'])
a.legend(loc='upper right',fontsize=15.96,frameon=False)
a.yaxis.labelpad += -6
f.canvas.draw()
left = a.yaxis.label.get_window_extent().x0 * 72 / f.dpi
f.text(left/567, 517/544, 'CMS', fontsize=22.8, weight='bold')
f.text((left+55.66)/567, 517/544, 'Preliminary', fontsize=15.96, style='italic')
f.text(547.7/567, 517/544, r'138 fb$^{-1}$ (13 TeV) + 62 fb$^{-1}$ (13.6 TeV)', ha='right', fontsize=15.96)

from io import BytesIO
buffer = BytesIO()
f.savefig(buffer, format='pdf')
with fitz.open(stream=buffer.getvalue(), filetype='pdf') as document:
    document[0].set_cropbox(fitz.Rect(18, 0, 567, 521))
    document.save(args.output, deflate=True)
